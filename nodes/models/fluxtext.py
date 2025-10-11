import contextlib
import contextvars
from typing import Optional, Union, Dict, Any, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
from comfy.ldm.common_dit import pad_to_patch_size
from comfy.ldm.flux.model import Flux as FluxInnerModel
from comfy.ldm.flux.layers import (
    SingleStreamBlock,
    DoubleStreamBlock,
    apply_mod,
    attention,
    timestep_embedding,
)
from comfy.model_base import Flux as FluxModel
from comfy.model_patcher import ModelPatcher
from comfy.weight_adapter import LoRAAdapter, lora
from comfy.ops import manual_cast
import comfy.utils
import comfy.lora
import comfy.lora_convert
import folder_paths


is_lora_enabled: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "is_lora_enabled", default=False
)


@contextlib.contextmanager
def enable_lora():
    """一个上下文管理器，用于启用 LoRA 特定代码块。"""
    token = is_lora_enabled.set(True)

    try:
        yield
    finally:
        is_lora_enabled.reset(token)


LoadedLoraType = Dict[Union[str, Tuple[str, Tuple[int, int, int]]], LoRAAdapter]


def patched_linear_forward(self: manual_cast.Linear, x: torch.Tensor) -> torch.Tensor:
    weight = self.weight.to(x.dtype, copy=True)
    bias = self.bias.to(x.dtype, copy=True) if self.bias is not None else None

    if (
        not is_lora_enabled.get()
        or not hasattr(self, "lora_a")
        or not hasattr(self, "lora_b")
    ):
        return F.linear(x, weight, bias)

    for lora_a, lora_b, scale, index in zip(
        self.lora_a, self.lora_b, self.lora_scale, self.lora_index
    ):
        diff = lora_a @ lora_b
        if scale != 1.0:
            diff = diff * scale
        if index is not None:
            weight[
                (slice(None),) * index[0] + (slice(index[1], index[1] + index[2]),)
            ] += diff
        else:
            weight += diff

    return F.linear(x, weight, bias)


def install_loaded_lora(
    model: ModelPatcher, loaded: LoadedLoraType, lora_scale: float = 1.0
) -> None:
    assert isinstance(
        model.model.diffusion_model, FluxInnerModel
    ), "Model must be a Flux model"
    if hasattr(model.model, "installed_fluxtext_lora"):
        return  # Already installed
    module_dict = {k: v for k, v in model.model.named_modules()}
    patched_module = set()

    for k, v in loaded.items():
        name = k if isinstance(k, str) else k[0]
        index = k[1] if isinstance(k, tuple) else None
        # Remove trailing '.weight' in name if present
        clean_name = name[:-7] if name.endswith(".weight") else name
        module = module_dict.get(clean_name)

        if module is None:
            raise ValueError(f"Module {clean_name} not found in model.")
        if not isinstance(module, manual_cast.Linear):
            raise ValueError(
                f"Module {clean_name} is a {type(module)}, expected comfy.ops.manual_cast.Linear."
            )

        if not hasattr(module, "lora_a"):
            module.lora_a = nn.ParameterList()
        if not hasattr(module, "lora_b"):
            module.lora_b = nn.ParameterList()
        if not hasattr(module, "lora_scale"):
            module.lora_scale = []
        if not hasattr(module, "lora_index"):
            module.lora_index = []

        module.lora_a.append(v.weights[0])
        module.lora_b.append(v.weights[1])
        module.lora_scale.append(lora_scale)
        module.lora_index.append(index)

        if not clean_name in patched_module:
            module.forward = patched_linear_forward.__get__(module, manual_cast.Linear)
            patched_module.add(clean_name)

    model.model.installed_fluxtext_lora = True


def patched_single_stream_block_forward(
    self: SingleStreamBlock,
    x: torch.Tensor,
    glyph: torch.Tensor,
    vec: torch.Tensor,
    glyph_vec: torch.Tensor,
    pe: torch.Tensor,
    attn_mask=None,
    modulation_dims=None,
):
    mod, _ = self.modulation(vec)
    qkv, mlp = torch.split(
        self.linear1(
            apply_mod(self.pre_norm(x), (1 + mod.scale), mod.shift, modulation_dims)
        ),
        [3 * self.hidden_size, self.mlp_hidden_dim],
        dim=-1,
    )
    q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(
        2, 0, 3, 1, 4
    )
    q, k = self.norm(q, k, v)

    with enable_lora():
        glyph_mod, _ = self.modulation(glyph_vec)
        glyph_qkv, glyph_mlp = torch.split(
            self.linear1(
                apply_mod(
                    self.pre_norm(glyph),
                    (1 + glyph_mod.scale),
                    glyph_mod.shift,
                    modulation_dims,
                ),
            ),
            [3 * self.hidden_size, self.mlp_hidden_dim],
            dim=-1,
        )
        glyph_q, glyph_k, glyph_v = glyph_qkv.view(
            glyph_qkv.shape[0], glyph_qkv.shape[1], 3, self.num_heads, -1
        ).permute(2, 0, 3, 1, 4)
        glyph_q, glyph_k = self.norm(glyph_q, glyph_k, glyph_v)

    # compute attention
    attn = attention(
        torch.cat((q, glyph_q), dim=2),
        torch.cat((k, glyph_k), dim=2),
        torch.cat((v, glyph_v), dim=2),
        pe=pe,
        mask=attn_mask,
    )

    # compute activation in mlp stream, cat again and run second linear layer
    output = self.linear2(torch.cat((attn[:, : x.shape[1]], self.mlp_act(mlp)), 2))
    x += apply_mod(output, mod.gate, None, modulation_dims)
    if x.dtype == torch.float16:
        x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)

    with enable_lora():
        glyph_output = self.linear2(
            torch.cat((attn[:, x.shape[1] :], self.mlp_act(glyph_mlp)), 2),
        )
        glyph += apply_mod(glyph_output, mod.gate, None, modulation_dims)
        if glyph.dtype == torch.float16:
            glyph = torch.nan_to_num(glyph, nan=0.0, posinf=65504, neginf=-65504)

    return x, glyph


def patched_double_stream_block_forward(
    self: DoubleStreamBlock,
    img: torch.Tensor,
    txt: torch.Tensor,
    glyph: torch.Tensor,
    vec: torch.Tensor,
    glyph_vec: torch.Tensor,
    pe: torch.Tensor,
    attn_mask=None,
    modulation_dims_img=None,
    modulation_dims_txt=None,
):
    img_mod1, img_mod2 = self.img_mod(vec)
    txt_mod1, txt_mod2 = self.txt_mod(vec)
    with enable_lora():
        glyph_mod1, glyph_mod2 = self.img_mod(glyph_vec)

    # prepare image for attention
    img_modulated = self.img_norm1(img)
    img_modulated = apply_mod(
        img_modulated, (1 + img_mod1.scale), img_mod1.shift, modulation_dims_img
    )
    img_qkv = self.img_attn.qkv(img_modulated)
    img_q, img_k, img_v = img_qkv.view(
        img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1
    ).permute(2, 0, 3, 1, 4)
    img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

    # prepare txt for attention
    txt_modulated = self.txt_norm1(txt)
    txt_modulated = apply_mod(
        txt_modulated, (1 + txt_mod1.scale), txt_mod1.shift, modulation_dims_txt
    )
    txt_qkv = self.txt_attn.qkv(txt_modulated)
    txt_q, txt_k, txt_v = txt_qkv.view(
        txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1
    ).permute(2, 0, 3, 1, 4)
    txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

    with enable_lora():
        glyph_modulated = self.img_norm1(glyph)
        glyph_modulated = apply_mod(
            glyph_modulated,
            (1 + glyph_mod1.scale),
            glyph_mod1.shift,
            modulation_dims_img,
        )
        glyph_qkv = self.img_attn.qkv(glyph_modulated)
        glyph_q, glyph_k, glyph_v = glyph_qkv.view(
            glyph_qkv.shape[0], glyph_qkv.shape[1], 3, self.num_heads, -1
        ).permute(2, 0, 3, 1, 4)
        glyph_q, glyph_k = self.img_attn.norm(glyph_q, glyph_k, glyph_v)

    # run actual attention
    attn = attention(
        torch.cat((txt_q, img_q, glyph_q), dim=2),
        torch.cat((txt_k, img_k, glyph_k), dim=2),
        torch.cat((txt_v, img_v, glyph_v), dim=2),
        pe=pe,
        mask=attn_mask,
    )

    txt_attn, img_attn, glyph_attn = (
        attn[:, : txt.shape[1]],
        attn[:, txt.shape[1] : txt.shape[1] + img.shape[1]],
        attn[:, txt.shape[1] + img.shape[1] :],
    )

    # calculate the img bloks
    img = img + apply_mod(
        self.img_attn.proj(img_attn), img_mod1.gate, None, modulation_dims_img
    )
    img = img + apply_mod(
        self.img_mlp(
            apply_mod(
                self.img_norm2(img),
                (1 + img_mod2.scale),
                img_mod2.shift,
                modulation_dims_img,
            )
        ),
        img_mod2.gate,
        None,
        modulation_dims_img,
    )
    if img.dtype == torch.float16:
        img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

    # calculate the txt bloks
    txt += apply_mod(
        self.txt_attn.proj(txt_attn), txt_mod1.gate, None, modulation_dims_txt
    )
    txt += apply_mod(
        self.txt_mlp(
            apply_mod(
                self.txt_norm2(txt),
                (1 + txt_mod2.scale),
                txt_mod2.shift,
                modulation_dims_txt,
            )
        ),
        txt_mod2.gate,
        None,
        modulation_dims_txt,
    )
    if txt.dtype == torch.float16:
        txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)

    with enable_lora():
        # calculate the glyph blocks
        glyph = glyph + apply_mod(
            self.img_attn.proj(glyph_attn), glyph_mod1.gate, None, modulation_dims_img
        )
        glyph = glyph + apply_mod(
            self.img_mlp(
                apply_mod(
                    self.img_norm2(glyph),
                    (1 + glyph_mod2.scale),
                    glyph_mod2.shift,
                    modulation_dims_img,
                )
            ),
            glyph_mod2.gate,
            None,
            modulation_dims_img,
        )
        if glyph.dtype == torch.float16:
            glyph = torch.nan_to_num(glyph, nan=0.0, posinf=65504, neginf=-65504)

    return img, txt, glyph


def patched_transformer_forward(
    self: FluxInnerModel,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    glyph: torch.Tensor,
    glyph_ids: torch.Tensor,
    timesteps: torch.Tensor,
    y: torch.Tensor,
    guidance: torch.Tensor = None,
    control=None,
    transformer_options={},
    attn_mask: torch.Tensor = None,
):
    if y is None:
        y = torch.zeros(
            (img.shape[0], self.params.vec_in_dim), device=img.device, dtype=img.dtype
        )

    if img.ndim != 3 or txt.ndim != 3:
        raise ValueError("Input img and txt tensors must have 3 dimensions.")

    # running on sequences img
    img = self.img_in(img)
    vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
    if self.params.guidance_embed:
        if guidance is not None:
            vec = vec + self.guidance_in(
                timestep_embedding(guidance, 256).to(img.dtype)
            )

    vec = vec + self.vector_in(y[:, : self.params.vec_in_dim])
    txt = self.txt_in(txt)

    with enable_lora():
        glyph = self.img_in(glyph)
        # XXX It seems that timesteps here are mistakenly set to zero when training the LoRA ...
        glyph_vec = self.time_in(
            timestep_embedding(torch.zeros_like(timesteps), 256).to(glyph.dtype)
        )
        if self.params.guidance_embed and guidance is not None:
            glyph_vec = glyph_vec + self.guidance_in(
                timestep_embedding(guidance, 256).to(glyph.dtype)
            )
        glyph_vec = glyph_vec + self.vector_in(y[:, : self.params.vec_in_dim])

    if img_ids is not None:
        ids = torch.cat((txt_ids, img_ids, glyph_ids), dim=1)
        pe = self.pe_embedder(ids)
    else:
        pe = None

    for i, block in enumerate(self.double_blocks):
        img, txt, glyph = block(
            img=img,
            txt=txt,
            glyph=glyph,
            vec=vec,
            glyph_vec=glyph_vec,
            pe=pe,
            attn_mask=attn_mask,
        )

        if control is not None:  # Controlnet
            control_i = control.get("input")
            if i < len(control_i):
                add = control_i[i]
                if add is not None:
                    img += add

    if img.dtype == torch.float16:
        img = torch.nan_to_num(img, nan=0.0, posinf=65504, neginf=-65504)

    img = torch.cat((txt, img), 1)

    for i, block in enumerate(self.single_blocks):
        img, glyph = block(
            img, glyph=glyph, vec=vec, glyph_vec=glyph_vec, pe=pe, attn_mask=attn_mask
        )

        if control is not None:  # Controlnet
            control_o = control.get("output")
            if i < len(control_o):
                add = control_o[i]
                if add is not None:
                    img[:, txt.shape[1] :, ...] += add

    img = img[:, txt.shape[1] :, ...]

    img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
    return img


def patched_flux_forward(
    self: FluxInnerModel,
    x,
    timestep,
    context,
    y=None,
    guidance=None,
    ref_latents=None,
    control=None,
    transformer_options={},
    **kwargs,
):
    bs, c, h_orig, w_orig = x.shape
    patch_size = self.patch_size

    h_len = (h_orig + (patch_size // 2)) // patch_size
    w_len = (w_orig + (patch_size // 2)) // patch_size
    img, img_ids = self.process_img(x)
    img_tokens = img.shape[1]
    if ref_latents is None:
        raise ValueError(
            "Reference latents is required for Flux Text model. Please pass glyph latents from ReferenceLatents node."
        )
    glyph = x.clone()
    glyph[:, :16] = ref_latents[0]
    glyph, glyph_ids = self.process_img(glyph)

    txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
    out = self.forward_orig(
        img,
        img_ids,
        context,
        txt_ids,
        glyph,
        glyph_ids,
        timestep,
        y,
        guidance,
        control,
        transformer_options,
        attn_mask=kwargs.get("attention_mask", None),
    )
    out = out[:, :img_tokens]
    return rearrange(
        out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h_len, w=w_len, ph=2, pw=2
    )[:, :, :h_orig, :w_orig]


class FluxTextLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "duanyll/models"

    def load_lora(self, model: ModelPatcher, lora_name: str, strength_model: float):
        model_type_str = str(type(model.model.model_config).__name__)
        if "Flux" not in model_type_str:
            raise Exception(
                f"Attempted to patch a {model_type_str} model. PhotoDoddle is only compatible with Flux models."
            )

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        key_map = comfy.lora.model_lora_keys_unet(model.model, {})
        lora = comfy.lora_convert.convert_lora(lora)
        loaded = comfy.lora.load_lora(lora, key_map)
        install_loaded_lora(model, loaded, lora_scale=strength_model)

        # Apply monkey patches
        inner_model: FluxInnerModel = model.model.diffusion_model
        inner_model.forward = patched_flux_forward.__get__(inner_model, FluxInnerModel)
        inner_model.forward_orig = patched_transformer_forward.__get__(
            inner_model, FluxInnerModel
        )
        for block in inner_model.double_blocks:
            block.forward = patched_double_stream_block_forward.__get__(
                block, DoubleStreamBlock
            )
        for block in inner_model.single_blocks:
            block.forward = patched_single_stream_block_forward.__get__(
                block, SingleStreamBlock
            )

        return (model,)
