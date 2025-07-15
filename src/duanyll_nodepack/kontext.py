import torch
from einops import repeat, rearrange
from comfy.ldm.flux.model import Flux as FluxInnerModel


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
    **kwargs
):
    bs, c, h_orig, w_orig = x.shape
    patch_size = self.patch_size

    h_len = (h_orig + (patch_size // 2)) // patch_size
    w_len = (w_orig + (patch_size // 2)) // patch_size
    img, img_ids = self.process_img(x)
    img_tokens = img.shape[1]
    if ref_latents is not None:
        for i, ref in enumerate(ref_latents):
            kontext, kontext_ids = self.process_img(
                ref, index=i + 1, h_offset=0, w_offset=0
            )
            img = torch.cat([img, kontext], dim=1)
            img_ids = torch.cat([img_ids, kontext_ids], dim=1)

    txt_ids = torch.zeros((bs, context.shape[1], 3), device=x.device, dtype=x.dtype)
    out = self.forward_orig(
        img,
        img_ids,
        context,
        txt_ids,
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


class FluxKontextTrue3DPE:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }
        
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "duanyll"
    
    def patch(self, model):
        model_type_str = str(type(model.model.model_config).__name__)
        if "Flux" not in model_type_str:
            raise Exception(
                f"Model type {model_type_str} is not a Flux model. "
                "This node only works with Flux models."
            )
        model.model.diffusion_model.forward = patched_flux_forward.__get__(
            model.model.diffusion_model
        )
        return (model,)