import torch
import math
import numbers
from typing import List, Union
from comfy.sd1_clip import SDClipModel

class Noise_DiffusersRandomNoise:
    def __init__(self, seed, device="cuda:0", dtype="bfloat16"):
        self.seed = seed
        self.device = device
        self.dtype = dtype

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        b, c, _, h, w = latent_image.shape
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        noise = torch.randn((b, c, h, w), 
                            generator=generator, 
                            device=self.device, 
                            dtype=getattr(torch, self.dtype))
        return noise.view_as(latent_image)
    

class DiffusersRandomNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise_seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
                "device": ("STRING", {
                    "default": "cuda:0",
                    "multiline": False,
                }),
                "dtype": ("STRING", {
                    "default": "bfloat16",
                    "multiline": False,
                }),
            }
        }
    
    RETURN_TYPES = ("NOISE",)
    FUNCTION = "get_noise"
    CATEGORY = "duanyll/models/diffusers"

    def get_noise(self, noise_seed, device="cuda:0", dtype="bfloat16"):
        noise_generator = Noise_DiffusersRandomNoise(
            seed=noise_seed,
            device=device,
            dtype=dtype,
        )
        return (noise_generator,)
    


def calculate_mu(image_seq_len: int,
                 base_seq_len: int = 256,
                 max_seq_len: int = 8192,
                 base_shift: float = 0.5,
                 max_shift: float = 0.9) -> float:
    """
    计算 mu（仍然是标量，因为它只依赖于序列长度）。
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def stretch_to_terminal(t: torch.Tensor, shift_terminal: float = 0.02) -> torch.Tensor:
    """
    对 Tensor 进行拉伸，使最后一个值映射到 shift_terminal。
    """
    one_minus_z = 1.0 - t
    scale_factor = one_minus_z[-1] / (1.0 - shift_terminal)
    stretched_t = 1.0 - (one_minus_z / scale_factor)
    return stretched_t


def get_shifted_timesteps(num_inference_steps: int,
                          image_seq_len: int) -> torch.Tensor:
    """
    返回形状为 (num_inference_steps,) 的 torch.Tensor。
    """
    # 1. 生成等间隔的 sigma（这里用 torch.linspace）
    sigmas = torch.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)

    # 2. 计算 mu（标量）
    mu = calculate_mu(image_seq_len)

    # 3. 对每个 sigma 应用 time_shift_exponential
    #    使用向量化方式，避免 Python 循环
    exp_mu = math.exp(mu)
    sigmas = exp_mu / (exp_mu + (1.0 / sigmas - 1.0))

    # 4. 拉伸到终端
    sigmas = stretch_to_terminal(sigmas)
    
    # Append 0.0 at the end
    sigmas = torch.cat([sigmas, torch.tensor([0.0], device=sigmas.device, dtype=sigmas.dtype)], dim=0)
    
    return sigmas


class DiffusersFluxScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_inference_steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 1000,
                }),
                "image_seq_len": ("INT", {
                    "default": 256,
                    "min": 16,
                    "max": 8192,
                }),
            }
        }
        
    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "get_sigmas"
    CATEGORY = "duanyll/models/diffusers"
    
    def get_sigmas(self, num_inference_steps: int, image_seq_len: int):
        sigmas = get_shifted_timesteps(num_inference_steps, image_seq_len)
        return (sigmas,)
    
    
def patched_sdclipmodel_process_tokens(self, tokens, device):
    end_token = self.special_tokens.get("end", None)
    if end_token is None:
        cmp_token = self.special_tokens.get("pad", -1)
    else:
        cmp_token = end_token

    embeds_out = []
    attention_masks = []
    num_tokens = []

    for x in tokens:
        attention_mask = []
        tokens_temp = []
        other_embeds = []
        eos = False
        index = 0
        for y in x:
            if isinstance(y, numbers.Integral):
                if eos:
                    attention_mask.append(0)
                else:
                    attention_mask.append(1)
                token = int(y)
                tokens_temp += [token]
                if not eos and token == cmp_token:
                    if end_token is None:
                        attention_mask[-1] = 0
                    eos = True
            else:
                other_embeds.append((index, y))
            index += 1

        tokens_embed = torch.tensor([tokens_temp], device=device, dtype=torch.long)
        tokens_embed = self.transformer.get_input_embeddings()(tokens_embed, out_dtype=torch.bfloat16)
        index = 0
        pad_extra = 0
        embeds_info = []
        for o in other_embeds:
            emb = o[1]
            if torch.is_tensor(emb):
                emb = {"type": "embedding", "data": emb}

            extra = None
            emb_type = emb.get("type", None)
            if emb_type == "embedding":
                emb = emb.get("data", None)
            else:
                if hasattr(self.transformer, "preprocess_embed"):
                    emb, extra = self.transformer.preprocess_embed(emb, device=device)
                else:
                    emb = None

            if emb is None:
                index += -1
                continue

            ind = index + o[0]
            emb = emb.view(1, -1, emb.shape[-1]).to(device=device, dtype=torch.bfloat16)
            emb_shape = emb.shape[1]
            if emb.shape[-1] == tokens_embed.shape[-1]:
                tokens_embed = torch.cat([tokens_embed[:, :ind], emb, tokens_embed[:, ind:]], dim=1)
                attention_mask = attention_mask[:ind] + [1] * emb_shape + attention_mask[ind:]
                index += emb_shape - 1
                embeds_info.append({"type": emb_type, "index": ind, "size": emb_shape, "extra": extra})
            else:
                index += -1
                pad_extra += emb_shape

        if pad_extra > 0:
            padd_embed = self.transformer.get_input_embeddings()(torch.tensor([[self.special_tokens["pad"]] * pad_extra], device=device, dtype=torch.long), out_dtype=torch.bfloat16)
            tokens_embed = torch.cat([tokens_embed, padd_embed], dim=1)
            attention_mask = attention_mask + [0] * pad_extra

        embeds_out.append(tokens_embed)
        attention_masks.append(attention_mask)
        num_tokens.append(sum(attention_mask))

    return torch.cat(embeds_out), torch.tensor(attention_masks, device=device, dtype=torch.long), num_tokens, embeds_info


def patched_sdclipmodel_forward(self, tokens):
    device = self.transformer.get_input_embeddings().weight.device
    embeds, attention_mask, num_tokens, embeds_info = self.process_tokens(tokens, device)

    attention_mask_model = None
    if self.enable_attention_masks:
        attention_mask_model = attention_mask

    if self.layer == "all":
        intermediate_output = "all"
    else:
        intermediate_output = self.layer_idx

    outputs = self.transformer(None, attention_mask_model, embeds=embeds, num_tokens=num_tokens, intermediate_output=intermediate_output, final_layer_norm_intermediate=self.layer_norm_hidden_state, dtype=torch.bfloat16, embeds_info=embeds_info)

    if self.layer == "last":
        z = outputs[0].float()
    else:
        z = outputs[1].float()

    if self.zero_out_masked:
        z *= attention_mask.unsqueeze(-1).float()

    pooled_output = None
    if len(outputs) >= 3:
        if not self.return_projected_pooled and len(outputs) >= 4 and outputs[3] is not None:
            pooled_output = outputs[3].float()
        elif outputs[2] is not None:
            pooled_output = outputs[2].float()

    extra = {}
    if self.return_attention_masks:
        extra["attention_mask"] = attention_mask

    if len(extra) > 0:
        return z, pooled_output, extra

    return z, pooled_output


class QwenImageClipEnforceBfloat16:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {}),
            }
        }
        
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "patch_clip"
    CATEGORY = "duanyll/models/diffusers"
    
    def patch_clip(self, clip):
        # patch clip.cond_stage_model
        clip.cond_stage_model.qwen25_7b.process_tokens = patched_sdclipmodel_process_tokens.__get__(clip.cond_stage_model.qwen25_7b, SDClipModel)
        clip.cond_stage_model.qwen25_7b.forward = patched_sdclipmodel_forward.__get__(clip.cond_stage_model.qwen25_7b, SDClipModel)
        return (clip,)