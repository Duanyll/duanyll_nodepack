import torch
from comfy.samplers import CFGGuider


class ChannelIncrementalConstrainedCFGGuider(CFGGuider):
    def __init__(self, model_patcher, vae, orig_image, constraint_strength):
        super().__init__(model_patcher)

        self.vae = vae
        self.orig_image = orig_image
        self.constraint_strength = constraint_strength

    def predict_noise(self, x, timestep, model_options=..., seed=None):
        # This function actually returns the denoised image instead of noise
        denoised = super().predict_noise(x, timestep, model_options, seed)
        denoised_image = self.vae.decode(denoised)  # b, h, w, c
        delta = denoised_image - self.orig_image
        constrained_image = self.orig_image + torch.mean(delta, dim=-1, keepdim=True)
        constrained_image = torch.clamp(constrained_image, 0, 1.0)
        constrained_image = constrained_image.squeeze(0)
        constrained = self.vae.encode(constrained_image)
        strength = ((1 - timestep) ** 2) * self.constraint_strength
        scaled = denoised + strength.to(denoised.device) * (constrained.to(denoised.device) - denoised)
        return scaled


class ChannelIncrementalConstrainedCFGGuiderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "vae": ("VAE",),
                "orig_image": ("IMAGE",),
                "constraint_strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "round": 0.01,
                    },
                ),
            }
        }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "duanyll/models/experimental"

    def get_guider(
        self, model, positive, negative, cfg, vae, orig_image, constraint_strength
    ):
        guider = ChannelIncrementalConstrainedCFGGuider(model, vae, orig_image, constraint_strength)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        return (guider,)
