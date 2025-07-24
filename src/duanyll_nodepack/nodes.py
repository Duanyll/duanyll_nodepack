from .photododdle import PhotoDoddleConditioning
from .difference import ImageDifferenceCmap
from .kontext import FluxKontextTrue3DPE
from .loader import (
    HfCheckpointLoader,
    HfDiffusionModelLoader,
    HfDualClipLoader,
    HfLoraLoader,
    HfLoraLoaderModelOnly,
    HfVaeLoader,
)
from .qwen import DrawBoundingBoxesQwen, CreateBoundingBoxesMaskQwen

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "PhotoDoddleConditioning": PhotoDoddleConditioning,
    "ImageDifferenceCmap": ImageDifferenceCmap,
    "FluxKontextTrue3DPE": FluxKontextTrue3DPE,
    "HfCheckpointLoader": HfCheckpointLoader,
    "HfDiffusionModelLoader": HfDiffusionModelLoader,
    "HfDualClipLoader": HfDualClipLoader,
    "HfLoraLoader": HfLoraLoader,
    "HfLoraLoaderModelOnly": HfLoraLoaderModelOnly,
    "HfVaeLoader": HfVaeLoader,
    "DrawBoundingBoxesQwen": DrawBoundingBoxesQwen,
    "CreateBoundingBoxesMaskQwen": CreateBoundingBoxesMaskQwen,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoDoddleConditioning": "PhotoDoddle Conditioning",
    "ImageDifferenceCmap": "Image Difference with Colormap",
    "FluxKontextTrue3DPE": "Patch Flux Kontext True 3D PE",
    "HfCheckpointLoader": "HuggingFace Checkpoint Loader",
    "HfDiffusionModelLoader": "HuggingFace Diffusion Model Loader",
    "HfDualClipLoader": "HuggingFace Dual CLIP Loader",
    "HfLoraLoader": "HuggingFace LoRA Loader",
    "HfLoraLoaderModelOnly": "HuggingFace LoRA Loader (Model Only)",
    "HfVaeLoader": "HuggingFace VAE Loader",
    "DrawBoundingBoxesQwen": "Draw Bounding Boxes (Qwen)",
    "CreateBoundingBoxesMaskQwen": "Create Bounding Boxes Mask (Qwen)",
}
