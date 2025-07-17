from .photododdle import PhotoDoddleConditioning
from .difference import ImageDifferenceCmap
from .kontext import FluxKontextTrue3DPE
from .loader import (
    HfCheckpointLoader,
    HfDualClipLoader,
    HfLoraLoader,
    HfLoraLoaderModelOnly,
    HfVaeLoader,
)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "PhotoDoddleConditioning": PhotoDoddleConditioning,
    "ImageDifferenceCmap": ImageDifferenceCmap,
    "FluxKontextTrue3DPE": FluxKontextTrue3DPE,
    "HfCheckpointLoader": HfCheckpointLoader,
    "HfDualClipLoader": HfDualClipLoader,
    "HfLoraLoader": HfLoraLoader,
    "HfLoraLoaderModelOnly": HfLoraLoaderModelOnly,
    "HfVaeLoader": HfVaeLoader,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoDoddleConditioning": "PhotoDoddle Conditioning",
    "ImageDifferenceCmap": "Image Difference with Colormap",
    "FluxKontextTrue3DPE": "Patch Flux Kontext True 3D PE",
    "HfCheckpointLoader": "HuggingFace Checkpoint Loader",
    "HfDualClipLoader": "HuggingFace Dual CLIP Loader",
    "HfLoraLoader": "HuggingFace LoRA Loader",
    "HfLoraLoaderModelOnly": "HuggingFace LoRA Loader (Model Only)",
    "HfVaeLoader": "HuggingFace VAE Loader",
}
