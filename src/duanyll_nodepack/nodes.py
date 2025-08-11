from .photododdle import PhotoDoddleConditioning
from .difference import ImageDifferenceCmap
from .kontext import FluxKontextTrue3DPE
from .loader import (
    HfCheckpointLoader,
    HfDiffusionModelLoader,
    HfClipLoader,
    HfDualClipLoader,
    HfLoraLoader,
    HfLoraLoaderModelOnly,
    HfVaeLoader,
    HfTripleClipLoader,
    HfQuadrupleClipLoader,
)
from .qwen import DrawBoundingBoxesQwen, CreateBoundingBoxesMaskQwen
from .fluxtext import FluxTextLoraLoader
from .morphology import CoverWordsWithRectangles, AdvancedMorphology
from .resize import ImagePadToResolution, ImageCropFromPadded
from .face import InsightFaceSimilarity
from .ark import CreateArkClient, SeedEditNode

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "PhotoDoddleConditioning": PhotoDoddleConditioning,
    "ImageDifferenceCmap": ImageDifferenceCmap,
    "FluxKontextTrue3DPE": FluxKontextTrue3DPE,
    "HfCheckpointLoader": HfCheckpointLoader,
    "HfDiffusionModelLoader": HfDiffusionModelLoader,
    "HfClipLoader": HfClipLoader,
    "HfDualClipLoader": HfDualClipLoader,
    "HfLoraLoader": HfLoraLoader,
    "HfLoraLoaderModelOnly": HfLoraLoaderModelOnly,
    "HfVaeLoader": HfVaeLoader,
    "HfTripleClipLoader": HfTripleClipLoader,
    "HfQuadrupleClipLoader": HfQuadrupleClipLoader,
    "DrawBoundingBoxesQwen": DrawBoundingBoxesQwen,
    "CreateBoundingBoxesMaskQwen": CreateBoundingBoxesMaskQwen,
    "FluxTextLoraLoader": FluxTextLoraLoader,
    "CoverWordsWithRectangles": CoverWordsWithRectangles,
    "AdvancedMorphology": AdvancedMorphology,
    "ImagePadToResolution": ImagePadToResolution,
    "ImageCropFromPadded": ImageCropFromPadded,
    "InsightFaceSimilarity": InsightFaceSimilarity,
    "CreateArkClient": CreateArkClient,
    "SeedEditNode": SeedEditNode,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoDoddleConditioning": "PhotoDoddle Conditioning",
    "ImageDifferenceCmap": "Image Difference with Colormap",
    "FluxKontextTrue3DPE": "Patch Flux Kontext True 3D PE",
    "HfCheckpointLoader": "HuggingFace Checkpoint Loader",
    "HfDiffusionModelLoader": "HuggingFace Diffusion Model Loader",
    "HfClipLoader": "HuggingFace CLIP Loader",
    "HfDualClipLoader": "HuggingFace Dual CLIP Loader",
    "HfLoraLoader": "HuggingFace LoRA Loader",
    "HfLoraLoaderModelOnly": "HuggingFace LoRA Loader (Model Only)",
    "HfVaeLoader": "HuggingFace VAE Loader",
    "HfTripleClipLoader": "HuggingFace Triple CLIP Loader",
    "HfQuadrupleClipLoader": "HuggingFace Quadruple CLIP Loader",
    "DrawBoundingBoxesQwen": "Draw Bounding Boxes (Qwen)",
    "CreateBoundingBoxesMaskQwen": "Create Bounding Boxes Mask (Qwen)",
    "FluxTextLoraLoader": "Flux Text LoRA Loader",
    "CoverWordsWithRectangles": "Cover Words with Rectangles",
    "AdvancedMorphology": "Advanced Morphology",
    "ImagePadToResolution": "Pad to Resolution",
    "ImageCropFromPadded": "Crop from Padded",
    "InsightFaceSimilarity": "InsightFace Similarity",
    "CreateArkClient": "Create Ark Client",
    "SeedEditNode": "SeedEdit Node",
}
