from .models.photododdle import PhotoDoddleConditioning
from .image.difference import ImageDifferenceCmap
from .models.kontext import FluxKontextTrue3DPE
from .loaders import (
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
from .bbox import (
    ParseBBoxQwenVL,
    DrawBBox,
    DrawBBoxMask,
    ExpandBBoxByRatio,
    BBoxCrop,
    BBoxImageStitcher
)
from .qwen import DrawBoundingBoxesQwen, CreateBoundingBoxesMaskQwen
from .models.fluxtext import FluxTextLoraLoader
from .morphology import CoverWordsWithRectangles, AdvancedMorphology
from .image.resize import ImagePadToResolution, ImageCropFromPadded
from .face import InsightFaceSimilarity
from .ark import CreateArkClient, SeedEditNode
from .data import ParseLlmJsonOutput, JsonPathQuery, JsonPathQuerySingle, JsonPathUpdate

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
    "ParseBBoxQwenVL": ParseBBoxQwenVL,
    "DrawBBox": DrawBBox,
    "DrawBBoxMask": DrawBBoxMask,
    "ExpandBBoxByRatio": ExpandBBoxByRatio,
    "BBoxCrop": BBoxCrop,
    "BBoxImageStitcher": BBoxImageStitcher,
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
    "ParseLlmJsonOutput": ParseLlmJsonOutput,
    "JsonPathQuery": JsonPathQuery,
    "JsonPathQuerySingle": JsonPathQuerySingle,
    "JsonPathUpdate": JsonPathUpdate
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
    "ParseBBoxQwenVL": "Parse BBox QwenVL",
    "DrawBBox": "Draw Bounding Boxes",
    "DrawBBoxMask": "Draw Bounding Box Mask",
    "ExpandBBoxByRatio": "Expand Bounding Box by Ratio",
    "BBoxCrop": "Bounding Box Crop",
    "BBoxImageStitcher": "Bounding Box Image Stitcher",
    "DrawBoundingBoxesQwen": "[DEPR] Draw Bounding Boxes Qwen",
    "CreateBoundingBoxesMaskQwen": "[DEPR] Create Bounding Boxes Mask Qwen",
    "FluxTextLoraLoader": "Flux Text LoRA Loader",
    "CoverWordsWithRectangles": "Cover Words with Rectangles",
    "AdvancedMorphology": "[DEPR] Advanced Morphology",
    "ImagePadToResolution": "Pad to Resolution",
    "ImageCropFromPadded": "Crop from Padded",
    "InsightFaceSimilarity": "InsightFace Similarity",
    "CreateArkClient": "Create Ark Client",
    "SeedEditNode": "SeedEdit Node",
    "ParseLlmJsonOutput": "Parse LLM JSON Output",
    "JsonPathQuery": "JSON Path Query",
    "JsonPathQuerySingle": "JSON Path Query Single",
    "JsonPathUpdate": "JSON Path Update"
}
