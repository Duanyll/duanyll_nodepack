from .models.photododdle import PhotoDoddleConditioning
from .image.difference import ImageDifferenceCmap
from .models.kontext import FluxKontextTrue3DPE
from .loaders.hf import (
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
from .bbox.basic import (
    ParseBBoxQwenVL,
    DrawBBox,
    DrawBBoxMask,
    ExpandBBoxByRatio,
    MaskToBBox,
    MergeBBoxes
)
from .bbox.image import BBoxCrop, BBoxImageStitcher, FillBBoxWithImage
from .bbox.text import GetTextBBoxWithAnchor, DrawTextInBBox
from .qwen import DrawBoundingBoxesQwen, CreateBoundingBoxesMaskQwen
from .models.fluxtext import FluxTextLoraLoader
from .morphology import CoverWordsWithRectangles, AdvancedMorphology
from .image.resize import ImagePadToResolution, ImageCropFromPadded
from .face import InsightFaceSimilarity
from .ark import CreateArkClient, SeedEditNode
from .data.any import AsAny
from .data.json import (
    ParseLlmJsonOutput,
    JsonPathQuery,
    JsonPathQuerySingle,
    JsonPathUpdate,
    ParseJson5,
    DumpJson,
)
from .loaders.basic import ReadTextFile
from .web.http import HttpPostForJson, DownloadImageFromUrl
from .web.s3 import CreateS3Client, UploadImageToS3

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
    "MaskToBBox": MaskToBBox,
    "MergeBBoxes": MergeBBoxes,
    "BBoxCrop": BBoxCrop,
    "BBoxImageStitcher": BBoxImageStitcher,
    "FillBBoxWithImage": FillBBoxWithImage,
    "GetTextBBoxWithAnchor": GetTextBBoxWithAnchor,
    "DrawTextInBBox": DrawTextInBBox,
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
    "AsAny": AsAny,
    "ParseLlmJsonOutput": ParseLlmJsonOutput,
    "JsonPathQuery": JsonPathQuery,
    "JsonPathQuerySingle": JsonPathQuerySingle,
    "JsonPathUpdate": JsonPathUpdate,
    "DownloadImageFromUrl": DownloadImageFromUrl,
    "ReadTextFile": ReadTextFile,
    "ParseJson5": ParseJson5,
    "DumpJson": DumpJson,
    "HttpPostForJson": HttpPostForJson,
    "CreateS3Client": CreateS3Client,
    "UploadImageToS3": UploadImageToS3
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
    "MaskToBBox": "Mask to Bounding Box",
    "MergeBBoxes": "Merge Bounding Boxes",
    "BBoxCrop": "Bounding Box Crop",
    "BBoxImageStitcher": "Bounding Box Image Stitcher",
    "FillBBoxWithImage": "Fill Bounding Box with Image",
    "GetTextBBoxWithAnchor": "Get Text BBox with Anchor",
    "DrawTextInBBox": "Draw Text in BBox",
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
    "AsAny": "As Any",
    "ParseLlmJsonOutput": "Parse LLM JSON Output",
    "JsonPathQuery": "JSON Path Query",
    "JsonPathQuerySingle": "JSON Path Query Single",
    "JsonPathUpdate": "JSON Path Update",
    "DownloadImageFromUrl": "Download Image from URL",
    "ReadTextFile": "Read Text File",
    "ParseJson5": "Parse JSON5",
    "DumpJson": "Dump JSON",
    "HttpPostForJson": "HTTP Post for JSON",
    "CreateS3Client": "Create S3 Client",
    "UploadImageToS3": "Upload Image to S3"
}
