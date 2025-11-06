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
    MergeBBoxes,
    BBoxesToImpactPackSegs
)
from .bbox.image import BBoxCrop, BBoxImageStitcher, FillBBoxWithImage
from .bbox.text import GetTextBBoxWithAnchor, DrawTextInBBox
from .models.fluxtext import FluxTextLoraLoader
from .morphology import CoverWordsWithRectangles
from .image.resize import ImagePadToResolution, ImageCropFromPadded
from .metric import InsightFaceSimilarity, LaplacianVariance
from .models.ark import CreateArkClient, SeedEditNode
from .data.any import AsAny
from .data.json import (
    ParseLlmJsonOutput,
    JsonPathQuery,
    JsonPathQuerySingle,
    JsonPathUpdate,
    ParseJson5,
    DumpJson,
)
from .data.text import TextContainsChinese, StringFormat
from .loaders.basic import ReadTextFile
from .web.http import DownloadImageFromUrl
from .web.s3 import CreateS3Client, UploadImageToS3
from .models.diffusers_repl import DiffusersRandomNoise, DiffusersFluxScheduler, QwenImageClipEnforceBfloat16
from .llm import LlmCreateClient, LlmClientSetSeed, LlmCreateChat, LlmChatAddMessage, LlmChatCompletion
from .models.hunyuan import VllmHunyuanImage3Node

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
    "BBoxesToImpactPackSegs": BBoxesToImpactPackSegs,
    "BBoxImageStitcher": BBoxImageStitcher,
    "FillBBoxWithImage": FillBBoxWithImage,
    "GetTextBBoxWithAnchor": GetTextBBoxWithAnchor,
    "DrawTextInBBox": DrawTextInBBox,
    "FluxTextLoraLoader": FluxTextLoraLoader,
    "CoverWordsWithRectangles": CoverWordsWithRectangles,
    "ImagePadToResolution": ImagePadToResolution,
    "ImageCropFromPadded": ImageCropFromPadded,
    "InsightFaceSimilarity": InsightFaceSimilarity,
    "LaplacianVariance": LaplacianVariance,
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
    "CreateS3Client": CreateS3Client,
    "UploadImageToS3": UploadImageToS3,
    "DiffusersRandomNoise": DiffusersRandomNoise,
    "DiffusersFluxScheduler": DiffusersFluxScheduler,
    "QwenImageClipEnforceBfloat16": QwenImageClipEnforceBfloat16,
    "LlmCreateClient": LlmCreateClient,
    "LlmClientSetSeed": LlmClientSetSeed,
    "LlmCreateChat": LlmCreateChat,
    "LlmChatAddMessage": LlmChatAddMessage,
    "LlmChatCompletion": LlmChatCompletion,
    "TextContainsChinese": TextContainsChinese,
    "StringFormat": StringFormat,
    "VllmHunyuanImage3Node": VllmHunyuanImage3Node,
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
    "BBoxesToImpactPackSegs": "BBoxes to ImpactPack Segs",
    "BBoxImageStitcher": "Bounding Box Image Stitcher",
    "FillBBoxWithImage": "Fill Bounding Box with Image",
    "GetTextBBoxWithAnchor": "Get Text BBox with Anchor",
    "DrawTextInBBox": "Draw Text in BBox",
    "FluxTextLoraLoader": "Flux Text LoRA Loader",
    "CoverWordsWithRectangles": "Cover Words with Rectangles",
    "ImagePadToResolution": "Pad to Resolution",
    "ImageCropFromPadded": "Crop from Padded",
    "InsightFaceSimilarity": "InsightFace Similarity",
    "LaplacianVariance": "Laplacian Variance",
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
    "CreateS3Client": "Create S3 Client",
    "UploadImageToS3": "Upload Image to S3",
    "DiffusersRandomNoise": "Diffusers Random Noise",
    "DiffusersFluxScheduler": "Diffusers Flux Scheduler",
    "QwenImageClipEnforceBfloat16": "Qwen-Image Clip Enforce Bfloat16",
    "LlmCreateClient": "LLM Create Client",
    "LlmClientSetSeed": "LLM Client Set Seed",
    "LlmCreateChat": "LLM Create Chat",
    "LlmChatAddMessage": "LLM Chat Add Message",
    "LlmChatCompletion": "LLM Chat Completion",
    "TextContainsChinese": "Text Contains Chinese",
    "StringFormat": "String Format",
    "VllmHunyuanImage3Node": "vLLM HunyuanImage3",
}

PREFIX = "duanyll::"
NODE_CLASS_MAPPINGS = {
    (k if k.startswith("__") else PREFIX + k): v
    for k, v in NODE_CLASS_MAPPINGS.items()
}
NODE_DISPLAY_NAME_MAPPINGS = {
    (k if k.startswith("__") else PREFIX + k): v
    for k, v in NODE_DISPLAY_NAME_MAPPINGS.items()
}