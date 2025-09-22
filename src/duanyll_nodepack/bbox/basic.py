import os
from collections import namedtuple

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..data.any import AnyType


# 字体路径查找逻辑，增强了兼容性
try:
    # 优先使用脚本相对路径
    FONT_PATH = os.path.join(os.path.dirname(__file__), "../../../assets/hei.TTF")
    if not os.path.exists(FONT_PATH):
        # 如果不存在，尝试一个常见的Linux系统路径
        FONT_PATH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
        if not os.path.exists(FONT_PATH):
            # 最终回退，让Pillow加载默认字体
            FONT_PATH = None
except (NameError, FileNotFoundError):
    # 在某些环境下 __file__ 可能未定义，直接回退
    FONT_PATH = None


class BoundingBox:
    """一个定义了矩形边界框的类"""
    def __init__(self, xmin, ymin, xmax, ymax, label=""):
        self.xmin = int(float(xmin))
        self.ymin = int(float(ymin))
        self.xmax = int(float(xmax))
        self.ymax = int(float(ymax))
        self.label = str(label)

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin
        
    @property
    def center_x(self):
        return self.xmin + self.width / 2

    @property
    def center_y(self):
        return self.ymin + self.height / 2

    def __repr__(self):
        return (f"BoundingBox(xmin={self.xmin}, ymin={self.ymin}, "
                    f"xmax={self.xmax}, ymax={self.ymax}, label='{self.label}')")

    def to_dict(self):
        return {
            "bbox_2d": [self.xmin, self.ymin, self.xmax, self.ymax],
            "label": self.label,
        }


class ParseBBoxQwenVL:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_data": (AnyType('*'), )
            }
        }

    RETURN_TYPES = ("BBOX",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "parse_bbox"
    CATEGORY = "duanyll/bbox"

    def parse_bbox(self, json_data):
        bboxes = []
        for bbox in json_data:
            if "bbox_2d" in bbox:
                bboxes.append(
                    BoundingBox(
                        xmin=bbox["bbox_2d"][0],
                        ymin=bbox["bbox_2d"][1],
                        xmax=bbox["bbox_2d"][2],
                        ymax=bbox["bbox_2d"][3],
                        label=bbox.get("label", ""),
                    )
                )
        return (bboxes,)


class DrawBBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bboxes": ("BBOX",),
                "draw_mode": (["Outline", "Fill"],),
                "line_thickness": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "fill_opacity": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "slider",
                    },
                ),
                "draw_label": (
                    "BOOLEAN",
                    {"default": True, "label_on": "enabled", "label_off": "disabled"},
                ),
                "font_size": (
                    "INT",
                    {
                        "default": 25,
                        "min": 8,
                        "max": 200,
                        "step": 1,
                        "display": "number",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    INPUT_IS_LIST = (True,)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "draw_bbox"
    CATEGORY = "duanyll/bbox"

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        image_np = tensor.cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        if image_np.ndim == 2:
            return Image.fromarray(image_np, "L").convert("RGB")
        return Image.fromarray(image_np, "RGB")

    def _pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0)

    def handle_image(
        self,
        images,
        bboxes,
        draw_mode,
        line_thickness,
        fill_opacity,
        draw_label,
        font_size,
    ):
        try:
            font = (
                ImageFont.truetype(FONT_PATH, font_size)
                if FONT_PATH
                else ImageFont.load_default()
            )
        except IOError:
            print(
                "Warning: [Draw Bounding Things] Custom font not found. Using default font."
            )
            font = ImageFont.load_default()

        for img_tensor in images:
            pil_img = self._tensor_to_pil(img_tensor)
            img_width, img_height = pil_img.size

            # 准备一个单独的透明层用于绘制，最后统一合成
            # 这样可以正确处理轮廓和填充模式，并确保标签总在最上层
            overlay_layer = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay_layer)

            processed_tensors = []

            for bbox in bboxes:
                label_ref_box = None  # 用于标签定位的参考框

                x_min_c = max(0, bbox.xmin)
                y_min_c = max(0, bbox.ymin)
                x_max_c = min(img_width - 1, bbox.xmax)
                y_max_c = min(img_height - 1, bbox.ymax)

                if x_min_c >= x_max_c or y_min_c >= y_max_c:
                    continue

                clipped_box = [x_min_c, y_min_c, x_max_c, y_max_c]
                label_ref_box = clipped_box  # Bbox本身就是标签的参考框

                if draw_mode == "Outline":
                    draw.rectangle(clipped_box, outline="red", width=line_thickness)
                else:  # Fill mode
                    alpha = int(fill_opacity * 255)
                    draw.rectangle(clipped_box, fill=(255, 0, 0, alpha))

                # --- 绘制标签 (如果存在参考框) ---
                if label_ref_box and draw_label and bbox.label:
                    label = str(bbox.label)
                    label_color, text_color, padding = "red", "white", 5

                    try:
                        text_bbox = font.getbbox(label)
                        text_width, text_height = (
                            text_bbox[2] - text_bbox[0],
                            text_bbox[3] - text_bbox[1],
                        )
                    except AttributeError:
                        text_width, text_height = font.getsize(label)

                    # 使用参考框的坐标进行智能定位
                    ref_x0, ref_y0, ref_x1, ref_y1 = label_ref_box

                    label_y0 = ref_y0 - text_height - padding * 2
                    if label_y0 < 0:
                        label_y0 = ref_y0
                    label_y1 = label_y0 + text_height + padding * 2

                    label_x0 = ref_x0
                    if label_x0 + text_width + padding * 2 > ref_x1:
                        label_x0 = ref_x1 - text_width - padding * 2

                    label_x0 = max(0, label_x0)
                    label_x1 = label_x0 + text_width + padding * 2

                    text_x, text_y = label_x0 + padding, label_y0 + padding

                    # 标签总是画在最上层，并且不透明
                    draw.rectangle(
                        [label_x0, label_y0, label_x1, label_y1], fill=label_color
                    )
                    draw.text((text_x, text_y), label, fill=text_color, font=font)

            # 将绘制层合成到原始图像上
            pil_img = pil_img.convert("RGBA")
            combined_img = Image.alpha_composite(pil_img, overlay_layer)
            pil_img = combined_img.convert("RGB")

            processed_tensors.append(self._pil_to_tensor(pil_img))

        output_image = torch.stack(processed_tensors)
        return output_image

    def draw_bbox(
        self,
        image,
        bboxes,
        draw_mode,
        line_thickness,
        fill_opacity,
        draw_label,
        font_size,
    ):
        return (
            [
                self.handle_image(
                    i,
                    bboxes,
                    draw_mode[0],
                    line_thickness[0],
                    fill_opacity[0],
                    draw_label[0],
                    font_size[0],
                )
                for i in image
            ],
        )


class DrawBBoxMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "width": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 8192,
                        "step": 8,
                        "display": "number",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 8192,
                        "step": 8,
                        "display": "number",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MASK",)
    INPUT_IS_LIST = (True,)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "draw_bbox_mask"
    CATEGORY = "duanyll/bbox"

    def draw_bbox_mask(self, bboxes, width, height):
        width = int(width[0])
        height = int(height[0])

        mask = torch.zeros(height, width, dtype=torch.float32)

        for bbox in bboxes:
            x_min_c = max(0, bbox.xmin)
            y_min_c = max(0, bbox.ymin)
            x_max_c = min(width - 1, bbox.xmax)
            y_max_c = min(height - 1, bbox.ymax)

            if x_min_c >= x_max_c or y_min_c >= y_max_c:
                continue

            mask[y_min_c : y_max_c + 1, x_min_c : x_max_c + 1] = 1.0

        mask = mask.unsqueeze(0)

        return (mask,)


class ExpandBBoxByRatio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox": ("BBOX",),
                "top_ratio": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": -0.5,
                        "max": 100.0,
                        "step": 0.01,
                    },
                ),
                "bottom_ratio": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": -0.5,
                        "max": 100.0,
                        "step": 0.01,
                    },
                ),
                "left_ratio": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": -0.5,
                        "max": 100.0,
                        "step": 0.01,
                    },
                ),
                "right_ratio": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": -0.5,
                        "max": 100.0,
                        "step": 0.01,
                    },
                ),
            }
        }

    RETURN_TYPES = ("BBOX",)
    FUNCTION = "expand_bbox_by_ratio"
    CATEGORY = "duanyll/bbox"

    def expand_bbox_by_ratio(
        self, bbox, top_ratio, bottom_ratio, left_ratio, right_ratio
    ):
        width = bbox.xmax - bbox.xmin
        height = bbox.ymax - bbox.ymin

        top_expand = int(height * top_ratio)
        bottom_expand = int(height * bottom_ratio)
        left_expand = int(width * left_ratio)
        right_expand = int(width * right_ratio)

        return (
            BoundingBox(
                bbox.xmin - left_expand,
                bbox.ymin - top_expand,
                bbox.xmax + right_expand,
                bbox.ymax + bottom_expand,
                label=bbox.label,
            ),
        )


class MaskToBBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "label": ("STRING", {"default": "object"}),
            }
        }

    RETURN_TYPES = ("BBOX",)
    FUNCTION = "mask_to_bbox"
    CATEGORY = "duanyll/bbox"

    def mask_to_bbox(self, mask, label):
        if not isinstance(mask, torch.Tensor):
            raise ValueError("Input mask must be a tensor.")

        if mask.dim() != 2:
            raise ValueError("Input mask must be a 2D tensor.")

        mask_bool = mask > 0.5  # 将掩码转换为布尔值
        if not mask_bool.any():
            # Got Empty Mask, return full image bbox
            return (BoundingBox(0, 0, mask.shape[1], mask.shape[0], label=label),)

        mask_x = mask_bool.any(dim=0).nonzero()
        mask_y = mask_bool.any(dim=1).nonzero()

        x_min = mask_x.min().item()
        x_max = mask_x.max().item() + 1  # +1 to include
        y_min = mask_y.min().item()
        y_max = mask_y.max().item() + 1  # +1 to include

        return (BoundingBox(x_min, y_min, x_max, y_max, label=label),)


class MergeBBoxes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bboxes": ("BBOX", ),
                "method": (["union", "intersect"], {"default": "union"}),
            }
        }
        
    RETURN_TYPES = ("BBOX",)
    FUNCTION = "merge"
    CATEGORY = "duanyll/bbox"
    INPUT_IS_LIST = True
    
    def merge(self, bboxes, method):
        method = method[0]
        
        if not bboxes:
            return (BoundingBox(0, 0, 0, 0, label=""),)
        
        if method == "union":
            # Union: find the minimum bbox that covers all input bboxes
            xmin = min(bbox.xmin for bbox in bboxes)
            ymin = min(bbox.ymin for bbox in bboxes)
            xmax = max(bbox.xmax for bbox in bboxes)
            ymax = max(bbox.ymax for bbox in bboxes)
            
            return (BoundingBox(xmin, ymin, xmax, ymax, label="union"),)
            
        elif method == "intersect":
            # Intersection: find the maximum bbox that is covered by all input bboxes
            xmin = max(bbox.xmin for bbox in bboxes)
            ymin = max(bbox.ymin for bbox in bboxes)
            xmax = min(bbox.xmax for bbox in bboxes)
            ymax = min(bbox.ymax for bbox in bboxes)
            
            # Check if intersection is valid (positive area)
            if xmin >= xmax or ymin >= ymax:
                # No valid intersection, return zero bbox
                return (BoundingBox(0, 0, 0, 0, label=""),)
            
            return (BoundingBox(xmin, ymin, xmax, ymax, label="intersect"),)
        
        else:
            # Unknown method, return zero bbox
            return (BoundingBox(0, 0, 0, 0, label=""),)


SEG = namedtuple("SEG",
                 ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'],
                 defaults=[None])


class BBoxesToImpactPackSegs:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "width": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 8192,
                        "step": 8,
                        "display": "number",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 8192,
                        "step": 8,
                        "display": "number",
                    },
                ),   
            }
        }
        
    INPUT_IS_LIST = True
    RETURN_TYPES = ("SEGS",)
    FUNCTION = "bboxes_to_impactpack_segs"
    CATEGORY = "duanyll/bbox"
    
    def bboxes_to_impactpack_segs(self, bboxes, width, height):
        width = int(width[0])
        height = int(height[0])
        
        return ((
            (width, height),
            [SEG(
                cropped_image=None,
                cropped_mask=None,
                confidence=1.0,
                crop_region=None,
                bbox=[i.xmin, i.ymin, i.xmax, i.ymax],
                label=i.label,
            ) for i in bboxes]
        ),)