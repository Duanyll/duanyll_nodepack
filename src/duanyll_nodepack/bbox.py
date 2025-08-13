import os
import json

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# 字体路径查找逻辑，增强了兼容性
try:
    # 优先使用脚本相对路径
    FONT_PATH = os.path.join(os.path.dirname(__file__), "../../assets/hei.TTF")
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
    def __init__(self, xmin, ymin, xmax, ymax, label=""):
        self.xmin = int(float(xmin))
        self.ymin = int(float(ymin))
        self.xmax = int(float(xmax))
        self.ymax = int(float(ymax))

        self.label = str(label)
                
    def __repr__(self):
        return f"BoundingBox(xmin={self.xmin}, ymin={self.ymin}, xmax={self.xmax}, ymax={self.ymax}, label='{self.label}')"


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
                "bbox_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": '[{"bbox_2d": [170, 10, 780, 830], "label": "face"}]',
                    },
                ),
            }
        }

    RETURN_TYPES = ("BBOX",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "parse_bbox"
    CATEGORY = "duanyll/bbox"

    def parse_bbox(self, bbox_json):
        try:
            bbox_data = json.loads(bbox_json)
            bboxes = [BoundingBox(**bbox) for bbox in bbox_data]
            return (bboxes,)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")


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
            return Image.fromarray(image_np, 'L').convert('RGB')
        return Image.fromarray(image_np, 'RGB')

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
            font = ImageFont.truetype(FONT_PATH, font_size) if FONT_PATH else ImageFont.load_default()
        except IOError:
            print("Warning: [Draw Bounding Things] Custom font not found. Using default font.")
            font = ImageFont.load_default()

        for img_tensor in images:
            pil_img = self._tensor_to_pil(img_tensor)
            img_width, img_height = pil_img.size
            
            # 准备一个单独的透明层用于绘制，最后统一合成
            # 这样可以正确处理轮廓和填充模式，并确保标签总在最上层
            overlay_layer = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay_layer)
            
            processed_tensors = []

            for bbox in bboxes:
                label_ref_box = None # 用于标签定位的参考框
                
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
                else: # Fill mode
                    alpha = int(fill_opacity * 255)
                    draw.rectangle(clipped_box, fill=(255, 0, 0, alpha))
                
                # --- 绘制标签 (如果存在参考框) ---
                if label_ref_box and draw_label and bbox.label:
                    label = str(bbox.label)
                    label_color, text_color, padding = "red", "white", 5

                    try:
                        text_bbox = font.getbbox(label)
                        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
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
                    draw.rectangle([label_x0, label_y0, label_x1, label_y1], fill=label_color)
                    draw.text((text_x, text_y), label, fill=text_color, font=font)

            # 将绘制层合成到原始图像上
            pil_img = pil_img.convert('RGBA')
            combined_img = Image.alpha_composite(pil_img, overlay_layer)
            pil_img = combined_img.convert('RGB')
            
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
                "width": ("INT", {
                    "default": 512, "min": 64, "max": 8192, "step": 8, "display": "number"
                }),
                "height": ("INT", {
                    "default": 512, "min": 64, "max": 8192, "step": 8, "display": "number"
                }),
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

            mask[y_min_c:y_max_c + 1, x_min_c:x_max_c + 1] = 1.0
            
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
                    }
                ),
                "bottom_ratio": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": -0.5,
                        "max": 100.0,
                        "step": 0.01,
                    }
                ),
                "left_ratio": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": -0.5,
                        "max": 100.0,
                        "step": 0.01,
                    }
                ),
                "right_ratio": (
                    "FLOAT",
                    {
                        "default": 0.1,
                        "min": -0.5,
                        "max": 100.0,
                        "step": 0.01,
                    }
                ),
            }
        }
        
    RETURN_TYPES = ("BBOX",)
    FUNCTION = "expand_bbox_by_ratio"
    CATEGORY = "duanyll/bbox"
    
    def expand_bbox_by_ratio(self, bbox, top_ratio, bottom_ratio, left_ratio, right_ratio):
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
                label=bbox.label
            ),
        )
        

class BBoxCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bbox": ("BBOX",),
            },
        }
        
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "bbox_crop"
    CATEGORY = "duanyll/bbox"
    
    def bbox_crop(self, image, bbox):
        if not isinstance(image, torch.Tensor):
            raise ValueError("Input image must be a tensor.")
        
        if not isinstance(bbox, BoundingBox):
            raise ValueError("Input bbox must be an instance of BoundingBox.")
        
        x_min_c = max(0, bbox.xmin)
        y_min_c = max(0, bbox.ymin)
        x_max_c = min(image.shape[2], bbox.xmax)
        y_max_c = min(image.shape[1], bbox.ymax)

        if x_min_c >= x_max_c or y_min_c >= y_max_c:
            raise ValueError("Invalid bounding box coordinates.")

        cropped_image = image[:, y_min_c:y_max_c, x_min_c:x_max_c]
        
        return (cropped_image,)
    
    
def tensor2pil(image: torch.Tensor) -> list[Image.Image]:
    """将输入的 torch.Tensor 转换为 PIL.Image.Image 列表"""
    return [Image.fromarray(np.clip(255. * i.cpu().numpy(), 0, 255).astype(np.uint8)) for i in image]

def pil2tensor(image: Image.Image) -> torch.Tensor:
    """将单个 PIL.Image.Image 转换为 torch.Tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class BBoxImageStitcher:
    # 设置输入是列表类型
    INPUT_IS_LIST = True
    
    # 设置第二个输出 (bbox) 是列表类型
    OUTPUT_IS_LIST = (False, True)

    @classmethod
    def INPUT_TYPES(cls):
        """定义节点的输入参数"""
        return {
            "required": {
                "images": ("IMAGE",),
                "direction": (["right", "down"],),  # 控制拼接方向的选项
                "size": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}), # 控制统一的宽度或高度
            }
        }

    RETURN_TYPES = ("IMAGE", "BBOX")
    RETURN_NAMES = ("stitched_image", "bbox")
    FUNCTION = "stitch_images"
    CATEGORY = "duanyll/bbox" # 您可以自定义节点的分类

    def stitch_images(self, images: list, direction: list, size: list):
        """
        节点的核心功能实现
        :param images: ComfyUI 传入的图像张量列表 (由于 INPUT_IS_LIST=True)
        :param direction: 拼接方向 ('right' 或 'down')
        :param size: 统一的尺寸 (高度或宽度)
        """
        # 由于 INPUT_IS_LIST=True，所有输入都被包装在列表中，我们需要解包
        if not images:
            # 如果没有输入图像，返回一个空的 torch 张量和空列表
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), [])

        direction = direction[0]
        size = size[0]
        
        # 将输入的 Tensor 列表转换为 PIL Image 列表
        pil_images = []
        for img_tensor in images:
            pil_images.extend(tensor2pil(img_tensor))

        if not pil_images:
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32), [])

        # --------------------------------------------------
        # 1. 确定统一缩放的基准尺寸
        # --------------------------------------------------
        target_size = size
        if size == 0:
            # 如果 size 为 0，自动计算最大宽度或高度
            if direction == "right": # 向右拼接，统一高度
                target_size = max(img.height for img in pil_images)
            else: # 向下拼接，统一宽度
                target_size = max(img.width for img in pil_images)

        # --------------------------------------------------
        # 2. 等比缩放所有图片
        # --------------------------------------------------
        resized_images = []
        for img in pil_images:
            original_width, original_height = img.width, img.height
            if direction == "right": # 统一高度
                new_height = target_size
                aspect_ratio = original_width / original_height
                new_width = int(new_height * aspect_ratio)
            else: # 统一宽度
                new_width = target_size
                aspect_ratio = original_height / original_width
                new_height = int(new_width * aspect_ratio)
            
            # 使用高质量的 LANCZOS 滤波器进行缩放
            resized_images.append(img.resize((new_width, new_height), Image.Resampling.LANCZOS))

        # --------------------------------------------------
        # 3. 计算拼接后大图的总尺寸
        # --------------------------------------------------
        if direction == "right":
            total_width = sum(img.width for img in resized_images)
            max_height = target_size
            canvas_size = (total_width, max_height)
        else: # direction == "down"
            total_height = sum(img.height for img in resized_images)
            max_width = target_size
            canvas_size = (max_width, total_height)

        # --------------------------------------------------
        # 4. 创建画布并拼接图像，同时生成 BBox
        # --------------------------------------------------
        # 创建一个 'RGB' 模式的空白画布
        stitched_image = Image.new('RGB', canvas_size)
        bboxes = []
        current_offset = 0 # 当前的偏移量（x 或 y）

        for i, img in enumerate(resized_images):
            if direction == "right":
                paste_position = (current_offset, 0)
                # 创建 BoundingBox
                bbox = BoundingBox(
                    xmin=current_offset,
                    ymin=0,
                    xmax=current_offset + img.width,
                    ymax=img.height,
                    label=f"image_{i}" # 使用索引作为默认标签
                )
                current_offset += img.width
            else: # direction == "down"
                paste_position = (0, current_offset)
                # 创建 BoundingBox
                bbox = BoundingBox(
                    xmin=0,
                    ymin=current_offset,
                    xmax=img.width,
                    ymax=current_offset + img.height,
                    label=f"image_{i}"
                )
                current_offset += img.height

            stitched_image.paste(img, paste_position)
            bboxes.append(bbox)
        
        # 将最终的 PIL 图像转回 Tensor
        output_tensor = pil2tensor(stitched_image)

        # 返回拼接后的图像和 BBox 列表
        return (output_tensor, bboxes)
    
    
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
    
    
class FillBBoxWithImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "src_image": ("IMAGE",),
                "bbox": ("BBOX",),
                "fill_image": ("IMAGE",),
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fill_bbox_with_image"
    CATEGORY = "duanyll/bbox"
    
    # Resize fill_image to fill the bbox area
    def fill_bbox_with_image(self, src_image, bbox, fill_image):
        if not isinstance(src_image, torch.Tensor):
            raise ValueError("Input src_image must be a tensor.")
        
        if not isinstance(fill_image, torch.Tensor):
            raise ValueError("Input fill_image must be a tensor.")
        
        if not isinstance(bbox, BoundingBox):
            raise ValueError("Input bbox must be an instance of BoundingBox.")
        
        x_min_c = max(0, bbox.xmin)
        y_min_c = max(0, bbox.ymin)
        x_max_c = min(src_image.shape[2], bbox.xmax)
        y_max_c = min(src_image.shape[1], bbox.ymax)

        if x_min_c >= x_max_c or y_min_c >= y_max_c:
            raise ValueError("Invalid bounding box coordinates.")

        # Resize fill_image to match the bbox size
        fill_height = y_max_c - y_min_c
        fill_width = x_max_c - x_min_c
        
        fill_image_pil = Image.fromarray((fill_image.cpu().numpy() * 255).astype(np.uint8))
        fill_image_resized = fill_image_pil.resize((fill_width, fill_height), Image.Resampling.LANCZOS)
        
        # Convert resized image back to tensor
        fill_tensor = torch.from_numpy(np.array(fill_image_resized).astype(np.float32) / 255.0).permute(2, 0, 1)
        
        # Fill the bbox area in the source image
        src_image[:, y_min_c:y_max_c, x_min_c:x_max_c] = fill_tensor
        
        return (src_image,)