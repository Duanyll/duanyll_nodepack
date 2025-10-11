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


class DrawBoundingBoxesQwen:
    """
    一个ComfyUI节点，用于在图像上根据JSON输入绘制边界框、点和标签。
    - 支持 bbox_2d (边界框) 和 point_2d (点)。
    - 兼容 str, int, float 类型的坐标。
    - 智能的标签定位，确保可见性。
    """
    @classmethod
    def INPUT_TYPES(cls):
        """
        定义节点的输入参数。
        """
        return {
            "required": {
                "image": ("IMAGE",),
                "json_string": ("STRING", {
                    "multiline": True,
                    # ### 改进点: 默认值现在包含两种类型的数据 ###
                    "default": '[{"bbox_2d": [170, 10, 780, 830], "label": "face"}, \n {"point_2d": ["394", "105"], "label": "eye"}]'
                }),
                "draw_mode": (["Outline", "Fill"],), 
                "line_thickness": ("INT", {
                    "default": 4, "min": 1, "max": 100, "step": 1, "display": "number"
                }),
                # ### 改进点: 新增点半径参数 ###
                "point_radius": ("INT", {
                    "default": 10, "min": 1, "max": 100, "step": 1, "display": "number"
                }),
                "fill_opacity": ("FLOAT", {
                    "default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"
                }),
                "draw_label": ("BOOLEAN", {
                    "default": True, "label_on": "enabled", "label_off": "disabled"
                }),
                "font_size": ("INT", {
                    "default": 25, "min": 8, "max": 200, "step": 1, "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw_on_image"
    CATEGORY = "duanyll/deprecated"
    
    # ### 改进点: 更新描述 ###
    DESCRIPTION = "Draw bounding boxes, points, and labels on images from JSON data (e.g., Qwen-VL)."

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        image_np = tensor.cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        if image_np.ndim == 2:
            return Image.fromarray(image_np, 'L').convert('RGB')
        return Image.fromarray(image_np, 'RGB')

    def _pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0)

    # ### 改进点: 新增一个通用的坐标解析和验证函数 ###
    def _parse_coords(self, coords_list, expected_length):
        """将任何类型的坐标列表 (str, int, float) 转换为整数坐标列表。"""
        if not isinstance(coords_list, list) or len(coords_list) != expected_length:
            return None
        try:
            # 使用 int(float(c)) 来处理所有数字类型
            return [int(float(c)) for c in coords_list]
        except (ValueError, TypeError):
            # 如果转换失败（例如，包含非数字字符串），则返回 None
            return None

    def draw_on_image(self, image: torch.Tensor, json_string: str, draw_mode: str, line_thickness: int, point_radius: int, fill_opacity: float, draw_label: bool, font_size: int):
        try:
            data_list = json.loads(json_string.strip())
            if not isinstance(data_list, list):
                print(f"Warning: [Draw Bounding Things] JSON data is not a list. Returning original image.")
                return (image,)
        except json.JSONDecodeError:
            print(f"Warning: [Draw Bounding Things] Invalid JSON format. Returning original image.")
            return (image,)

        processed_tensors = []
        
        try:
            font = ImageFont.truetype(FONT_PATH, font_size) if FONT_PATH else ImageFont.load_default()
        except IOError:
            print("Warning: [Draw Bounding Things] Custom font not found. Using default font.")
            font = ImageFont.load_default()

        for img_tensor in image:
            pil_img = self._tensor_to_pil(img_tensor)
            img_width, img_height = pil_img.size
            
            # 准备一个单独的透明层用于绘制，最后统一合成
            # 这样可以正确处理轮廓和填充模式，并确保标签总在最上层
            overlay_layer = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay_layer)

            for item in data_list:
                if not isinstance(item, dict):
                    continue

                label_ref_box = None # 用于标签定位的参考框
                
                # --- 处理 bbox_2d ---
                if "bbox_2d" in item:
                    box = self._parse_coords(item["bbox_2d"], 4)
                    if not box:
                        print(f"Warning: [Draw Bounding Things] Skipping invalid bbox_2d: {item['bbox_2d']}")
                        continue

                    x_min, y_min, x_max, y_max = box
                    x_min_c = max(0, x_min)
                    y_min_c = max(0, y_min)
                    x_max_c = min(img_width - 1, x_max)
                    y_max_c = min(img_height - 1, y_max)
                    
                    if x_min_c >= x_max_c or y_min_c >= y_max_c:
                        continue
                    
                    clipped_box = [x_min_c, y_min_c, x_max_c, y_max_c]
                    label_ref_box = clipped_box  # Bbox本身就是标签的参考框

                    if draw_mode == "Outline":
                        draw.rectangle(clipped_box, outline="red", width=line_thickness)
                    else: # Fill mode
                        alpha = int(fill_opacity * 255)
                        draw.rectangle(clipped_box, fill=(255, 0, 0, alpha))

                # --- 处理 point_2d ---
                elif "point_2d" in item:
                    point = self._parse_coords(item["point_2d"], 2)
                    if not point:
                        print(f"Warning: [Draw Bounding Things] Skipping invalid point_2d: {item['point_2d']}")
                        continue
                    
                    px, py = point
                    if not (0 <= px < img_width and 0 <= py < img_height):
                        continue

                    # 将点转换为一个用于绘制圆和定位标签的框
                    circle_box = [px - point_radius, py - point_radius, px + point_radius, py + point_radius]
                    label_ref_box = circle_box # 点的参考框就是这个圆的外接矩形

                    if draw_mode == "Outline":
                        draw.ellipse(circle_box, outline="red", width=line_thickness)
                    else: # Fill mode
                        alpha = int(fill_opacity * 255)
                        draw.ellipse(circle_box, fill=(255, 0, 0, alpha))
                
                # --- 绘制标签 (如果存在参考框) ---
                if label_ref_box and draw_label and "label" in item:
                    label = str(item["label"])
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
        return (output_image,)
    
    
class CreateBoundingBoxesMaskQwen:
    """
    一个ComfyUI节点，用于根据JSON数据创建一个黑白蒙版。
    """
    @classmethod
    def INPUT_TYPES(cls):
        """
        定义节点的输入参数。
        """
        return {
            "required": {
                "json_string": ("STRING", {
                    "multiline": True,
                    "default": '[{"bbox_2d": [170, 10, 780, 830], "label": "face"}]'
                }),
                "width": ("INT", {
                    "default": 512, "min": 64, "max": 8192, "step": 8, "display": "number"
                }),
                "height": ("INT", {
                    "default": 512, "min": 64, "max": 8192, "step": 8, "display": "number"
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "create_mask"
    CATEGORY = "duanyll/deprecated"
    
    DESCRIPTION = "Create a black and white mask from JSON bounding box data. The mask will be white in the areas defined by bbox_2d and black elsewhere."

    def create_mask(self, json_string: str, width: int, height: int):
        # 尝试解析JSON
        try:
            bbox_data = json.loads(json_string.strip())
            if not isinstance(bbox_data, list):
                print(f"Warning: [Create Mask from JSON] JSON data is not a list. Returning empty mask.")
                bbox_data = []
        except json.JSONDecodeError:
            print(f"Warning: [Create Mask from JSON] Invalid JSON format. Returning empty mask.")
            bbox_data = []
        
        # 创建一个全黑的蒙版（使用numpy数组，效率更高）
        # 形状为 (height, width)，数据类型为32位浮点数
        mask_np = np.zeros((height, width), dtype=np.float32)

        # 遍历所有坐标框
        for item in bbox_data:
            if not isinstance(item, dict) or "bbox_2d" not in item:
                continue

            box = item["bbox_2d"]
            if not isinstance(box, list) or len(box) != 4:
                print(f"Warning: [Create Mask from JSON] Skipping invalid bbox_2d: {box}")
                continue

            x_min, y_min, x_max, y_max = box

            # 裁剪坐标以确保它们在指定的 width 和 height 范围内
            # 注意：numpy切片的上边界是“不包含”的，所以用 width/height 而不是 width-1/height-1
            x_start = max(0, x_min)
            y_start = max(0, y_min)
            x_end = min(width, x_max)
            y_end = min(height, y_max)
            
            # 如果区域无效，则跳过
            if x_start >= x_end or y_start >= y_end:
                continue
            
            # 使用numpy切片将指定区域设置为1.0（白色）
            # 格式为 [y_start:y_end, x_start:x_end]
            mask_np[y_start:y_end, x_start:x_end] = 1.0
        
        # 将numpy数组转换为PyTorch张量
        mask_tensor = torch.from_numpy(mask_np)
        
        # ComfyUI的MASK格式要求是 (batch_size, height, width)
        # 所以我们需要在最前面增加一个维度
        mask_tensor = mask_tensor.unsqueeze(0)

        return (mask_tensor, )