import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json

class DrawBoundingBoxes:
    """
    一个ComfyUI节点，用于在图像上根据JSON输入绘制边界框和标签。
    新增功能：坐标裁剪、半透明填充模式。
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
                    "default": '[{"bbox_2d": [170, 10, 780, 830], "label": "face"}]'
                }),
                "draw_mode": (["Outline", "Fill"],), # 新增：绘制模式下拉菜单
                "line_thickness": ("INT", {
                    "default": 4, "min": 1, "max": 100, "step": 1, "display": "number"
                }),
                "fill_opacity": ("FLOAT", { # 新增：填充透明度滑块
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
    CATEGORY = "Image/Drawing"

    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        image_np = tensor.cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        return Image.fromarray(image_np, 'RGB')

    def _pil_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        return torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0)

    def draw_on_image(self, image: torch.Tensor, json_string: str, draw_mode: str, line_thickness: int, fill_opacity: float, draw_label: bool, font_size: int):
        # 解析JSON
        try:
            bbox_data = json.loads(json_string.strip())
            if not isinstance(bbox_data, list):
                print(f"Warning: [Draw Bounding Boxes] JSON data is not a list. Returning original image.")
                return (image,)
        except json.JSONDecodeError:
            print(f"Warning: [Draw Bounding Boxes] Invalid JSON format. Returning original image.")
            return (image,)

        processed_tensors = []
        
        # 加载字体
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except IOError:
            print("Warning: [Draw Bounding Boxes] Arial font not found. Using default font.")
            font = ImageFont.load_default()

        # 循环处理批次中的每张图片
        for img_tensor in image:
            pil_img = self._tensor_to_pil(img_tensor)
            img_width, img_height = pil_img.size

            # 如果是填充模式，需要创建一个新的RGBA层来绘制半透明物体
            if draw_mode == "Fill":
                # 创建一个与原图等大且完全透明的图层
                fill_layer = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
                draw_fill = ImageDraw.Draw(fill_layer)
            
            draw = ImageDraw.Draw(pil_img) # 用于绘制不透明的标签文本

            # 遍历JSON中的每个对象
            for item in bbox_data:
                if not isinstance(item, dict) or "bbox_2d" not in item:
                    continue

                box = item["bbox_2d"]
                if not isinstance(box, list) or len(box) != 4:
                    print(f"Warning: [Draw Bounding Boxes] Skipping invalid bbox_2d: {box}")
                    continue

                x_min, y_min, x_max, y_max = box

                # **新增功能: 裁剪坐标**
                # 将坐标限制在图片边界 (0, 0) 到 (width-1, height-1) 之内
                x_min_clipped = max(0, x_min)
                y_min_clipped = max(0, y_min)
                x_max_clipped = min(img_width - 1, x_max)
                y_max_clipped = min(img_height - 1, y_max)
                
                # 如果裁剪后区域无效（例如，x_min > x_max），则跳过绘制
                if x_min_clipped >= x_max_clipped or y_min_clipped >= y_max_clipped:
                    continue
                
                clipped_box = [x_min_clipped, y_min_clipped, x_max_clipped, y_max_clipped]

                # 根据模式进行绘制
                if draw_mode == "Outline":
                    draw.rectangle(clipped_box, outline="red", width=line_thickness)
                elif draw_mode == "Fill":
                    # 计算带透明度的填充色
                    alpha = int(fill_opacity * 255)
                    fill_color = (255, 0, 0, alpha) # R, G, B, Alpha
                    # 在透明图层上绘制填充矩形
                    draw_fill.rectangle(clipped_box, fill=fill_color)

                # 绘制标签文本（在最上层绘制，保证可见）
                if draw_label and "label" in item:
                    label = item["label"]
                    text_x = clipped_box[0]
                    text_y = clipped_box[1] - font_size - 2
                    if text_y < 0:
                        text_y = clipped_box[1] + 2
                    
                    # 标签文本直接绘制在原图的 Draw 对象上
                    draw.text((text_x, text_y), label, fill="red", font=font)

            # 如果是填充模式，将绘制好的透明图层合成到原图上
            if draw_mode == "Fill":
                # 需要先将原图转为RGBA才能进行alpha合成
                pil_img_rgba = pil_img.convert('RGBA')
                # 将填充图层叠加到图像上
                combined_img = Image.alpha_composite(pil_img_rgba, fill_layer)
                # 转回RGB以符合ComfyUI的IMAGE格式
                pil_img = combined_img.convert('RGB')
            
            processed_tensors.append(self._pil_to_tensor(pil_img))

        output_image = torch.stack(processed_tensors)
        return (output_image,)