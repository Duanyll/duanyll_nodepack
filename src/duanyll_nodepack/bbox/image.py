import torch
from PIL import Image
import numpy as np

from .basic import BoundingBox


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
    return [
        Image.fromarray(np.clip(255.0 * i.cpu().numpy(), 0, 255).astype(np.uint8))
        for i in image
    ]


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
                "size": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8192, "step": 1},
                ),  # 控制统一的宽度或高度
            }
        }

    RETURN_TYPES = ("IMAGE", "BBOX")
    RETURN_NAMES = ("stitched_image", "bbox")
    FUNCTION = "stitch_images"
    CATEGORY = "duanyll/bbox"  # 您可以自定义节点的分类

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
            if direction == "right":  # 向右拼接，统一高度
                target_size = max(img.height for img in pil_images)
            else:  # 向下拼接，统一宽度
                target_size = max(img.width for img in pil_images)

        # --------------------------------------------------
        # 2. 等比缩放所有图片
        # --------------------------------------------------
        resized_images = []
        for img in pil_images:
            original_width, original_height = img.width, img.height
            if direction == "right":  # 统一高度
                new_height = target_size
                aspect_ratio = original_width / original_height
                new_width = int(new_height * aspect_ratio)
            else:  # 统一宽度
                new_width = target_size
                aspect_ratio = original_height / original_width
                new_height = int(new_width * aspect_ratio)

            # 使用高质量的 LANCZOS 滤波器进行缩放
            resized_images.append(
                img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            )

        # --------------------------------------------------
        # 3. 计算拼接后大图的总尺寸
        # --------------------------------------------------
        if direction == "right":
            total_width = sum(img.width for img in resized_images)
            max_height = target_size
            canvas_size = (total_width, max_height)
        else:  # direction == "down"
            total_height = sum(img.height for img in resized_images)
            max_width = target_size
            canvas_size = (max_width, total_height)

        # --------------------------------------------------
        # 4. 创建画布并拼接图像，同时生成 BBox
        # --------------------------------------------------
        # 创建一个 'RGB' 模式的空白画布
        stitched_image = Image.new("RGB", canvas_size)
        bboxes = []
        current_offset = 0  # 当前的偏移量（x 或 y）

        for i, img in enumerate(resized_images):
            if direction == "right":
                paste_position = (current_offset, 0)
                # 创建 BoundingBox
                bbox = BoundingBox(
                    xmin=current_offset,
                    ymin=0,
                    xmax=current_offset + img.width,
                    ymax=img.height,
                    label=f"image_{i}",  # 使用索引作为默认标签
                )
                current_offset += img.width
            else:  # direction == "down"
                paste_position = (0, current_offset)
                # 创建 BoundingBox
                bbox = BoundingBox(
                    xmin=0,
                    ymin=current_offset,
                    xmax=img.width,
                    ymax=current_offset + img.height,
                    label=f"image_{i}",
                )
                current_offset += img.height

            stitched_image.paste(img, paste_position)
            bboxes.append(bbox)

        # 将最终的 PIL 图像转回 Tensor
        output_tensor = pil2tensor(stitched_image)

        # 返回拼接后的图像和 BBox 列表
        return (output_tensor, bboxes)
    
    
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

        fill_image_pil = Image.fromarray(
            (fill_image.cpu().numpy() * 255).astype(np.uint8)
        )
        fill_image_resized = fill_image_pil.resize(
            (fill_width, fill_height), Image.Resampling.LANCZOS
        )

        # Convert resized image back to tensor
        fill_tensor = torch.from_numpy(
            np.array(fill_image_resized).astype(np.float32) / 255.0
        ).permute(2, 0, 1)

        # Fill the bbox area in the source image
        src_image[:, y_min_c:y_max_c, x_min_c:x_max_c] = fill_tensor

        return (src_image,)
