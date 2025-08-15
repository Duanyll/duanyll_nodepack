import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from .basic import BoundingBox
from ..data.any import AnyType

FONT_ROOT = os.path.join(os.path.dirname(__file__), "../../../assets/")

FONT_MAP = {
    "default": os.path.join(FONT_ROOT, "arial.ttf"),
    "bold": os.path.join(FONT_ROOT, "BRITANIC.TTF"),
    "comic": os.path.join(FONT_ROOT, "comic.ttf"),
    "cursive_italic": os.path.join(FONT_ROOT, "VLADIMIR.TTF"),
    "chinese": os.path.join(FONT_ROOT, "hei.TTF"),
}

_font_cache = {}

def get_font(font_name: str, text_height: int) -> ImageFont.FreeTypeFont:
    """
    加载并返回一个Pillow字体对象。
    这个函数是对之前 get_text_width 的扩展，直接返回字体对象。
    """
    font_key = (font_name, text_height)
    if font_key in _font_cache:
        return _font_cache[font_key]
    
    try:
        font_path = FONT_MAP.get(font_name, FONT_MAP["default"])
        font = ImageFont.truetype(font_path, size=text_height)
        _font_cache[font_key] = font
        return font
    except IOError:
        print(f"警告：字体 '{font_name}' 在路径 '{font_path}' 未找到。使用Pillow默认字体。")
        return ImageFont.load_default()

def get_text_width(text: str, font_name: str, text_height: int) -> int:
    font = get_font(font_name, text_height)
    return int(font.getlength(text))


ALL_POSITIONS = [
    "top-left", "top", "top-right",
    "left", "center", "right",
    "bottom-left", "bottom", "bottom-right"
]


def place_text(
    canvas_width: int,
    canvas_height: int,
    text: str,
    text_height: int,
    font_name: str,
    position: str,
    margin: int,
    anchor_box: BoundingBox | None
) -> BoundingBox | None:
    """
    在画布上为文本找到最佳放置位置的 BoundingBox。

    Args:
        canvas_width (int): 画布宽度。
        canvas_height (int): 画布高度。
        text (str): 文本内容。
        text_height (int): 期望的文本高度。
        font_name (str): 字体名称。
        anchor_box (BoundingBox | None): 用于定位的辅助框。如果为 None，则相对于整个画布。
        position (str): 相对位置 ("top", "bottom", "left", "right", "center" 等)。
        margin (int): 文本框与辅助框之间的期望间距。

    Returns:
        BoundingBox | None: 如果成功找到位置，则返回文本框的 BoundingBox；否则返回 None。
    """
    MIN_CANVAS_MARGIN = 20
    MIN_TEXT_HEIGHT = 30

    is_relative_to_canvas = anchor_box is None
    if is_relative_to_canvas:
        # 如果相对于画布，创建一个代表整个画布的虚拟 anchor_box
        # 注意：这里的边距被设为负值，因为文本要放在“内部”
        anchor_box = BoundingBox(0, 0, canvas_width, canvas_height)
        
    current_text_height = text_height

    # 算法会首先尝试满足所有条件，如果空间不足，会进入循环，逐步缩小字体。
    while current_text_height >= MIN_TEXT_HEIGHT:
        text_width = get_text_width(text, font_name, current_text_height)
        
        # 1. 根据位置计算理想的 BoundingBox
        xmin, ymin = 0, 0
        
        # is_relative_to_canvas 标志决定文本是在 anchor_box 外部还是内部
        effective_margin = margin if not is_relative_to_canvas else margin

        # 水平定位
        if position in ["top", "bottom", "center"]:
            xmin = anchor_box.center_x - text_width / 2
        elif position in ["left", "top-left", "bottom-left"]:
            if not is_relative_to_canvas:
                xmin = anchor_box.xmin - effective_margin - text_width
            else:
                xmin = anchor_box.xmin + effective_margin
        elif position in ["right", "top-right", "bottom-right"]:
            if not is_relative_to_canvas:
                xmin = anchor_box.xmax + effective_margin
            else:
                xmin = anchor_box.xmax - effective_margin - text_width

        # 垂直定位
        if position in ["left", "right", "center"]:
            ymin = anchor_box.center_y - current_text_height / 2
        elif position in ["top", "top-left", "top-right"]:
            if not is_relative_to_canvas:
                ymin = anchor_box.ymin - effective_margin - current_text_height
            else:
                ymin = anchor_box.ymin + effective_margin
        elif position in ["bottom", "bottom-left", "bottom-right"]:
            if not is_relative_to_canvas:
                ymin = anchor_box.ymax + effective_margin
            else:
                ymin = anchor_box.ymax - effective_margin - current_text_height

        # 对于 "center" 位置，文本始终在 anchor_box 的中心
        if position == "center":
            xmin = anchor_box.center_x - text_width / 2
            ymin = anchor_box.center_y - current_text_height / 2

        proposed_box = BoundingBox(xmin, ymin, xmin + text_width, ymin + current_text_height, label=text)

        # 2. 检查并调整位置以满足画布边界约束 (MIN_CANVAS_MARGIN)
        # 计算溢出量
        overflow_left = max(0, MIN_CANVAS_MARGIN - proposed_box.xmin)
        overflow_top = max(0, MIN_CANVAS_MARGIN - proposed_box.ymin)
        overflow_right = max(0, proposed_box.xmax - (canvas_width - MIN_CANVAS_MARGIN))
        overflow_bottom = max(0, proposed_box.ymax - (canvas_height - MIN_CANVAS_MARGIN))

        # 通过移动来修正溢出
        final_box = proposed_box
        final_box.xmin += overflow_left - overflow_right
        final_box.xmax += overflow_left - overflow_right
        final_box.ymin += overflow_top - overflow_bottom
        final_box.ymax += overflow_top - overflow_bottom

        # 3. 验证调整后的结果
        # 检查调整后的框是否仍然太大（例如，宽度超过画布总可用宽度）
        if final_box.width > (canvas_width - 2 * MIN_CANVAS_MARGIN) or \
           final_box.height > (canvas_height - 2 * MIN_CANVAS_MARGIN):
            
            # 空间绝对不足，只能缩小字体重试
            current_text_height -= 2 # 递减步长可以调整
            continue

        # 检查相对于 anchor_box 的间距是否被挤压成负数（仅在非画布相对模式下）
        # 如果是负数，意味着即使把 margin 降到 0 也无法放下，必须缩小字体
        is_violated = False
        if not is_relative_to_canvas:
            if position in ["right", "top-right", "bottom-right"] and final_box.xmin < anchor_box.xmax:
                is_violated = True
            if position in ["left", "top-left", "bottom-left"] and final_box.xmax > anchor_box.xmin:
                is_violated = True
            if position in ["bottom", "bottom-left", "bottom-right"] and final_box.ymin < anchor_box.ymax:
                is_violated = True
            if position in ["top", "top-left", "top-right"] and final_box.ymax > anchor_box.ymin:
                is_violated = True
        
        if is_violated:
             # 间距被侵犯，缩小字体重试
            current_text_height -= 2
            continue

        # 如果所有检查都通过，我们找到了一个有效的位置
        return final_box

    # 如果循环结束仍未找到位置，则放置失败
    print(f"警告：无法为文本 '{text}' 找到合适的位置。")
    return None


class GetTextBBoxWithAnchor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", ),
                "height": ("INT", ),
                "text": ("STRING", ),
                "text_height": ("INT", ),
                "font_name": ([key for key in FONT_MAP.keys()], ),
                "position": (ALL_POSITIONS, ),
                "margin": ("INT", )
            },
            "optional": {
                "anchor_box": ("BBOX", )
            }
        }
        
    FUNCTION = "get_text_bbox"
    RETURN_TYPES = ("BBOX", )
    CATEGORY = "duanyll/bbox"

    def get_text_bbox(self, width, height, text, text_height, font_name, position, margin, anchor_box=None):
        # Enforce "chinese" font for Chinese text
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            font_name = "chinese"
        bbox = place_text(width, height, text, text_height, font_name, position, margin, anchor_box)
        return (bbox, )
    
    

def draw_text_with_shadow(
    image: Image.Image,
    text: str,
    bbox: BoundingBox,
    font_name: str,
    text_color: tuple[int, int, int] = (255, 255, 255),
    shadow_color: tuple[int, int, int] = (0, 0, 0),
    shadow_offset: tuple[int, int] = (5, 5),
    blur_radius: int = 5
) -> Image.Image:
    """
    在给定的图片上根据 BoundingBox 绘制带【模糊阴影】的文本。

    Args:
        image (Image.Image): Pillow 图像对象，将在此对象上绘制。
        text (str): 要绘制的文本。
        bbox (BoundingBox): 由 place_text 计算出的文本位置和大小。
        font_name (str): 'default', 'bold' 等字体名称。
        text_color (str | tuple): 文本颜色。
        shadow_color (str | tuple): 阴影颜色。
        shadow_offset (tuple[int, int]): (x, y) 阴影相对于主文本的偏移量。
        blur_radius (int): 阴影的模糊半径，数值越大越模糊。

    Returns:
        Image.Image: 绘制了文本的新图像对象 (RGBA 模式)。
    """
    # 1. 确保主图片为 RGBA 模式，以便进行透明度合成
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    # 2. 创建一个全新的、与主图片同样大小的完全透明的图层，用于绘制阴影
    shadow_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)

    # 3. 加载字体，并在透明图层上绘制阴影文本
    # 字体大小由最终的 bbox 高度决定
    font_size = bbox.height
    font = get_font(font_name, font_size) # 假设 get_font 函数已定义

    shadow_position = (bbox.xmin + shadow_offset[0], bbox.ymin + shadow_offset[1])
    shadow_draw.text(shadow_position, text, font=font, fill=shadow_color)

    # 4. 对整个阴影图层应用高斯模糊滤镜
    blurred_shadow_layer = shadow_layer.filter(
        ImageFilter.GaussianBlur(radius=blur_radius)
    )

    # 5. 使用 alpha_composite 将模糊的阴影图层合成到原始图片上
    # 这会将模糊的阴影“贴”在背景上
    image_with_shadow = Image.alpha_composite(image, blurred_shadow_layer)

    # 6. 最后，在合成后的图片上绘制清晰的主文本
    final_draw = ImageDraw.Draw(image_with_shadow)
    text_position = (bbox.xmin, bbox.ymin)
    final_draw.text(text_position, text, font=font, fill=text_color)
    
    # 7. 返回最终的图像
    # 注意：返回的图像将是 RGBA 模式。如果需要保存为 JPG，
    # 可能需要转换回 RGB: final_image.convert('RGB')
    return image_with_shadow.convert('RGB') 


def tensor2pil(image: torch.Tensor) -> list[Image.Image]:
    """将输入的 torch.Tensor 转换为 PIL.Image.Image 列表"""
    return [
        Image.fromarray(np.clip(255.0 * i.cpu().numpy(), 0, 255).astype(np.uint8))
        for i in image
    ]


def pil2tensor(image: Image.Image) -> torch.Tensor:
    """将单个 PIL.Image.Image 转换为 torch.Tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class DrawTextInBBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "text": ("STRING", ),
                "bbox": ("BBOX", ),
                "font_name": ([key for key in FONT_MAP.keys()], ),
                "text_color": (AnyType("*"), ),
            }
        }

    FUNCTION = "draw_text_in_bbox"
    RETURN_TYPES = ("IMAGE", )
    CATEGORY = "duanyll/bbox"

    def draw_text_in_bbox(self, image, text, bbox, font_name, text_color):
        if bbox is None:
            print("## ERROR: Bounding box is None. Cannot draw text.")
            return (image,)
        image_pil = tensor2pil(image)[0]
        if text_color is None:
            text_color = (255, 255, 255)
        else:
            text_color = tuple(int(c) for c in text_color)
        image_pil = draw_text_with_shadow(
            image=image_pil,
            text=text,
            bbox=bbox,
            font_name=font_name,
            text_color=text_color
        )
        return (pil2tensor(image_pil), )