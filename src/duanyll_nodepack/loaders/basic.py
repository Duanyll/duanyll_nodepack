import os
from io import BytesIO
import torch
import requests
from PIL import Image
import numpy as np

import folder_paths

def url_to_tensor(image_url: str) -> torch.Tensor:
    """
    从 URL 下载图片并将其转换为 ComfyUI 的 IMAGE (Tensor) 格式。
    """
    try:
        # 发送 GET 请求下载图片
        response = requests.get(image_url, timeout=20)
        # 检查请求是否成功
        response.raise_for_status()

        # 从响应内容中打开图片
        image = Image.open(BytesIO(response.content))
        # 将图片转换为 RGB 格式，以确保通道数正确
        image = image.convert("RGB")

        # 将 PIL Image 转换为 numpy 数组，并将值范围归一化到 [0, 1]
        np_image = np.array(image).astype(np.float32) / 255.0
        # 将 numpy 数组转换为 torch Tensor
        tensor = torch.from_numpy(np_image)

        # 在最前面增加一个批处理维度 (batch dimension)，符合 ComfyUI 的格式 [B, H, W, C]
        return tensor.unsqueeze(0)

    except requests.RequestException as e:
        print(f"下载图片失败: {e}")
        return None
    

class DownloadImageFromUrl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_url": ("STRING", {"default": "", "multiline": False}),
            }
        }
        
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "download_image"
    CATEGORY = "duanyll/loaders"

    def download_image(self, image_url: str):
        """
        下载图片并转换为 Tensor 格式。
        """
        return (url_to_tensor(image_url), )
    
    
class ReadTextFile:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dir": ("STRING", {"default": "input", "multiline": False}),
                "file": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "encoding": ("STRING", {"default": "utf-8", "multiline": False}),
                "ignore_errors": ("BOOLEAN", {"default": False})
            }
        }
        
    RETURN_TYPES = ("STRING", )
    FUNCTION = "read_text_file"
    CATEGORY = "duanyll/loaders"

    def read_text_file(self, dir: str, file: str, encoding, ignore_errors) -> str:
        fullpath = os.path.join(folder_paths.base_path, dir, file)
        try:
            with open(fullpath, "r", encoding=encoding, errors="ignore" if ignore_errors else "strict") as f:
                return (f.read(), )
        except:
            if not ignore_errors:
                raise