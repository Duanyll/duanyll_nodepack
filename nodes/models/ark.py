import os
import base64
from io import BytesIO
import torch
import numpy as np
from PIL import Image
import requests


def tensor_to_base64(image_tensor: torch.Tensor) -> str:
    """
    将 ComfyUI 的 IMAGE (Tensor) 格式转换为 Base64 编码的 data URL。
    API 需要 'data:image/png;base64,<base64_image>' 格式的字符串。
    """
    # 从批处理中取出第一张图片
    image = image_tensor[0]
    # 将 Tensor 的值从 [0, 1] 范围转换到 [0, 255]
    i = 255.0 * image.cpu().numpy()
    # 从 numpy 数组创建 PIL Image 对象，并转换为 8 位无符号整数
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    # 创建一个内存中的二进制流
    buffered = BytesIO()
    # 将图片以 PNG 格式保存到内存流中
    img.save(buffered, format="PNG")
    # 获取内存流的内容，并进行 base64 编码
    base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # 返回符合格式的 data URL
    return f"data:image/png;base64,{base64_str}"


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


class CreateArkClient:
    """
    创建火山方舟 Ark 客户端的节点。
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "base_url": (
                    "STRING",
                    {"default": "https://ark.cn-beijing.volces.com/api/v3"},
                ),
                "api_key": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("ARK_CLIENT",)
    FUNCTION = "create_client"
    CATEGORY = "duanyll/models/ark"

    def create_client(self, base_url: str, api_key: str):
        """
        初始化并返回 Ark 客户端实例。
        """
        try:
            from volcenginesdkarkruntime import Ark
        except ImportError:
            raise ImportError(
                "请安装 volcengine-python-sdk[ark] 包以使用 Ark API 客户端。运行: pip install volcengine-python-sdk[ark]"
            )

        # 如果节点中没有提供 api_key，则尝试从环境变量 ARK_API_KEY 中获取
        if not api_key:
            api_key = os.environ.get("ARK_API_KEY")
            if not api_key:
                # 如果环境变量中也没有，则抛出异常
                raise ValueError(
                    "API Key 未在节点中提供，也未在环境变量 ARK_API_KEY 中找到。"
                )

        # 初始化客户端
        client = Ark(base_url=base_url, api_key=api_key)
        print("Ark Client created successfully.")
        return (client,)


class SeedEditNode:
    """
    使用 SeedEdit 模型进行图生图的节点。
    """

    @classmethod
    def INPUT_TYPES(s):
        # 支持的模型列表
        supported_models = [
            "doubao-seededit-3-0-i2i-250628",
            # 如果有其他模型，可以加在这里
        ]
        return {
            "required": {
                "client": ("ARK_CLIENT",),
                "image": ("IMAGE",),
                "prompt": (
                    "STRING",
                    {"default": "改成爱心形状的泡泡", "multiline": True},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 123,
                        "min": 0,
                        "max": 2 ** 31 - 1,
                        "control_after_generate": True,
                    },
                ),
            },
            "optional": {
                "model": (supported_models, {"default": supported_models[0]}),
                "size": ("STRING", {"default": "adaptive"}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 5.5, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "watermark": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "duanyll/models/ark"

    def generate_image(
        self,
        client,
        image: torch.Tensor,
        prompt: str,
        seed: int,
        model: str,
        size: str,
        guidance_scale: float,
        watermark: bool,
    ):
        """
        调用 API 生成图片。
        """

        print("Starting SeedEdit image generation...")

        # 1. 将输入图片转换为 Base64
        base64_image = tensor_to_base64(image)
        print(f"Image converted to base64 format.")

        try:
            # 2. 调用 API
            print(f"Calling model '{model}' with seed {seed}...")
            response = client.images.generate(
                model=model,
                prompt=prompt,
                image=base64_image,
                size=size,
                seed=seed,
                guidance_scale=guidance_scale,
                watermark=watermark,
            )

            # 3. 处理返回结果
            if response and response.data and len(response.data) > 0:
                result_url = response.data[0].url
                print(f"Image generated successfully. URL: {result_url}")

                # 4. 从 URL 下载图片并转换为 Tensor
                output_image = url_to_tensor(result_url)
                if output_image is None:
                    raise RuntimeError("从 URL 下载或转换图片失败。")

                return (output_image,)
            else:
                # 如果 API 返回了空数据或无效数据
                error_message = (
                    f"API did not return valid image data. Response: {response}"
                )
                print(error_message)
                raise RuntimeError(error_message)

        except Exception as e:
            # 捕获并打印所有异常
            print(f"An error occurred during image generation: {e}")
            # 抛出异常，ComfyUI 会在界面上显示错误
            raise
