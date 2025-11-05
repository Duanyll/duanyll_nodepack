from io import BytesIO
from PIL import Image
import base64
import torch
import requests
from typing import Literal, TypedDict, List, Union, Dict, Any

from .data.any import AnyType

Role = Literal["system", "user", "assistant"]


class TextContent(TypedDict):
    type: Literal["text"]
    text: str


class ImageUrl(TypedDict):
    url: str


class ImageContent(TypedDict):
    type: Literal["image_url"]
    image_url: ImageUrl
    image_tensor: torch.Tensor


class Message(TypedDict):
    role: Role
    content: List[Union[TextContent, ImageContent]]


class LlmClient:
    api_key: str = ""
    """API key"""
    base_url: str = "https://api.openai.com/v1"
    """OpenAI API compatible endpoint URL base"""
    model: str
    """Model name to use"""
    model_options: dict = {}
    """Additional options for the API call"""
    image_max_pixels: int = 1200 * 1200
    """Maximum number of pixels for image image inputs"""
    timeout: int = 120

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model="",
        model_options={},
        timeout: int = 120,
        image_max_pixels: int = 1200 * 1200,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        if model is None or model == "" or model == "auto":
            model = self.detect_model()
        self.model = model
        self.model_options = model_options
        self.timeout = timeout
        self.image_max_pixels = image_max_pixels

    def get_headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def detect_model(self):
        # Query available models from the API and select first one
        url = f"{self.base_url}/models"
        response = requests.get(url, headers=self.get_headers())
        response.raise_for_status()
        models = response.json().get("data", [])
        if not models:
            raise ValueError("No models available from the API.")
        return models[0]["id"]

    def tensor_to_base64_url(self, image_tensor: torch.Tensor) -> str:
        # Image tensor is expected to be in CHW format with values in [0, 1]
        image_array = (image_tensor.cpu().numpy() * 255).astype("uint8")
        image = Image.fromarray(image_array)
        # Limit image size: total pixels should not exceed image_max_pixels. Resize if necessary.
        width, height = image.size
        total_pixels = width * height
        if total_pixels > self.image_max_pixels:
            scale_factor = (self.image_max_pixels / total_pixels) ** 0.5
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        # Convert image to base64-encoded PNG
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

    def chat_completion(self, messages: List[Message]) -> str:
        for msg in messages:
            for content in msg["content"]:
                if content["type"] == "image_url" and "image_tensor" in content:
                    content["image_url"] = {
                        "url": self.tensor_to_base64_url(content["image_tensor"])
                    }
                    del content["image_tensor"]

        url = f"{self.base_url}/chat/completions"
        payload = {"model": self.model, "messages": messages, **self.model_options}
        response = requests.post(
            url, headers=self.get_headers(), json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        res = response.json()
        return res["choices"][0]["message"]["content"]


class LlmCreateClient:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False}),
                "base_url": (
                    "STRING",
                    {"default": "https://api.openai.com/v1", "multiline": False},
                ),
                "model": ("STRING", {"default": "auto", "multiline": False}),
                "timeout": ("INT", {"default": 120, "min": 1, "max": 600, "step": 1}),
                "image_max_pixels": (
                    "INT",
                    {"default": 1200 * 1200, "min": 256 * 256, "max": 4096 * 4096, "step": 1},
                ),
            },
            "optional": {
                "model_options": (AnyType("*"),),
            },
        }

    RETURN_TYPES = ("LLM_CLIENT",)
    FUNCTION = "create"
    CATEGORY = "duanyll/llm"

    def create(
        self, api_key: str, base_url, model, timeout, image_max_pixels, model_options={}
    ):
        model_options = model_options or {}
        client = LlmClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            model_options=model_options,
            timeout=timeout,
            image_max_pixels=image_max_pixels,
        )
        return (client,)


class LlmClientSetSeed:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("LLM_CLIENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
            }
        }

    RETURN_TYPES = ("LLM_CLIENT",)
    FUNCTION = "set_seed"
    CATEGORY = "duanyll/llm"

    def set_seed(self, client: LlmClient, seed: int):
        client.model_options["seed"] = seed
        return (client,)


class LlmCreateChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": (
                    "STRING",
                    {"multiline": True, "default": "You are a helpful assistant."},
                ),
            }
        }

    RETURN_TYPES = ("CHAT",)
    FUNCTION = "create"
    CATEGORY = "duanyll/llm"

    def create(self, system_prompt: str):
        chat = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        ]

        return (chat,)


class LlmChatAddMessage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chat": ("CHAT",),
                "user_message": ("STRING", {"multiline": True}),
                "assistant_response": ("STRING", {"multiline": True}),
            },
            "optional": {
                "images": ("IMAGE",),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("CHAT",)
    FUNCTION = "add_message"
    CATEGORY = "duanyll/llm"

    def add_message(self, chat, user_message, assistant_response, images=[]):
        chat = list(chat[0])  # INPUT_IS_LIST = True
        user_message = user_message[0]
        assistant_response = assistant_response[0]

        chat.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": user_message}]
                + [{"type": "image_url", "image_tensor": image[0]} for image in images],
            }
        )

        chat.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_response}],
            }
        )

        return (chat,)


class LlmChatCompletion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("LLM_CLIENT",),
                "chat": ("CHAT",),
                "user_message": ("STRING", {"multiline": True}),
            },
            "optional": {
                "images": ("IMAGE",),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING", "CHAT")
    FUNCTION = "run"
    CATEGORY = "duanyll/llm"

    def run(self, client, chat, user_message, images=[]):
        client = client[0]
        chat = list(chat[0])
        user_message = user_message[0]

        chat.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": user_message}]
                + [{"type": "image_url", "image_tensor": image[0]} for image in images],
            }
        )

        response_text = client.chat_completion(chat)

        chat.append(
            {"role": "assistant", "content": [{"type": "text", "text": response_text}]}
        )

        return (response_text, chat)
