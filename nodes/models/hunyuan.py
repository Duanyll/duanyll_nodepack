import torch
import numpy as np
import json
import requests
import base64
import random
from PIL import Image
import io


class VllmHunyuanImage3Node:
    """
    ComfyUI node for calling vLLM HunyuanImage3 API
    """
    
    ASPECT_RATIOS = {
        "1:1": "1024x1024",
        "4:3": "1152x896",
        "3:4": "896x1152",
        "16:9": "1280x768",
        "9:16": "768x1280",
        "21:9": "1408x640",
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "一张电影海报，主体是一个穿着皮夹克和牛仔裤的亚洲女孩，帅气地望着前方，手里拿着一朵小红花。"
                }),
                "aspect_ratio": (list(cls.ASPECT_RATIOS.keys()), {
                    "default": "1:1"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0xffffffff
                }),
                "diff_infer_steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "bot_task": (["image", "auto"], {
                    "default": "image"
                }),
                "use_system_prompt": (["None", "dynamic", "en_vanilla", "en_recaption", "en_think_recaption", "custom"], {
                    "default": "None"
                }),
                "api_url": ("STRING", {
                    "default": "http://127.0.0.1:8000/v1/chat/completions"
                }),
                "model_name": ("STRING", {
                    "default": "vllm_hunyuan_image3"
                }),
            },
            "optional": {
                "system_prompt": ("STRING", ),
                "temperature": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "duanyll/models"
    
    def generate_image(self, prompt, aspect_ratio, seed, diff_infer_steps, 
                      bot_task, use_system_prompt, api_url, model_name,
                      system_prompt="", temperature=0.0):
        """
        Generate image using vLLM HunyuanImage3 API
        """
        
        # Get image size from aspect ratio
        image_size = self.ASPECT_RATIOS[aspect_ratio]
        
        # Build chat template based on bot_task
        if bot_task == "image":
            chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}"
                "<|startoftext|>{{ message['content'] }}"
                "{% endif %}"
                "{% endfor %}"
            )
        else:  # auto
            chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}"
                "<|startoftext|>{{ message['content'] }}<boi><image_shape_1024>"
                "{% endif %}"
                "{% endfor %}"
            )
        
        # Build task extra kwargs
        task_extra_kwargs = {
            "diff_infer_steps": diff_infer_steps,
            "use_system_prompt": use_system_prompt,
            "bot_task": bot_task,
            "image_size": image_size,
        }
        
        # Build system message
        system_content = system_prompt if use_system_prompt == "custom" else ""
        
        # Build payload
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ],
            "max_completion_tokens": 1,
            "temperature": temperature,
            "seed": seed,
            "chat_template": chat_template,
            "task_type": "hunyuan_image3",
            "task_extra_kwargs": task_extra_kwargs,
        }
        
        try:
            # Make API request
            headers = {"Content-Type": "application/json"}
            response = requests.post(api_url, data=json.dumps(payload), 
                                   headers=headers, timeout=10000)
            
            if response.status_code != 200:
                raise Exception(f"API Error {response.status_code}: {response.text}")
            
            # Parse response
            data = response.json()
            base64_image = data['image']
            
            # Remove possible data:image/png;base64, prefix
            if ',' in base64_image:
                base64_image = base64_image.split(',')[1]
            
            # Decode image
            image_data = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert PIL Image to ComfyUI format (tensor)
            image_np = np.array(image).astype(np.float32) / 255.0
            
            # Ensure RGB format
            if len(image_np.shape) == 2:  # Grayscale
                image_np = np.stack([image_np] * 3, axis=-1)
            elif image_np.shape[2] == 4:  # RGBA
                image_np = image_np[:, :, :3]
            
            # Convert to torch tensor with shape [1, H, W, C]
            image_tensor = torch.from_numpy(image_np)[None,]
            
            print(f"[VLLMHunyuanImage3] Successfully generated image with seed: {seed}")
            
            return (image_tensor,)
            
        except Exception as e:
            print(f"[VLLMHunyuanImage3] Error: {str(e)}")
            # Return a black image on error
            error_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            return (error_image,)