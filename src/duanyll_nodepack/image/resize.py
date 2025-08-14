import torch
import numpy as np
from PIL import Image, ImageOps


class ImagePadToResolution:
    """
    A ComfyUI node to scale an image to fit within one of the specified resolutions by padding it,
    finding the resolution with the most similar aspect ratio.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolutions": (
                    "STRING",
                    {"multiline": True, "default": "1024x1024\n1536x1024\n1024x1536"},
                ),
                "pad_color": (["black", "white"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("PADDED_IMAGE", "SCALED_WIDTH", "SCALED_HEIGHT")
    FUNCTION = "pad_and_resize"
    CATEGORY = "duanyll/image"
    NODE_DISPLAY_NAME = "Pad to Resolution List"

    def tensor_to_pil(self, tensor_image):
        """Converts a torch tensor (B, H, W, C) to a list of PIL Images."""
        return [
            Image.fromarray(np.clip(255.0 * i.cpu().numpy(), 0, 255).astype(np.uint8))
            for i in tensor_image
        ]

    def pil_to_tensor(self, pil_images):
        """Converts a list of PIL Images to a torch tensor (B, H, W, C)."""
        return torch.stack(
            [
                torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
                for img in pil_images
            ]
        )

    def pad_and_resize(self, image: torch.Tensor, resolutions: str, pad_color: str):
        pil_images = self.tensor_to_pil(image)

        # Parse resolutions
        res_lines = resolutions.strip().split("\n")
        target_resolutions = []
        for line in res_lines:
            try:
                w_str, h_str = line.strip().lower().split("x")
                target_resolutions.append((int(w_str), int(h_str)))
            except ValueError:
                print(f"Warning: Could not parse resolution line: '{line}'. Skipping.")
                continue

        if not target_resolutions:
            raise ValueError(
                "No valid resolutions provided. Please use the format 'widthxheight'."
            )

        processed_images = []
        final_scaled_width, final_scaled_height = 0, 0

        for pil_image in pil_images:
            original_width, original_height = pil_image.size
            original_aspect_ratio = original_width / original_height

            # Find the best target resolution by aspect ratio
            best_res = min(
                target_resolutions,
                key=lambda res: abs(res[0] / res[1] - original_aspect_ratio),
            )
            target_width, target_height = best_res

            # Calculate scaling factor to fit inside the target resolution
            width_ratio = target_width / original_width
            height_ratio = target_height / original_height
            scale_factor = min(width_ratio, height_ratio)

            scaled_width = int(round(original_width * scale_factor))
            scaled_height = int(round(original_height * scale_factor))

            # Keep track of the scaled dimensions for output (assuming all images in batch are same size)
            if final_scaled_width == 0:
                final_scaled_width = scaled_width
                final_scaled_height = scaled_height

            # Resize the image
            resized_image = pil_image.resize(
                (scaled_width, scaled_height), Image.LANCZOS
            )

            # Create a new image with the padding color
            color_value = 0 if pad_color == "black" else 255
            padded_image = Image.new(
                "RGB",
                (target_width, target_height),
                (color_value, color_value, color_value),
            )

            # Paste the resized image into the center of the padded image
            paste_x = (target_width - scaled_width) // 2
            paste_y = (target_height - scaled_height) // 2
            padded_image.paste(resized_image, (paste_x, paste_y))

            processed_images.append(padded_image)

        output_tensor = self.pil_to_tensor(processed_images)
        return (output_tensor, final_scaled_width, final_scaled_height)


class ImageCropFromPadded:
    """
    A ComfyUI node to crop a padded image back to its original aspect ratio,
    using the scaled dimensions provided by the padding node.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "padded_image": ("IMAGE",),
                "original_scaled_width": ("INT", {"default": 512, "min": 1}),
                "original_scaled_height": ("INT", {"default": 512, "min": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("RESTORED_IMAGE",)
    FUNCTION = "crop_and_restore"
    CATEGORY = "duanyll/image"
    NODE_DISPLAY_NAME = "Crop from Padded"

    def tensor_to_pil(self, tensor_image):
        return [
            Image.fromarray(np.clip(255.0 * i.cpu().numpy(), 0, 255).astype(np.uint8))
            for i in tensor_image
        ]

    def pil_to_tensor(self, pil_images):
        return torch.stack(
            [
                torch.from_numpy(np.array(img).astype(np.float32) / 255.0)
                for img in pil_images
            ]
        )

    def crop_and_restore(
        self,
        padded_image: torch.Tensor,
        original_scaled_width: int,
        original_scaled_height: int,
    ):
        pil_images = self.tensor_to_pil(padded_image)

        restored_images = []

        for pil_image in pil_images:
            padded_width, padded_height = pil_image.size

            # Calculate the position of the original content
            left = (padded_width - original_scaled_width) // 2
            top = (padded_height - original_scaled_height) // 2
            right = left + original_scaled_width
            bottom = top + original_scaled_height

            # Crop the image
            restored_image = pil_image.crop((left, top, right, bottom))
            restored_images.append(restored_image)

        output_tensor = self.pil_to_tensor(restored_images)
        return (output_tensor,)
