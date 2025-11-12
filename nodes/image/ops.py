import torch
from einops import repeat, rearrange, reduce
import numpy as np
import matplotlib.pyplot as plt

class ImageDifferenceCmap:
    CMAP_OPTIONS = [
        'viridis', 'plasma', 'inferno', 'magma', 'cividis', # Perceptually Uniform
        'jet', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter', # Sequential
        'gray', 'bone', 'pink', # Grayscale
        'bwr', 'coolwarm', 'seismic' # Diverging
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "cmap": (s.CMAP_OPTIONS, {"default": 'viridis'}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("difference",)
    FUNCTION = "process"
    CATEGORY = "duanyll/image"
    DESCRIPTION = "Calculate the difference between two images and apply a colormap to visualize the difference. Input images should have the same shape."
    
    def process(self, image1: torch.Tensor, image2: torch.Tensor, cmap: str):
        """
        Processes the two input images to produce a colormapped difference image.

        Args:
            image1 (torch.Tensor): The first input image tensor (Batch, Height, Width, Channels).
            image2 (torch.Tensor): The second input image tensor.
            cmap (str): The name of the colormap to apply.

        Returns:
            A tuple containing the resulting image tensor.
        """
        # Ensure tensors are on the CPU for processing with NumPy/Matplotlib
        image1_cpu = image1.cpu()
        image2_cpu = image2.cpu()

        # --- 1. Validate shapes ---
        if image1_cpu.shape != image2_cpu.shape:
            print(f"## ERROR: Input images must have the same shape. "
                  f"Got {image1_cpu.shape} and {image2_cpu.shape}.")
            # Return one of the original images as a fallback to prevent crashing
            return (image1,)

        # --- 2. Calculate the difference ---
        # The pixel values are floats in [0, 1], so the difference is in [-1, 1]
        diff_tensor = image1_cpu - image2_cpu
        
        # --- 3. Calculate vector magnitude (L2 norm) ---
        # torch.norm calculates the norm across the channel dimension (dim=-1)
        # The result is a single-channel tensor of shape (Batch, Height, Width)
        magnitude_tensor = torch.norm(diff_tensor, p=2, dim=-1)

        # --- 4. Normalize the magnitude to [0, 1] for the colormap ---
        # We handle the case where the max magnitude is zero (identical images)
        min_val = torch.min(magnitude_tensor)
        max_val = torch.max(magnitude_tensor)
        
        if max_val > min_val:
            magnitude_normalized = (magnitude_tensor - min_val) / (max_val - min_val)
        else:
            magnitude_normalized = torch.zeros_like(magnitude_tensor)

        # --- 5. Apply the colormap ---
        # Matplotlib's colormaps work with NumPy arrays
        # The input shape for colormap is (H, W) or (B, H, W)
        magnitude_numpy = magnitude_normalized.numpy()

        try:
            colormap = plt.get_cmap(cmap)
            # The colormap function returns a (B, H, W, 4) RGBA NumPy array
            colored_image_numpy = colormap(magnitude_numpy)
        except ValueError:
            print(f"## ERROR: Invalid colormap name '{cmap}'. Defaulting to 'viridis'.")
            colormap = plt.get_cmap('viridis')
            colored_image_numpy = colormap(magnitude_numpy)

        # --- 6. Convert back to a ComfyUI-compatible tensor ---
        # Remove the alpha channel (last dimension) to get RGB
        colored_image_rgb = colored_image_numpy[..., :3]

        # Convert the NumPy array back to a torch.Tensor with float type
        output_tensor = torch.from_numpy(colored_image_rgb).float()

        # The final shape should be (Batch, Height, Width, Channels)
        return (output_tensor,)
    
    
class ImageLinstretch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "tol_low": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tol_high": ("FLOAT", {"default": 0.99, "min": 0.0, "max": 1.0, "step": 0.01}),
                "clip": ("BOOLEAN", {"default": True}),
                "mode": (["per_image", "per_channel"], {"default": "per_image"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process"
    CATEGORY = "duanyll/image"
    DESCRIPTION = "Apply linear stretch to the input image to enhance contrast."

    def process(self, image: torch.Tensor, tol_low: float, tol_high: float, clip: bool, mode: str):
        """
        Applies linear stretch to the input image.

        Args:
            image (torch.Tensor): The input image tensor (Batch, Height, Width, Channels).

        Returns:
            A tuple containing the linstretched image tensor.
        """
        if mode == "per_image":
            # Compute global min and max across all channels
            min_val = torch.quantile(image, tol_low)
            max_val = torch.quantile(image, tol_high)
        elif mode == "per_channel":
            # Compute min and max per channel
            min_val = torch.quantile(image, tol_low, dim=(1, 2), keepdim=True)
            max_val = torch.quantile(image, tol_high, dim=(1, 2), keepdim=True)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        # Apply linear stretch
        scale = max_val - min_val
        scale[scale < 1e-6] = 1.0  # Prevent division by zero

        linstretched_image = (image - min_val) / scale
        if clip:
            linstretched_image = torch.clamp(linstretched_image, 0.0, 1.0)

        return (linstretched_image,)
    
    
class MaskLinstretch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "tol_low": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tol_high": ("FLOAT", {"default": 0.99, "min": 0.0, "max": 1.0, "step": 0.01}),
                "clip": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "process"
    CATEGORY = "duanyll/image"
    DESCRIPTION = "Apply linear stretch to the input mask to enhance contrast."

    def process(self, mask: torch.Tensor, tol_low: float, tol_high: float, clip: bool):
        """
        Applies linear stretch to the input mask.

        Args:
            mask (torch.Tensor): The input mask tensor (Batch, Height, Width).

        Returns:
            A tuple containing the linstretched mask tensor.
        """
        # Compute global min and max
        min_val = torch.quantile(mask, tol_low)
        max_val = torch.quantile(mask, tol_high)

        # Apply linear stretch
        scale = max_val - min_val
        if scale < 1e-6:
            scale = 1.0  # Prevent division by zero

        linstretched_mask = (mask - min_val) / scale
        if clip:
            linstretched_mask = torch.clamp(linstretched_mask, 0.0, 1.0)

        return (linstretched_mask,)