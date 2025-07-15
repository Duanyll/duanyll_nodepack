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
    CATEGORY = "duanyll"
    
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