import torch
import numpy as np
import cv2
import kornia

class CoverWordsWithRectangles:
    """
    一个 ComfyUI 节点，用于读取 MASK，并为每个单词绘制白色旋转矩形以完全覆盖。
    这对于从图像中移除或替换文本很有用。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "dilation_width_ratio": ("FLOAT", {
                    "default": 0.02, 
                    "min": 0.0, 
                    "max": 0.2, 
                    "step": 0.001,
                    "display": "number"
                }),
                "min_area": ("INT", {
                    "default": 100, 
                    "min": 0, 
                    "max": 10000, 
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "cover_words"
    CATEGORY = "duanyll"

    def tensor_to_cv2_img(self, tensor):
        """将单张 MASK 张量转换为 OpenCV 图像 (np.ndarray)"""
        # 将张量从 [H, W] 转换为 [H, W] numpy 数组
        img_np = tensor.cpu().numpy()
        # 将 0-1 范围的浮点数转换为 0-255 范围的 uint8
        img_np = (img_np * 255).astype(np.uint8)
        return img_np

    def cv2_img_to_tensor(self, img_np):
        """将 OpenCV 图像 (np.ndarray) 转换为 MASK 张量"""
        # 将 0-255 范围的 uint8 转换回 0-1 范围的浮点数
        img_np = img_np.astype(np.float32) / 255.0
        # 将 numpy 数组转换为 [H, W] 的 PyTorch 张量
        return torch.from_numpy(img_np)

    def cover_words(self, mask: torch.Tensor, dilation_width_ratio: float, min_area: int):
        """
        处理输入的 MASK 张量。

        Parameters
        ----------
        mask : torch.Tensor
            输入的 MASK 张量，形状为 (B, H, W)。
        dilation_width_ratio : float
            闭运算核宽与图像宽度之比。
        min_area : int
            忽略面积小于该值的轮廓。

        Returns
        -------
        (torch.Tensor,)
            处理后的 MASK 张量。
        """
        if mask.dim() == 2:
            # 如果输入是二维张量 [H, W]，则增加一个批次维度
            mask = mask.unsqueeze(0)
        
        batch_size, height, width = mask.shape
        result_tensors = []

        # 按批次处理每一张 MASK
        for i in range(batch_size):
            # --- 1. 将 Tensor 转换为 OpenCV 格式 ---
            bw = self.tensor_to_cv2_img(mask[i])

            # --- 2. 闭运算以合并字符 ---
            # 核宽度至少为 1 像素
            kernel_width = max(1, int(width * dilation_width_ratio))
            # 使用一个矩形核，高度较小以避免垂直方向的单词粘连
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 3))
            closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

            # --- 3. 查找轮廓 ---
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # --- 4. 遍历轮廓，在新的画布上绘制最小外接矩形 ---
            # 创建一个与输入大小相同的黑色画布
            canvas = np.zeros((height, width), dtype=np.uint8)
            for cnt in contours:
                # 过滤掉面积过小的噪声轮廓
                if cv2.contourArea(cnt) < min_area:
                    continue
                
                # 计算最小面积的旋转矩形
                rect = cv2.minAreaRect(cnt)  # (center, (w,h), angle)
                box = cv2.boxPoints(rect)     # 获取矩形的四个顶点
                box = np.intp(box)            # 将顶点坐标转换为整数
                
                # 在画布上用白色填充这个多边形（矩形）
                cv2.fillPoly(canvas, [box], 255)

            # --- 5. 将处理后的图像转回 Tensor ---
            result_tensors.append(self.cv2_img_to_tensor(canvas))

        # 将处理后的张量列表堆叠成一个批次
        output_mask = torch.stack(result_tensors, dim=0)
        
        return (output_mask,)
    

class AdvancedMorphology:
    """
    A ComfyUI node to perform large-radius, smooth erosion or dilation on masks
    using a memory-efficient Gaussian blur + binarization technique. This avoids
    the out-of-memory errors that occur with large, traditional morphology kernels.
    """
    
    # This decorator tells ComfyUI about the inputs this node needs
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "method": (["dilate", "erode"], {"default": "dilate"}),
                "sigma": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 200.0, "step": 0.1, "display": "slider"}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01, "display": "slider"}),
                "auto_kernel_size": ("BOOLEAN", {"default": True}),
                # This kernel_size is only used if auto_kernel_size is False
                "kernel_size": ("INT", {"default": 49, "min": 3, "max": 401, "step": 2}), 
            },
        }

    # Define the return types and names
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)

    # The name of the function that will be executed
    FUNCTION = "process"

    # The category for the node in the ComfyUI menu
    CATEGORY = "duanyll"

    def process(self, mask: torch.Tensor, method: str, sigma: float, threshold: float, auto_kernel_size: bool, kernel_size: int):
        
        if mask is None or mask.numel() == 0:
            # Return an empty mask if the input is empty
            return (torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu"),)

        # Kornia expects a 4D tensor in (B, C, H, W) format.
        # ComfyUI masks are (B, H, W), so we add a channel dimension.
        input_tensor = mask.unsqueeze(1)
        device = input_tensor.device
        
        # --- Parameter Logic ---
        if auto_kernel_size:
            # Rule of thumb: kernel size should be about 6 times sigma
            k_size = int(sigma * 6) + 1
            if k_size % 2 == 0:
                k_size += 1 # Ensure kernel size is odd
        else:
            k_size = kernel_size
            if k_size % 2 == 0:
                # Ensure user-provided kernel size is odd, otherwise Kornia will error
                k_size += 1
                print(f"Warning: Kernel size must be odd. Adjusting to {k_size}.")

        print(f"Advanced Morphology: method={method}, sigma={sigma}, threshold={threshold}, kernel_size={k_size}")

        # --- Core Algorithm ---
        tensor_to_blur = None
        if method == 'dilate':
            # To dilate (expand white areas), we blur the original mask
            tensor_to_blur = input_tensor
        else: # erode
            # To erode (shrink white areas), we dilate the background.
            # So, we invert the mask, blur it, and then invert it back.
            tensor_to_blur = 1.0 - input_tensor
            
        # Apply the highly optimized Gaussian blur
        blurred_tensor = kornia.filters.gaussian_blur2d(
            tensor_to_blur, 
            (k_size, k_size), 
            (sigma, sigma),
            border_type='replicate'
        )
        
        # Binarize the result based on the threshold
        # For dilation, we take areas > threshold
        # For erosion, the blurred tensor was the inverted mask, so we also take > threshold
        # which corresponds to the expanded background.
        binarized_result = (blurred_tensor > threshold).to(input_tensor.dtype)
        
        # If we were eroding, we need to invert the result back
        if method == 'erode':
            final_result = 1.0 - binarized_result
        else:
            final_result = binarized_result
            
        # Kornia returns (B, C, H, W), ComfyUI needs (B, H, W) for MASK output
        output_mask = final_result.squeeze(1)

        # The function must return a tuple
        return (output_mask,)