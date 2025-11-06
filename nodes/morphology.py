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
    CATEGORY = "duanyll/morphology"

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
