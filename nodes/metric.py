import os
import cv2
import numpy as np
import torch
import folder_paths

# --- Tensor to CV2 Image Conversion ---
def tensor_to_cv2_img(tensor):
    """Converts a PyTorch tensor (in ComfyUI's format) to a CV2 image (NumPy array)."""
    # Clone the tensor to avoid modifying the original, move to CPU, and convert to NumPy
    image_np = tensor.squeeze(0).cpu().numpy()
    
    # Denormalize from [0, 1] to [0, 255] and change type to uint8
    image_np = (image_np * 255).astype(np.uint8)
    
    # Convert from RGB (ComfyUI) to BGR (cv2)
    # This is not always necessary for InsightFace's get() method, which handles both,
    # but it's good practice for general cv2 compatibility.
    # import cv2
    # image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    return image_np

# --- InsightFace Model Loader ---
# This part ensures the model is loaded only once and shared across nodes.
g_insightface_model = None
INSIGHTFACE_MODEL_ROOT = os.path.join(folder_paths.base_path, "models", "insightface")

def get_insightface_model():
    """Initializes and returns the InsightFace model."""
    global g_insightface_model
    if g_insightface_model is None:
        try:
            import insightface
            from insightface.app import FaceAnalysis
            print("Initializing InsightFace model for Face Similarity node...")
            # Use 'CPUExecutionProvider' to avoid potential CUDA conflicts with ComfyUI
            providers = ['CPUExecutionProvider']
            g_insightface_model = FaceAnalysis(name='buffalo_l', providers=providers, root=INSIGHTFACE_MODEL_ROOT)
            g_insightface_model.prepare(ctx_id=0, det_size=(640, 640))
            print("InsightFace model initialized successfully.")
        except ImportError:
            raise ImportError("Please install 'insightface' and 'onnxruntime' to use this node. Run: pip install insightface onnxruntime")
    return g_insightface_model

# --- ComfyUI Custom Node ---
class InsightFaceSimilarity:
    """
    A ComfyUI node to calculate the similarity score between faces in two images
    using the InsightFace library.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("similarity_score", "details")
    FUNCTION = "calculate_similarity"
    CATEGORY = "duanyll/metric"

    def calculate_similarity(self, image1: torch.Tensor, image2: torch.Tensor):
        try:
            model = get_insightface_model()
            
            # Convert ComfyUI's image tensors to NumPy arrays that insightface can process
            img1_np = tensor_to_cv2_img(image1)
            img2_np = tensor_to_cv2_img(image2)

            # Detect faces
            faces1 = model.get(img1_np)
            faces2 = model.get(img2_np)

            if not faces1 or not faces2:
                details = "No faces detected in one or both images."
                score = 0.0
                print(f"WARNING: {details}")
                return {"ui": {"text": [details]}, "result": (score, details)}

            # --- Core Similarity Logic ---
            # When multiple faces exist, find the pair with the highest similarity
            max_similarity = -1.0
            for face1 in faces1:
                for face2 in faces2:
                    emb1 = face1.embedding
                    emb2 = face2.embedding
                    # Calculate cosine similarity
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    max_similarity = max(max_similarity, similarity)
            
            # Scale the score from [-1, 1] range to a more intuitive [0, 10] range
            # A score of ~0.5 in cosine similarity often represents the same person.
            # Scaling by 10 makes it more readable.
            score = float(10 * max(0, max_similarity))
            details = f"Highest cosine similarity: {max_similarity:.4f}\nFaces found (img1/img2): {len(faces1)}/{len(faces2)}"
            
            # The 'ui' part allows displaying text directly on the node in the UI
            # The 'result' part provides the actual outputs for other nodes to use
            return {"ui": {"text": [f"Score: {score:.2f}"]}, "result": (score, details)}

        except Exception as e:
            error_message = f"Error in Face Similarity node: {str(e)}"
            print(f"\033[91m{error_message}\033[0m") # Print error in red
            return {"ui": {"text": [error_message]}, "result": (0.0, error_message)}
        
        
class LaplacianVariance:
    """
    一个计算图像拉普拉斯算子方差的ComfyUI节点。
    输入：图像 (IMAGE)
    输出：方差值 (FLOAT)，值越高通常表示图像越清晰。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("laplacian_variance",)
    FUNCTION = "calculate_variance"
    CATEGORY = "duanyll/metric"

    def calculate_variance(self, image: torch.Tensor):
        # 将输入的PyTorch Tensor转换为NumPy数组
        # Tensor 格式: [batch_size, height, width, channels], 值范围 [0.0, 1.0]
        # 我们只处理批次中的第一张图片
        image_np = image[0].cpu().numpy()

        # 将图像从 [0, 1] 的浮点数范围转换为 [0, 255] 的8位整数范围
        image_np = (image_np * 255).astype(np.uint8)

        # 将RGB图像转换为灰度图，因为模糊度分析在亮度通道上进行即可
        # ComfyUI 使用 RGB 顺序
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # 计算拉普拉斯算子
        # cv2.CV_64F 是为了避免信息丢失（因为拉普拉斯算子会产生负值）
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

        # 计算拉普拉斯结果的方差
        variance = laplacian.var()

        # 打印结果用于调试
        print(f"Laplacian Variance: {variance}")

        # 返回一个包含浮点数结果的元组
        return (float(variance),)