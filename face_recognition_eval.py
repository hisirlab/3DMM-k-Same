# coding: utf-8
"""
人脸识别评估模块
使用预训练的人脸识别模型（如FaceNet、ArcFace等）评估匿名化效果
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


class FaceRecognitionEvaluator:
    """人脸识别评估器"""
    
    def __init__(self, model_type='facenet', device='cpu'):
        """
        初始化人脸识别模型
        Args:
            model_type: 模型类型，支持 'facenet', 'arcface', 'opencv'
            device: 设备类型
        """
        self.model_type = model_type
        self.device = device
        
        if model_type == 'opencv':
            # 使用OpenCV的DNN模块加载预训练模型
            self._init_opencv_model()
        elif model_type == 'facenet':
            try:
                from facenet_pytorch import InceptionResnetV1
                import torch
                self.model = InceptionResnetV1(pretrained='vggface2').eval()
                if device == 'cpu':
                    self.model = self.model.cpu()
                else:
                    self.model = self.model.cuda()
                print("使用 FaceNet (facenet_pytorch) 模型")
            except ImportError:
                print("警告: facenet_pytorch 未安装，回退到 OpenCV 模型")
                print("如需使用 FaceNet，请运行: pip install facenet-pytorch")
                self._init_opencv_model()
                self.model_type = 'opencv'
        elif model_type == 'deepface':
            try:
                from deepface import DeepFace
                self.deepface = DeepFace
                print("使用 DeepFace 框架")
            except ImportError:
                print("警告: deepface 未安装，回退到 OpenCV 模型")
                print("如需使用 DeepFace，请运行: pip install deepface")
                self._init_opencv_model()
                self.model_type = 'opencv'
        else:
            self._init_opencv_model()
    
    def _init_opencv_model(self):
        """初始化OpenCV人脸识别模型"""
        print("使用 OpenCV DNN 模型")
        # 可以使用OpenCV的人脸识别器
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.model_type = 'opencv'
    
    def extract_face_embedding(self, img_path_or_array):
        """
        提取人脸特征向量
        Args:
            img_path_or_array: 图像路径或numpy数组
        Returns:
            embedding: 特征向量
        """
        if isinstance(img_path_or_array, str):
            img = cv2.imread(img_path_or_array)
        else:
            img = img_path_or_array
        
        if img is None:
            return None
        
        if self.model_type == 'facenet':
            try:
                import torch
                from facenet_pytorch import MTCNN
                from PIL import Image
                
                # 转换为RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                
                # 检测和对齐人脸
                mtcnn = MTCNN(keep_all=False, device=self.device)
                img_cropped = mtcnn(img_pil)
                
                if img_cropped is None:
                    return None
                
                # 提取特征
                with torch.no_grad():
                    if self.device == 'cpu':
                        img_cropped = img_cropped.cpu()
                    else:
                        img_cropped = img_cropped.cuda()
                    embedding = self.model(img_cropped.unsqueeze(0))
                
                return embedding.cpu().numpy().flatten()
            except Exception as e:
                print(f"FaceNet特征提取失败: {e}")
                return None
        
        elif self.model_type == 'deepface':
            try:
                # DeepFace可以自动处理多种模型
                embedding = self.deepface.represent(
                    img_path=img_path_or_array if isinstance(img_path_or_array, str) else img,
                    model_name='Facenet',
                    enforce_detection=False
                )
                return np.array(embedding[0]['embedding'])
            except Exception as e:
                print(f"DeepFace特征提取失败: {e}")
                return None
        
        else:  # opencv
            # 使用简单的方法：将人脸图像调整大小并展平
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(gray, (128, 128))
            return face_resized.flatten().astype(np.float32) / 255.0
    
    def compute_similarity(self, embedding1, embedding2):
        """
        计算两个特征向量的相似度
        Args:
            embedding1: 特征向量1
            embedding2: 特征向量2
        Returns:
            similarity: 相似度 (0-1之间)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # 余弦相似度
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        # 转换到0-1范围
        similarity = (similarity + 1) / 2
        return similarity
    
    def evaluate_recognition_rate(self, original_dir, anonymized_dir, threshold=0.6):
        """
        评估人脸识别成功率
        Args:
            original_dir: 原始图像目录
            anonymized_dir: 匿名化图像目录
            threshold: 识别阈值
        Returns:
            results: 评估结果字典
        """
        original_files = sorted([f for f in os.listdir(original_dir) 
                               if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        print(f"评估 {len(original_files)} 对图像...")
        
        matched_count = 0
        total_count = 0
        similarities = []
        
        for filename in tqdm(original_files, desc="计算人脸相似度"):
            orig_path = os.path.join(original_dir, filename)
            anon_path = os.path.join(anonymized_dir, filename)
            
            if not os.path.exists(anon_path):
                continue
            
            # 提取特征
            emb_orig = self.extract_face_embedding(orig_path)
            emb_anon = self.extract_face_embedding(anon_path)
            
            if emb_orig is None or emb_anon is None:
                continue
            
            # 计算相似度
            similarity = self.compute_similarity(emb_orig, emb_anon)
            similarities.append(similarity)
            
            # 判断是否匹配
            if similarity >= threshold:
                matched_count += 1
            
            total_count += 1
        
        if total_count == 0:
            print("警告：没有有效的图像对进行评估")
            return None
        
        recognition_rate = matched_count / total_count
        avg_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        
        results = {
            'recognition_rate': recognition_rate,
            'matched_count': matched_count,
            'total_count': total_count,
            'avg_similarity': avg_similarity,
            'std_similarity': std_similarity,
            'similarities': similarities,
            'threshold': threshold
        }
        
        return results
    
    def evaluate_from_npz(self, npz_path, original_dir, threshold=0.6):
        """
        从npz文件评估（用于渲染后的图像）
        Args:
            npz_path: 参数文件路径
            original_dir: 原始图像目录
            threshold: 识别阈值
        Returns:
            results: 评估结果字典
        """
        data = np.load(npz_path, allow_pickle=True)
        filenames = data['filenames']
        
        # 假设匿名化图像在同一目录下的anonymized_faces文件夹
        anonymized_dir = os.path.join(os.path.dirname(npz_path), 'anonymized_faces')
        
        return self.evaluate_recognition_rate(original_dir, anonymized_dir, threshold)


def main():
    parser = argparse.ArgumentParser(description='人脸识别评估')
    parser.add_argument('--original', type=str, default='dataset',
                       help='原始图像目录')
    parser.add_argument('--anonymized', type=str, default='results/anonymized/anonymized_faces',
                       help='匿名化图像目录')
    parser.add_argument('--model', type=str, default='opencv',
                       choices=['opencv', 'facenet', 'deepface'],
                       help='人脸识别模型类型')
    parser.add_argument('--threshold', type=float, default=0.6,
                       help='识别阈值')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='设备类型')
    
    args = parser.parse_args()
    
    # 初始化评估器
    evaluator = FaceRecognitionEvaluator(model_type=args.model, device=args.device)
    
    # 评估识别率
    results = evaluator.evaluate_recognition_rate(
        original_dir=args.original,
        anonymized_dir=args.anonymized,
        threshold=args.threshold
    )
    
    if results:
        print("\n" + "="*50)
        print("人脸识别评估结果:")
        print("="*50)
        print(f"识别成功率: {results['recognition_rate']:.2%}")
        print(f"匹配数量: {results['matched_count']}/{results['total_count']}")
        print(f"平均相似度: {results['avg_similarity']:.4f} ± {results['std_similarity']:.4f}")
        print(f"识别阈值: {results['threshold']}")
        print("="*50)
        print(f"\n注意：识别成功率越低，说明匿名化效果越好")


if __name__ == '__main__':
    main()
