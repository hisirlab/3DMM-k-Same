# coding: utf-8
"""
完整评估流程：Shape Anonymization + Face Recognition + Emotion Recognition
整合所有步骤，提供一键运行的完整评估
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime

from shape_anonymization import ShapeAnonymizer
from face_recognition_eval import FaceRecognitionEvaluator
from emotion_recognition_eval import EmotionRecognitionEvaluator


class FullEvaluationPipeline:
    """完整评估流程"""
    
    def __init__(self, config):
        self.config = config
        self.results = {}
    
    def run_anonymization(self):
        """步骤1: 运行形状匿名化"""
        print("\n" + "="*70)
        print("步骤 1/3: 形状匿名化 (Shape Anonymization)")
        print("="*70)
        
        anonymizer = ShapeAnonymizer(
            config_path=self.config['tddfa_config'],
            onnx=self.config['use_onnx']
        )
        
        results = anonymizer.process_dataset(
            dataset_path=self.config['dataset_path'],
            output_dir=self.config['output_dir'],
            k_value=self.config['k_value']
        )
        
        self.results['anonymization'] = {
            'total_images': len(results['filenames']) if results else 0,
            'output_dir': self.config['output_dir']
        }
        
        return results is not None
    
    def run_face_recognition_eval(self):
        """步骤2: 人脸识别评估"""
        print("\n" + "="*70)
        print("步骤 2/3: 人脸识别评估 (Face Recognition Evaluation)")
        print("="*70)
        print("目标: 识别成功率应该越低越好 (说明匿名化效果好)")
        print("-"*70)
        
        evaluator = FaceRecognitionEvaluator(
            model_type=self.config['face_model'],
            device=self.config['device']
        )
        
        anonymized_dir = os.path.join(self.config['output_dir'], 'anonymized_faces')
        
        results = evaluator.evaluate_recognition_rate(
            original_dir=self.config['dataset_path'],
            anonymized_dir=anonymized_dir,
            threshold=self.config['face_threshold']
        )
        
        if results:
            self.results['face_recognition'] = {
                'recognition_rate': results['recognition_rate'],
                'matched_count': results['matched_count'],
                'total_count': results['total_count'],
                'avg_similarity': results['avg_similarity'],
                'std_similarity': results['std_similarity'],
                'threshold': results['threshold']
            }
            
            print(f"\n识别成功率: {results['recognition_rate']:.2%}")
            print(f"匹配数量: {results['matched_count']}/{results['total_count']}")
            print(f"平均相似度: {results['avg_similarity']:.4f} ± {results['std_similarity']:.4f}")
        
        return results is not None
    
    def run_emotion_recognition_eval(self):
        """步骤3: 表情识别评估"""
        print("\n" + "="*70)
        print("步骤 3/3: 表情识别评估 (Emotion Recognition Evaluation)")
        print("="*70)
        print("目标: 表情保持率应该越高越好 (说明表情保留效果好)")
        print("-"*70)
        
        evaluator = EmotionRecognitionEvaluator(
            model_type=self.config['emotion_model'],
            device=self.config['device']
        )
        
        anonymized_dir = os.path.join(self.config['output_dir'], 'anonymized_faces')
        
        results = evaluator.evaluate_emotion_preservation(
            original_dir=self.config['dataset_path'],
            anonymized_dir=anonymized_dir
        )
        
        if results:
            self.results['emotion_recognition'] = {
                'preservation_rate': results['preservation_rate'],
                'same_emotion_count': results['same_emotion_count'],
                'total_count': results['total_count'],
                'avg_similarity': results['avg_similarity'],
                'std_similarity': results['std_similarity']
            }
            
            print(f"\n表情保持率: {results['preservation_rate']:.2%}")
            print(f"相同表情数量: {results['same_emotion_count']}/{results['total_count']}")
            print(f"平均相似度: {results['avg_similarity']:.4f} ± {results['std_similarity']:.4f}")
            
            # 显示一些示例
            if len(results['emotion_pairs']) > 0:
                print("\n前5个表情对比示例:")
                for i, (orig, anon) in enumerate(results['emotion_pairs'][:5]):
                    match = "✓" if orig == anon else "✗"
                    print(f"  {i+1}. 原始: {orig:10s} -> 匿名化: {anon:10s} {match}")
        
        return results is not None
    
    def save_results(self):
        """保存评估结果"""
        results_file = os.path.join(self.config['output_dir'], 'evaluation_results.json')
        
        # 添加元数据
        self.results['metadata'] = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_path': self.config['dataset_path'],
            'k_value': self.config['k_value'],
            'face_model': self.config['face_model'],
            'emotion_model': self.config['emotion_model'],
            'face_threshold': self.config['face_threshold']
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=4, ensure_ascii=False)
        
        print(f"\n评估结果已保存到: {results_file}")
    
    def print_summary(self):
        """打印总结报告"""
        print("\n" + "="*70)
        print("评估总结报告")
        print("="*70)
        
        if 'anonymization' in self.results:
            print(f"\n1. 匿名化处理:")
            print(f"   - 处理图像数量: {self.results['anonymization']['total_images']}")
            print(f"   - 输出目录: {self.results['anonymization']['output_dir']}")
        
        if 'face_recognition' in self.results:
            fr = self.results['face_recognition']
            print(f"\n2. 人脸识别评估:")
            print(f"   - 识别成功率: {fr['recognition_rate']:.2%} (越低越好)")
            print(f"   - 匹配数量: {fr['matched_count']}/{fr['total_count']}")
            print(f"   - 平均相似度: {fr['avg_similarity']:.4f}")
            
            # 评分
            if fr['recognition_rate'] < 0.3:
                score = "优秀 ✓✓✓"
            elif fr['recognition_rate'] < 0.5:
                score = "良好 ✓✓"
            elif fr['recognition_rate'] < 0.7:
                score = "一般 ✓"
            else:
                score = "较差 ✗"
            print(f"   - 匿名化效果: {score}")
        
        if 'emotion_recognition' in self.results:
            er = self.results['emotion_recognition']
            print(f"\n3. 表情识别评估:")
            print(f"   - 表情保持率: {er['preservation_rate']:.2%} (越高越好)")
            print(f"   - 相同表情数量: {er['same_emotion_count']}/{er['total_count']}")
            print(f"   - 平均相似度: {er['avg_similarity']:.4f}")
            
            # 评分
            if er['preservation_rate'] > 0.7:
                score = "优秀 ✓✓✓"
            elif er['preservation_rate'] > 0.5:
                score = "良好 ✓✓"
            elif er['preservation_rate'] > 0.3:
                score = "一般 ✓"
            else:
                score = "较差 ✗"
            print(f"   - 表情保留效果: {score}")
        
        print("\n" + "="*70)
        print("结论:")
        print("- 理想情况：人脸识别成功率低 + 表情保持率高")
        print("- 这说明成功实现了身份匿名化，同时保留了表情信息")
        print("="*70)
    
    def run_full_pipeline(self):
        """运行完整评估流程"""
        start_time = time.time()
        
        print("\n" + "="*70)
        print("开始完整评估流程")
        print("="*70)
        print(f"数据集: {self.config['dataset_path']}")
        print(f"输出目录: {self.config['output_dir']}")
        print(f"K值: {self.config['k_value'] if self.config['k_value'] else '全部'}")
        print("="*70)
        
        # 步骤1: 匿名化
        success = self.run_anonymization()
        if not success:
            print("错误: 匿名化失败")
            return False
        
        # 步骤2: 人脸识别评估
        success = self.run_face_recognition_eval()
        if not success:
            print("警告: 人脸识别评估失败")
        
        # 步骤3: 表情识别评估
        success = self.run_emotion_recognition_eval()
        if not success:
            print("警告: 表情识别评估失败")
        
        # 保存结果
        self.save_results()
        
        # 打印总结
        self.print_summary()
        
        elapsed_time = time.time() - start_time
        print(f"\n总耗时: {elapsed_time:.2f} 秒")
        
        return True


def main():
    parser = argparse.ArgumentParser(description='完整的形状匿名化评估流程')
    
    # 基本参数
    parser.add_argument('--dataset', type=str, default='dataset',
                       help='数据集路径')
    parser.add_argument('--output', type=str, default='results/full_evaluation',
                       help='输出目录')
    parser.add_argument('--config', type=str, default='configs/mb1_120x120.yml',
                       help='3DDFA配置文件')
    
    # 匿名化参数
    parser.add_argument('--k', type=int, default=None,
                       help='k-same算法的k值，None表示使用所有图像')
    parser.add_argument('--onnx', action='store_true', default=False,
                       help='使用ONNX模式')
    
    # 评估参数
    parser.add_argument('--face-model', type=str, default='opencv',
                       choices=['opencv', 'facenet', 'deepface'],
                       help='人脸识别模型')
    parser.add_argument('--emotion-model', type=str, default='fer',
                       choices=['opencv', 'fer', 'deepface'],
                       help='表情识别模型')
    parser.add_argument('--face-threshold', type=float, default=0.6,
                       help='人脸识别阈值')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='设备类型')
    
    # 可选参数
    parser.add_argument('--skip-anonymization', action='store_true',
                       help='跳过匿名化步骤（如果已经运行过）')
    
    args = parser.parse_args()
    
    # 构建配置
    config = {
        'dataset_path': args.dataset,
        'output_dir': args.output,
        'tddfa_config': args.config,
        'k_value': args.k,
        'use_onnx': args.onnx,
        'face_model': args.face_model,
        'emotion_model': args.emotion_model,
        'face_threshold': args.face_threshold,
        'device': args.device,
        'skip_anonymization': args.skip_anonymization
    }
    
    # 运行评估流程
    pipeline = FullEvaluationPipeline(config)
    
    if args.skip_anonymization:
        print("跳过匿名化步骤，直接进行评估...")
        pipeline.run_face_recognition_eval()
        pipeline.run_emotion_recognition_eval()
        pipeline.save_results()
        pipeline.print_summary()
    else:
        pipeline.run_full_pipeline()


if __name__ == '__main__':
    main()
