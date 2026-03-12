# coding: utf-8
"""
改进版的形状匿名化
提供多种策略来提升匿名性效果
"""

import os
import sys
import cv2
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render
from utils.tddfa_util import _parse_param


class ImprovedShapeAnonymizer:
    """改进的形状匿名化器 - 提供多种匿名化策略"""
    
    def __init__(self, config_path='configs/mb1_120x120.yml', onnx=False):
        cfg = yaml.load(open(config_path), Loader=yaml.SafeLoader)
        
        if onnx:
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            os.environ['OMP_NUM_THREADS'] = '4'
            from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
            from TDDFA_ONNX import TDDFA_ONNX
            self.face_boxes = FaceBoxes_ONNX()
            self.tddfa = TDDFA_ONNX(**cfg)
        else:
            self.tddfa = TDDFA(gpu_mode=False, **cfg)
            self.face_boxes = FaceBoxes()
        
        self.shape_dim = 40
        self.exp_dim = 10
        self.trans_dim = 12
        
    def extract_params(self, img_path):
        """提取3DMM参数"""
        img = cv2.imread(img_path)
        if img is None:
            return None, None, None
            
        boxes = self.face_boxes(img)
        if len(boxes) == 0:
            return None, None, None
        
        param_lst, roi_box_lst = self.tddfa(img, boxes)
        return param_lst[0], roi_box_lst[0], img
    
    def parse_param(self, param):
        """解析参数"""
        pose = param[:self.trans_dim]
        shape = param[self.trans_dim:self.trans_dim + self.shape_dim]
        exp = param[self.trans_dim + self.shape_dim:]
        return pose, shape, exp
    
    def reconstruct_param(self, pose, shape, exp):
        """重建参数"""
        return np.concatenate([pose, shape, exp])
    
    def compute_average_shape(self, shape_list):
        """计算平均形状"""
        return np.mean(shape_list, axis=0)
    
    def add_noise_to_shape(self, shape, noise_level=0.1):
        """
        为形状参数添加噪声以增强匿名性
        Args:
            shape: 形状参数
            noise_level: 噪声水平 (0-1)
        """
        noise = np.random.normal(0, noise_level, shape.shape)
        return shape + noise
    
    def normalize_pose(self, pose, target_pose=None):
        """
        标准化姿态参数以增强匿名性
        Args:
            pose: 原始姿态参数
            target_pose: 目标姿态（None则使用平均姿态）
        """
        if target_pose is None:
            # 使用中性姿态
            target_pose = np.zeros_like(pose)
            # 保留缩放信息
            target_pose[:3] = pose[:3] * 0.5  # 部分保留旋转
        return target_pose
    
    def blend_shapes(self, shape1, shape2, alpha=0.5):
        """
        混合两个形状参数
        Args:
            shape1, shape2: 两个形状参数
            alpha: 混合比例 (0-1)
        """
        return alpha * shape1 + (1 - alpha) * shape2
    
    def process_dataset_improved(self, dataset_path, output_dir, 
                                 strategy='average',
                                 noise_level=0.0,
                                 normalize_pose_flag=False,
                                 blend_alpha=1.0,
                                 exp_blend_alpha=1.0,
                                 component_noise_level=0.0,
                                 render_alpha=0.65,
                                 k_value=None):
        """
        改进的数据集处理
        Args:
            dataset_path: 数据集路径
            output_dir: 输出目录
            strategy: 匿名化策略
                - 'average': 使用平均形状（默认）
                - 'noise': 平均形状+噪声
                - 'blend': 混合形状
            noise_level: 噪声水平 (0-1)
            normalize_pose_flag: 是否标准化姿态
            blend_alpha: 混合比例（用于blend策略）
            exp_blend_alpha: 表情保留权重，1.0表示完全保留原始表情
            component_noise_level: 组件级噪声（眼/鼻/口等局部额外扰动）
            k_value: k-same的k值
        """
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'anonymized_faces'), exist_ok=True)
        
        image_files = sorted([f for f in os.listdir(dataset_path) 
                            if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        if k_value:
            image_files = image_files[:k_value]
        
        print(f"处理 {len(image_files)} 张图像...")
        print(f"匿名化策略: {strategy}")
        print(f"噪声水平: {noise_level}")
        print(f"组件噪声: {component_noise_level}")
        print(f"姿态标准化: {normalize_pose_flag}")
        print(f"表情保留权重: {exp_blend_alpha}")
        print(f"渲染融合alpha: {render_alpha}")
        
        # 第一步：提取所有参数
        all_params = []
        all_roi_boxes = []
        all_images = []
        valid_files = []
        
        for img_file in tqdm(image_files, desc="提取参数"):
            img_path = os.path.join(dataset_path, img_file)
            param, roi_box, img = self.extract_params(img_path)
            
            if param is not None:
                all_params.append(param)
                all_roi_boxes.append(roi_box)
                all_images.append(img)
                valid_files.append(img_file)
        
        print(f"成功提取 {len(all_params)} 张人脸参数")
        
        if len(all_params) == 0:
            return None
        
        # 第二步：计算平均形状和姿态
        all_shapes = []
        all_poses = []
        all_exps = []
        for param in all_params:
            pose, shape, exp = self.parse_param(param)
            all_shapes.append(shape)
            all_poses.append(pose)
            all_exps.append(exp)
        
        avg_shape = self.compute_average_shape(all_shapes)
        avg_pose = np.mean(all_poses, axis=0) if normalize_pose_flag else None
        avg_exp = np.mean(all_exps, axis=0)
        
        print(f"计算得到平均形状参数")
        if normalize_pose_flag:
            print(f"计算得到平均姿态参数")
        
        # 第三步：根据策略进行匿名化
        results = {
            'original_params': [],
            'anonymized_params': [],
            'filenames': [],
            'avg_shape': avg_shape,
            'strategy': strategy,
            'noise_level': noise_level
        }
        
        for idx, (param, roi_box, img, filename) in enumerate(
            tqdm(zip(all_params, all_roi_boxes, all_images, valid_files), 
                 total=len(all_params), desc="生成匿名化人脸")):
            
            pose, shape, exp = self.parse_param(param)
            
            # 根据策略选择匿名化方法
            if strategy == 'average':
                # 策略1：纯平均形状
                anon_shape = avg_shape
                
            elif strategy == 'noise':
                # 策略2：平均形状 + 噪声
                anon_shape = self.add_noise_to_shape(avg_shape, noise_level)
                
            elif strategy == 'blend':
                # 策略3：与平均形状混合
                anon_shape = self.blend_shapes(shape, avg_shape, blend_alpha)
            
            else:
                anon_shape = avg_shape

            if component_noise_level > 0:
                # 对关键组件施加额外噪声，提升去识别性
                comp_noise = np.zeros_like(anon_shape)
                # 简单分段：0-9眼睛，10-19鼻子，20-29嘴，30-39下颌
                for start, end in [(0, 10), (10, 20), (20, 30), (30, 40)]:
                    comp_noise[start:end] = np.random.normal(0, component_noise_level, end - start)
                anon_shape = anon_shape + comp_noise
            
            # 是否标准化姿态
            if normalize_pose_flag and avg_pose is not None:
                anon_pose = self.normalize_pose(pose, avg_pose)
            else:
                anon_pose = pose

            # 表情保留/微调：exp_blend_alpha=1.0完全保留原始表情，<1时向平均表情平滑
            anon_exp = exp_blend_alpha * exp + (1 - exp_blend_alpha) * avg_exp

            # 重建参数
            anonymized_param = self.reconstruct_param(anon_pose, anon_shape, anon_exp)

            # 渲染
            ver_anon = self.tddfa.recon_vers([anonymized_param], [roi_box], dense_flag=True)
            img_anon = img.copy()
            # 使用更强融合以避免“原脸+匿名脸”双重叠加
            render(img_anon, ver_anon, self.tddfa.tri, alpha=render_alpha, show_flag=False,
                   wfp=os.path.join(output_dir, 'anonymized_faces', filename))
            
            results['original_params'].append(param)
            results['anonymized_params'].append(anonymized_param)
            results['filenames'].append(filename)
        
        # 保存结果
        np.savez(os.path.join(output_dir, 'params.npz'),
                 original_params=np.array(results['original_params']),
                 anonymized_params=np.array(results['anonymized_params']),
                 avg_shape=avg_shape,
                 filenames=results['filenames'],
                 strategy=strategy,
                 noise_level=noise_level,
                 component_noise_level=component_noise_level,
                 exp_blend_alpha=exp_blend_alpha,
                 render_alpha=render_alpha)
        
        print(f"匿名化完成！结果保存在: {output_dir}")
        return results


def main():
    parser = argparse.ArgumentParser(description='改进的形状匿名化')
    parser.add_argument('--config', type=str, default='configs/mb1_120x120.yml')
    parser.add_argument('--dataset', type=str, default='dataset')
    parser.add_argument('--output', type=str, default='results/anonymized_improved')
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--onnx', action='store_true', default=False)
    
    # 匿名化策略参数
    parser.add_argument('--strategy', type=str, default='average',
                       choices=['average', 'noise', 'blend'],
                       help='匿名化策略: average/noise/blend')
    parser.add_argument('--noise-level', type=float, default=0.0,
                       help='噪声水平 (0-1)')
    parser.add_argument('--normalize-pose', action='store_true',
                       help='是否标准化姿态')
    parser.add_argument('--blend-alpha', type=float, default=1.0,
                       help='混合比例 (0-1, 用于blend策略)')
    parser.add_argument('--exp-blend-alpha', type=float, default=1.0,
                       help='表情保留权重 (1.0表示完全保留原表情)')
    parser.add_argument('--component-noise-level', type=float, default=0.0,
                       help='组件级噪声水平 (眼/鼻/口局部增强扰动)')
    parser.add_argument('--render-alpha', type=float, default=0.65,
                       help='渲染融合alpha (0.6-0.7较自然, 1.0完全覆盖)')
    
    args = parser.parse_args()
    
    anonymizer = ImprovedShapeAnonymizer(config_path=args.config, onnx=args.onnx)
    
    results = anonymizer.process_dataset_improved(
        dataset_path=args.dataset,
        output_dir=args.output,
        strategy=args.strategy,
        noise_level=args.noise_level,
        normalize_pose_flag=args.normalize_pose,
        blend_alpha=args.blend_alpha,
        exp_blend_alpha=args.exp_blend_alpha,
        component_noise_level=args.component_noise_level,
        render_alpha=args.render_alpha,
        k_value=args.k
    )
    
    if results:
        print(f"\n处理完成:")
        print(f"- 策略: {args.strategy}")
        print(f"- 图像数量: {len(results['filenames'])}")


if __name__ == '__main__':
    main()
