# coding: utf-8
"""
Shape Anonymization with 3DDFA_V2
提取形状和表情参数，使用k-same算法替换形状参数，重建匿名化人脸
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


class ShapeAnonymizer:
    """使用k-same算法进行形状匿名化"""
    
    def __init__(self, config_path='configs/mb1_120x120.yml', onnx=False):
        """
        初始化3DDFA模型
        Args:
            config_path: 配置文件路径
            onnx: 是否使用ONNX模式
        """
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
        
        self.shape_dim = 40  # 形状参数维度
        self.exp_dim = 10    # 表情参数维度
        self.trans_dim = 12  # 姿态参数维度
        
    def extract_params(self, img_path):
        """
        从图像中提取3DMM参数
        Args:
            img_path: 图像路径
        Returns:
            param: 完整的3DMM参数 (62维)
            roi_box: 人脸区域框
            img: 原始图像
        """
        img = cv2.imread(img_path)
        if img is None:
            return None, None, None
            
        boxes = self.face_boxes(img)
        if len(boxes) == 0:
            print(f"未检测到人脸: {img_path}")
            return None, None, None
        
        # 只处理第一张人脸
        param_lst, roi_box_lst = self.tddfa(img, boxes)
        return param_lst[0], roi_box_lst[0], img
    
    def parse_param(self, param):
        """
        解析3DMM参数
        Args:
            param: 62维参数向量
        Returns:
            pose: 姿态参数 (12维)
            shape: 形状参数 (40维)
            exp: 表情参数 (10维)
        """
        pose = param[:self.trans_dim]
        shape = param[self.trans_dim:self.trans_dim + self.shape_dim]
        exp = param[self.trans_dim + self.shape_dim:]
        return pose, shape, exp
    
    def reconstruct_param(self, pose, shape, exp):
        """
        重建3DMM参数
        Args:
            pose: 姿态参数 (12维)
            shape: 形状参数 (40维)
            exp: 表情参数 (10维)
        Returns:
            param: 完整的62维参数向量
        """
        return np.concatenate([pose, shape, exp])
    
    def compute_average_shape(self, shape_list):
        """
        计算平均形状参数 (k-same算法的核心)
        Args:
            shape_list: 形状参数列表
        Returns:
            avg_shape: 平均形状参数
        """
        return np.mean(shape_list, axis=0)
    
    def process_dataset(self, dataset_path, output_dir, k_value=None):
        """
        处理整个数据集，应用k-same匿名化
        Args:
            dataset_path: 数据集路径
            output_dir: 输出目录
            k_value: k-same中的k值，None表示使用所有图像
        Returns:
            results: 包含原始和匿名化参数的字典
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'anonymized_faces'), exist_ok=True)
        
        # 获取所有图像文件
        image_files = sorted([f for f in os.listdir(dataset_path) 
                            if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        if k_value:
            image_files = image_files[:k_value]
        
        print(f"处理 {len(image_files)} 张图像...")
        
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
            print("没有成功提取的人脸参数，退出")
            return None
        
        # 第二步：计算平均形状
        all_shapes = []
        for param in all_params:
            _, shape, _ = self.parse_param(param)
            all_shapes.append(shape)
        
        avg_shape = self.compute_average_shape(all_shapes)
        print(f"计算得到平均形状参数")
        
        # 第三步：替换形状参数并重建人脸
        results = {
            'original_params': [],
            'anonymized_params': [],
            'original_images': [],
            'anonymized_images': [],
            'filenames': [],
            'avg_shape': avg_shape
        }
        
        for idx, (param, roi_box, img, filename) in enumerate(
            tqdm(zip(all_params, all_roi_boxes, all_images, valid_files), 
                 total=len(all_params), desc="生成匿名化人脸")):
            
            # 解析参数
            pose, shape, exp = self.parse_param(param)
            
            # 使用平均形状替换原始形状
            anonymized_param = self.reconstruct_param(pose, avg_shape, exp)
            
            # 重建人脸可视化
            ver_orig = self.tddfa.recon_vers([param], [roi_box], dense_flag=True)
            ver_anon = self.tddfa.recon_vers([anonymized_param], [roi_box], dense_flag=True)
            
            # 渲染匿名化人脸
            img_anon = img.copy()
            render(img_anon, ver_anon, self.tddfa.tri, alpha=0.6, show_flag=False, 
                   wfp=os.path.join(output_dir, 'anonymized_faces', filename))
            
            # 保存结果
            results['original_params'].append(param)
            results['anonymized_params'].append(anonymized_param)
            results['original_images'].append(img)
            results['anonymized_images'].append(img_anon)
            results['filenames'].append(filename)
        
        # 保存参数到文件
        np.savez(os.path.join(output_dir, 'params.npz'),
                 original_params=np.array(results['original_params']),
                 anonymized_params=np.array(results['anonymized_params']),
                 avg_shape=avg_shape,
                 filenames=results['filenames'])
        
        print(f"匿名化完成！结果保存在: {output_dir}")
        return results


def main():
    parser = argparse.ArgumentParser(description='3DDFA Shape Anonymization')
    parser.add_argument('--config', type=str, default='configs/mb1_120x120.yml',
                       help='配置文件路径')
    parser.add_argument('--dataset', type=str, default='dataset',
                       help='数据集路径')
    parser.add_argument('--output', type=str, default='results/anonymized',
                       help='输出目录')
    parser.add_argument('--k', type=int, default=None,
                       help='k-same中的k值，None表示使用所有图像')
    parser.add_argument('--onnx', action='store_true', default=False,
                       help='使用ONNX模式')
    
    args = parser.parse_args()
    
    # 初始化匿名化器
    anonymizer = ShapeAnonymizer(config_path=args.config, onnx=args.onnx)
    
    # 处理数据集
    results = anonymizer.process_dataset(
        dataset_path=args.dataset,
        output_dir=args.output,
        k_value=args.k
    )
    
    if results:
        print(f"\n处理完成:")
        print(f"- 原始图像数量: {len(results['filenames'])}")
        print(f"- 匿名化图像保存在: {args.output}/anonymized_faces")
        print(f"- 参数保存在: {args.output}/params.npz")


if __name__ == '__main__':
    main()
