# coding: utf-8
"""
自适应贴脸口罩基线：
- 使用 FaceBoxes 检测人脸框
- 基于人脸框生成近似椭圆形脸部区域
- 对脸部下半部分进行遮挡（可调起始位置）
不影响原版 baseline，实现可回退。
"""
import os
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from FaceBoxes import FaceBoxes


def mask_lower_face(img, box, start_ratio=0.55, color=(40, 40, 40)):
    """基于检测框创建贴脸遮挡：下半脸椭圆遮挡。
    Args:
        img: BGR图像
        box: [x1, y1, x2, y2]
        start_ratio: 从脸高的百分比开始遮挡 (0-1)
        color: 遮挡颜色
    Returns:
        masked_img
    """
    # FaceBoxes 返回 [x1, y1, x2, y2, score]
    if isinstance(box, (list, tuple, np.ndarray)):
        if len(box) >= 4:
            x1, y1, x2, y2 = map(int, box[:4])
        else:
            raise ValueError(f"Invalid box format: {box}")
    else:
        raise ValueError(f"Invalid box type: {type(box)}")
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w // 2, y1 + h // 2

    # 构建椭圆近似脸型掩膜
    axes = (max(1, int(w * 0.5)), max(1, int(h * 0.65)))  # x轴半径、y轴半径
    face_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.ellipse(face_mask, (cx, cy), axes, 0, 0, 360, 255, thickness=-1)

    # 生成下半脸区域（从 start_ratio 开始向下）
    y_start = int(y1 + h * start_ratio)
    y_start = max(y1, min(y_start, y2))
    lower_mask = np.zeros_like(face_mask)
    cv2.rectangle(lower_mask, (x1, y_start), (x2, y2), 255, thickness=-1)

    # 椭圆与下半部分的交集作为遮挡区域
    mask = cv2.bitwise_and(face_mask, lower_mask)

    # 平滑边缘，贴合更自然
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    mask_3 = cv2.merge([mask, mask, mask])

    # 应用遮挡颜色
    overlay = np.full_like(img, color, dtype=np.uint8)
    masked = np.where(mask_3 > 0, overlay, img)
    return masked


def run_adaptive_baseline(dataset_path, output_dir, start_ratio=0.55, color=(40, 40, 40)):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(dataset_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    detector = FaceBoxes()

    start = time.time()
    processed = 0
    for name in tqdm(files, desc="Adaptive口罩遮挡"):
        p = os.path.join(dataset_path, name)
        img = cv2.imread(p)
        if img is None:
            continue
        boxes = detector(img)
        if len(boxes) == 0:
            # 无人脸，直接复制
            cv2.imwrite(os.path.join(output_dir, name), img)
            processed += 1
            continue
        # 选择最大框
        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
        box = boxes[np.argmax(areas)]
        masked = mask_lower_face(img, box, start_ratio=start_ratio, color=color)
        cv2.imwrite(os.path.join(output_dir, name), masked)
        processed += 1

    elapsed = time.time() - start
    return {
        "processed_files": processed,
        "elapsed": elapsed,
        "avg_time": elapsed / processed if processed else 0.0,
        "output_dir": output_dir,
        "start_ratio": start_ratio
    }


def main():
    parser = argparse.ArgumentParser(description="自适应贴脸口罩基线")
    parser.add_argument("--dataset", type=str, default="dataset/Celeb-A", help="数据集路径")
    parser.add_argument("--output", type=str, default="results/baseline_mask_adaptive", help="输出目录")
    parser.add_argument("--start-ratio", type=float, default=0.55, help="遮挡起始位置比例(0-1)")
    args = parser.parse_args()

    stats = run_adaptive_baseline(args.dataset, args.output, start_ratio=args.start_ratio)
    print(f"完成: {stats['processed_files']} 张, 平均 {stats['avg_time']:.3f}s/张, 输出: {stats['output_dir']}")


if __name__ == "__main__":
    main()
