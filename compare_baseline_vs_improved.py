# coding: utf-8
"""
对比基线(下半脸口罩遮挡)与改进匿名化策略的统一评测脚本。
- 基线：模拟口罩遮挡下半张脸
- 改进：使用 ImprovedShapeAnonymizer，增强去识别、保留表情
- 评测：人脸识别、表情保持、PSNR、SSIM、Precision/Recall/F1、处理速度
- 数据集：从 dataset 目录下选择子目录（默认 Celeb-A/LFW/FFHQ 之一）
"""

import os
import time
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from shape_anonymization_improved import ImprovedShapeAnonymizer
from baseline_mask_face_adaptive import run_adaptive_baseline
from face_recognition_eval import FaceRecognitionEvaluator
from emotion_recognition_eval import EmotionRecognitionEvaluator


def apply_mask_baseline(dataset_path, output_dir, mask_ratio=0.55, mask_color=(40, 40, 40)):
    """基线：用矩形遮挡下半张脸，模拟口罩。
    Args:
        dataset_path: 原始数据集目录
        output_dir: 输出目录
        mask_ratio: 遮挡起始位置，高度比例(0-1)，默认从 55% 高度向下遮挡
        mask_color: 遮挡颜色(BGR)
    Returns:
        dict: {processed_files, elapsed, avg_time}
    """
    os.makedirs(output_dir, exist_ok=True)
    files = sorted([f for f in os.listdir(dataset_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    start = time.time()
    processed = 0

    for name in tqdm(files, desc="Baseline遮挡"):
        img_path = os.path.join(dataset_path, name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        y0 = int(h * mask_ratio)
        masked = img.copy()
        cv2.rectangle(masked, (0, y0), (w, h), mask_color, thickness=-1)
        cv2.imwrite(os.path.join(output_dir, name), masked)
        processed += 1

    elapsed = time.time() - start
    return {
        "processed_files": processed,
        "elapsed": elapsed,
        "avg_time": elapsed / processed if processed else 0.0,
        "output_dir": output_dir
    }


def compute_image_metrics(original_dir, processed_dir):
    """计算 PSNR 和 SSIM 均值。"""
    files = sorted([f for f in os.listdir(original_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    psnrs, ssims = [], []
    for name in files:
        p_orig = os.path.join(original_dir, name)
        p_proc = os.path.join(processed_dir, name)
        if not os.path.exists(p_proc):
            continue
        orig = cv2.imread(p_orig)
        proc = cv2.imread(p_proc)
        if orig is None or proc is None:
            continue
        # 调整大小对齐
        if orig.shape != proc.shape:
            proc = cv2.resize(proc, (orig.shape[1], orig.shape[0]))
        psnr_val = peak_signal_noise_ratio(orig, proc, data_range=255)
        ssim_val = structural_similarity(orig, proc, channel_axis=2)
        psnrs.append(psnr_val)
        ssims.append(ssim_val)
    return {
        "psnr_mean": float(np.mean(psnrs)) if psnrs else 0.0,
        "psnr_std": float(np.std(psnrs)) if psnrs else 0.0,
        "ssim_mean": float(np.mean(ssims)) if ssims else 0.0,
        "ssim_std": float(np.std(ssims)) if ssims else 0.0,
        "count": len(psnrs)
    }


def compute_prf(matched, total):
    """基于同一身份对的匹配统计，给出Precision/Recall/F1(与识别率一致)。"""
    if total == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = matched / total  # 同一身份对，无假阳性
    recall = matched / total
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_all(original_dir, processed_dir, face_model, emotion_model, threshold=0.6, device="cpu"):
    """统一评估：人脸识别、表情保持、图像质量。"""
    face_eval = FaceRecognitionEvaluator(model_type=face_model, device=device)
    emo_eval = EmotionRecognitionEvaluator(model_type=emotion_model, device=device)

    fr = face_eval.evaluate_recognition_rate(original_dir, processed_dir, threshold)
    er = emo_eval.evaluate_emotion_preservation(original_dir, processed_dir)
    iq = compute_image_metrics(original_dir, processed_dir)

    if fr:
        prf = compute_prf(fr["matched_count"], fr["total_count"])
    else:
        prf = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    return {
        "face_recognition": fr,
        "emotion_recognition": er,
        "image_quality": iq,
        "prf": prf
    }


def run_improved(dataset_path, output_dir, noise=0.1, component_noise=0.1, exp_blend=1.0, normalize_pose=True, blend_alpha=0.7, render_alpha=0.65):
    """运行改进匿名化策略。"""
    os.makedirs(output_dir, exist_ok=True)
    start = time.time()
    anonymizer = ImprovedShapeAnonymizer()
    anonymizer.process_dataset_improved(
        dataset_path=dataset_path,
        output_dir=output_dir,
        strategy="noise",
        noise_level=noise,
        component_noise_level=component_noise,
        exp_blend_alpha=exp_blend,
        normalize_pose_flag=normalize_pose,
        blend_alpha=blend_alpha,
        render_alpha=render_alpha
    )
    elapsed = time.time() - start
    return {
        "elapsed": elapsed,
        "avg_time": None,  # 由调用处根据图像数量补全
        "output_dir": os.path.join(output_dir, "anonymized_faces")
    }


def main():
    parser = argparse.ArgumentParser(description="基线(口罩遮挡) vs 改进匿名化 对比评测")
    parser.add_argument("--dataset-name", type=str, default="Celeb-A", help="数据集名称(位于dataset目录下)，如 Celeb-A/LFW/FFHQ")
    parser.add_argument("--dataset-root", type=str, default="dataset", help="数据集根目录")
    parser.add_argument("--out-root", type=str, default="results/compare_baseline", help="输出根目录")
    parser.add_argument("--face-model", type=str, default="facenet", choices=["opencv", "facenet", "deepface"], help="人脸识别模型")
    parser.add_argument("--emotion-model", type=str, default="fer", choices=["opencv", "fer", "deepface"], help="表情识别模型")
    parser.add_argument("--threshold", type=float, default=0.6, help="人脸识别阈值")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="设备")
    parser.add_argument("--noise", type=float, default=0.1, help="改进策略: 全局噪声")
    parser.add_argument("--component-noise", type=float, default=0.1, help="改进策略: 组件级噪声")
    parser.add_argument("--exp-blend", type=float, default=1.0, help="表情保留权重(1=完全保留)")
    parser.add_argument("--normalize-pose", action="store_true", help="启用姿态标准化")
    parser.add_argument("--render-alpha", type=float, default=0.65, help="渲染融合alpha(0.6-0.7较自然)")
    parser.add_argument("--use-adaptive-baseline", action="store_true", help="使用自适应贴脸口罩基线")
    args = parser.parse_args()

    dataset_path = os.path.join(args.dataset_root, args.dataset_name)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"数据集不存在: {dataset_path}")

    out_root = Path(args.out_root) / args.dataset_name
    baseline_dir = out_root / "baseline_mask"
    improved_dir = out_root / "improved"

    print("\n=== 运行基线：口罩遮挡 ===")
    if args.use_adaptive_baseline:
        print("使用自适应贴脸口罩基线")
        baseline_stats = run_adaptive_baseline(dataset_path, str(baseline_dir))
    else:
        print("使用矩形口罩基线")
        baseline_stats = apply_mask_baseline(dataset_path, str(baseline_dir))
    print(f"基线完成: {baseline_stats['processed_files']} 张, 平均 {baseline_stats['avg_time']:.3f}s/张")

    print("\n=== 运行改进匿名化 ===")
    improved_stats = run_improved(dataset_path, str(improved_dir), noise=args.noise, component_noise=args.component_noise, exp_blend=args.exp_blend, normalize_pose=args.normalize_pose, render_alpha=args.render_alpha)
    # 计算平均耗时(改进方法)
    processed_count = baseline_stats["processed_files"]  # 与原始数量对齐
    improved_stats["avg_time"] = improved_stats["elapsed"] / processed_count if processed_count else 0.0
    print(f"改进完成: 约 {processed_count} 张, 平均 {improved_stats['avg_time']:.3f}s/张")

    print("\n=== 评测基线 ===")
    baseline_eval = evaluate_all(dataset_path, str(baseline_dir), args.face_model, args.emotion_model, args.threshold, args.device)

    print("\n=== 评测改进方法 ===")
    improved_eval = evaluate_all(dataset_path, improved_stats["output_dir"], args.face_model, args.emotion_model, args.threshold, args.device)

    results = {
        "dataset": dataset_path,
        "baseline": {
            "processing": baseline_stats,
            "evaluation": baseline_eval
        },
        "improved": {
            "processing": improved_stats,
            "evaluation": improved_eval,
            "params": {
                "noise": args.noise,
                "component_noise": args.component_noise,
                "exp_blend": args.exp_blend,
                "normalize_pose": args.normalize_pose
            }
        }
    }

    os.makedirs(out_root, exist_ok=True)
    with open(out_root / "compare_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # 控制台摘要
    def fmt_fr(res):
        if not res:
            return "N/A"
        return f"R={res['recognition_rate']:.2%}, sim={res['avg_similarity']:.3f}"

    def fmt_er(res):
        if not res:
            return "N/A"
        return f"P={res['preservation_rate']:.2%}, sim={res['avg_similarity']:.3f}"

    def fmt_iq(res):
        if not res:
            return "N/A"
        return f"PSNR={res['psnr_mean']:.2f}, SSIM={res['ssim_mean']:.3f}"

    print("\n=== 对比摘要 ===")
    print(f"数据集: {dataset_path}")
    print("- 基线: ")
    print(f"  识别: {fmt_fr(baseline_eval['face_recognition'])}")
    print(f"  表情: {fmt_er(baseline_eval['emotion_recognition'])}")
    print(f"  质量: {fmt_iq(baseline_eval['image_quality'])}")
    print(f"  PRF: P={baseline_eval['prf']['precision']:.2%}, R={baseline_eval['prf']['recall']:.2%}, F1={baseline_eval['prf']['f1']:.2%}")
    print(f"  速度: {baseline_stats['avg_time']:.3f}s/张")

    print("- 改进: ")
    print(f"  识别: {fmt_fr(improved_eval['face_recognition'])}")
    print(f"  表情: {fmt_er(improved_eval['emotion_recognition'])}")
    print(f"  质量: {fmt_iq(improved_eval['image_quality'])}")
    print(f"  PRF: P={improved_eval['prf']['precision']:.2%}, R={improved_eval['prf']['recall']:.2%}, F1={improved_eval['prf']['f1']:.2%}")
    print(f"  速度: {improved_stats['avg_time']:.3f}s/张")

    print(f"\n详细结果已保存: {out_root / 'compare_results.json'}")


if __name__ == "__main__":
    main()
