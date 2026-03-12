# coding: utf-8
"""
消融实验：对比三种人脸匿名化方法
- 方法A：改进匿名化 (噪声+组件扰动+表情保留)
- 方法B：原始3DDFA (仅3D重建+渲染，无匿名化)
- 方法C：k-same算法 (Sweeney等人的k-anonymity，平均形状替换)

评估指标：人脸识别率、表情保持率、图像质量(PSNR/SSIM)、PRF、处理速度
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path

import cv2
import yaml
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm

from shape_anonymization_improved import ImprovedShapeAnonymizer
from face_recognition_eval import FaceRecognitionEvaluator
from emotion_recognition_eval import EmotionRecognitionEvaluator
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
from utils.render import render


def compute_image_metrics(original_dir, processed_dir):
    """计算图像质量指标：PSNR和SSIM"""
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
    """计算Precision/Recall/F1"""
    if total == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = matched / total
    recall = matched / total
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_all(original_dir, processed_dir, face_model, emotion_model, threshold=0.6, device="cpu"):
    """统一评估接口"""
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


def run_method_a_improved(dataset_path, out_dir, noise=0.20, component_noise=0.08, 
                          exp_blend=0.9, render_alpha=0.8, normalize_pose=False):
    """
    方法A：改进匿名化
    - 使用噪声扰动形状参数
    - 组件级局部噪声
    - 表情保留机制
    - 可调渲染融合度
    """
    print("\n[方法A] 改进匿名化")
    print(f"  参数: noise={noise}, component_noise={component_noise}, exp_blend={exp_blend}, render_alpha={render_alpha}")
    
    os.makedirs(out_dir, exist_ok=True)
    start = time.time()
    
    anonymizer = ImprovedShapeAnonymizer()
    anonymizer.process_dataset_improved(
        dataset_path=dataset_path,
        output_dir=out_dir,
        strategy="noise",
        noise_level=noise,
        component_noise_level=component_noise,
        exp_blend_alpha=exp_blend,
        normalize_pose_flag=normalize_pose,
        render_alpha=render_alpha
    )
    
    elapsed = time.time() - start
    return {
        "method": "improved",
        "elapsed": elapsed,
        "output_dir": os.path.join(out_dir, "anonymized_faces"),
        "params": {
            "noise": noise,
            "component_noise": component_noise,
            "exp_blend": exp_blend,
            "render_alpha": render_alpha
        }
    }


def run_method_b_original_3ddfa(dataset_path, out_dir, config_path='configs/mb1_120x120.yml', 
                                render_alpha=0.6):
    """
    方法B：原始3DDFA
    - 仅进行3D人脸重建
    - 使用原始参数渲染（无匿名化）
    - 作为视觉质量和识别率的上限基线
    """
    print("\n[方法B] 原始3DDFA（无匿名化）")
    print(f"  仅3D重建+渲染, render_alpha={render_alpha}")
    
    os.makedirs(out_dir, exist_ok=True)
    
    # 初始化3DDFA
    cfg = yaml.load(open(config_path), Loader=yaml.SafeLoader)
    tddfa = TDDFA(gpu_mode=False, **cfg)
    face_boxes = FaceBoxes()
    
    files = sorted([f for f in os.listdir(dataset_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    
    start = time.time()
    processed = 0
    
    for name in tqdm(files, desc="方法B: 3DDFA重建"):
        img_path = os.path.join(dataset_path, name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # 检测人脸
        boxes = face_boxes(img)
        if len(boxes) == 0:
            # 无人脸则复制原图
            cv2.imwrite(os.path.join(out_dir, name), img)
            processed += 1
            continue
        
        # 提取3DMM参数（使用原始参数，不做匿名化）
        param_lst, roi_box_lst = tddfa(img, boxes)
        
        # 重建3D顶点
        ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
        
        # 渲染到图像上
        img_out = img.copy()
        render(img_out, ver_lst, tddfa.tri, alpha=render_alpha, show_flag=False,
               wfp=os.path.join(out_dir, name))
        
        processed += 1
    
    elapsed = time.time() - start
    return {
        "method": "original_3ddfa",
        "elapsed": elapsed,
        "output_dir": out_dir,
        "processed": processed
    }


def run_method_c_k_same(dataset_path, out_dir, k_value=5, feature_model='facenet', device='cpu'):
    """
    方法C：k-same算法（Sweeney的k-anonymity）
    独立于3DDFA的纯人脸匿名化算法
    
    核心思想：
    1. 检测面部区域（使用FaceBoxes）
    2. 提取人脸特征（使用FaceNet embedding）
    3. 基于特征相似度聚类（KMeans），将相似人脸分成k个一组
    4. 对每组的k张人脸在像素层面取平均，生成匿名化人脸
    5. 将平均后的人脸区域替换回原图对应位置
    
    结果：每组k个人共享平均后的人脸，满足k-anonymity
    """
    print("\n[方法C] k-same算法（k-anonymity）")
    print(f"  Sweeney的k-same: 聚类+像素平均 (k={k_value})")
    
    from sklearn.cluster import KMeans
    from facenet_pytorch import InceptionResnetV1
    import torch
    
    os.makedirs(out_dir, exist_ok=True)
    
    # 初始化人脸检测和特征提取
    face_boxes = FaceBoxes()
    print("  加载FaceNet特征提取模型...")
    facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    files = sorted([f for f in os.listdir(dataset_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    print(f"  处理 {len(files)} 张图像")
    
    # ===== 第一步：检测人脸并提取特征 =====
    print("\n步骤1/4: 检测人脸区域并提取特征")
    face_data = []
    start = time.time()
    
    for name in tqdm(files, desc="提取特征"):
        img_path = os.path.join(dataset_path, name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # 检测人脸
        boxes = face_boxes(img)
        if len(boxes) == 0:
            continue
        
        box = boxes[0][:4]  # [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, box)
        
        # 确保边界合法
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            continue
        
        # 提取人脸区域
        face_region = img[y1:y2, x1:x2].copy()
        
        # 预处理并提取FaceNet特征
        face_resized = cv2.resize(face_region, (160, 160))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
        face_tensor = (face_tensor - 0.5) / 0.5  # 归一化到[-1, 1]
        face_tensor = face_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = facenet_model(face_tensor).cpu().numpy().flatten()
        
        face_data.append({
            'filename': name,
            'image': img,
            'face_box': (x1, y1, x2, y2),
            'face_region': face_region,
            'embedding': embedding
        })
    
    print(f"  成功提取 {len(face_data)} 张人脸")
    
    if len(face_data) == 0:
        print("  错误：无有效人脸，退出")
        return None
    
    # ===== 第二步：基于特征相似度聚类 =====
    print(f"\n步骤2/4: 聚类（k={k_value}）")
    embeddings = np.array([fd['embedding'] for fd in face_data])
    
    # 计算聚类数量
    n_clusters = max(1, len(face_data) // k_value)
    print(f"  {len(face_data)}张人脸 → {n_clusters}个聚类组")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    for i, fd in enumerate(face_data):
        fd['cluster'] = cluster_labels[i]
    
    # ===== 第三步：对每个聚类计算平均人脸 =====
    print("\n步骤3/4: 计算每组的平均人脸")
    cluster_avg_faces = {}
    
    for cluster_id in range(n_clusters):
        cluster_faces = [fd for fd in face_data if fd['cluster'] == cluster_id]
        
        if len(cluster_faces) == 0:
            continue
        
        # 统一尺寸（使用聚类中第一张人脸的尺寸作为基准）
        ref_shape = cluster_faces[0]['face_region'].shape[:2]
        
        # 收集所有人脸区域并resize到统一尺寸
        aligned_faces = []
        for fd in cluster_faces:
            face_resized = cv2.resize(fd['face_region'], (ref_shape[1], ref_shape[0]))
            aligned_faces.append(face_resized.astype(np.float32))
        
        # 计算像素级平均
        avg_face = np.mean(aligned_faces, axis=0).astype(np.uint8)
        cluster_avg_faces[cluster_id] = avg_face
        
        print(f"  聚类 {cluster_id}: {len(cluster_faces)} 张人脸")
    
    # ===== 第四步：替换原图中的人脸区域 =====
    print("\n步骤4/4: 生成匿名化图像")
    processed = 0
    
    for fd in tqdm(face_data, desc="替换人脸"):
        img = fd['image'].copy()
        x1, y1, x2, y2 = fd['face_box']
        cluster_id = fd['cluster']
        
        # 获取该聚类的平均人脸
        avg_face = cluster_avg_faces[cluster_id]
        
        # Resize平均人脸到原始人脸框大小
        avg_face_resized = cv2.resize(avg_face, (x2 - x1, y2 - y1))
        
        # 替换人脸区域（使用简单的羽化边缘）
        mask = np.ones((y2 - y1, x2 - x1, 3), dtype=np.float32)
        
        # 创建渐变mask实现平滑过渡
        fade_size = min(10, (y2 - y1) // 10, (x2 - x1) // 10)
        if fade_size > 0:
            for i in range(fade_size):
                alpha = i / fade_size
                mask[i, :] *= alpha
                mask[-(i+1), :] *= alpha
                mask[:, i] *= alpha
                mask[:, -(i+1)] *= alpha
        
        # 混合
        img[y1:y2, x1:x2] = (
            mask * avg_face_resized.astype(np.float32) +
            (1 - mask) * img[y1:y2, x1:x2].astype(np.float32)
        ).astype(np.uint8)
        
        # 保存
        cv2.imwrite(os.path.join(out_dir, fd['filename']), img)
        processed += 1
    
    elapsed = time.time() - start
    
    # 保存聚类信息
    cluster_info = {}
    for cluster_id in range(n_clusters):
        cluster_faces = [fd['filename'] for fd in face_data if fd['cluster'] == cluster_id]
        cluster_info[int(cluster_id)] = {
            'size': len(cluster_faces),
            'filenames': cluster_faces
        }
    
    import json
    with open(os.path.join(out_dir, 'k_same_clusters.json'), 'w', encoding='utf-8') as f:
        json.dump(cluster_info, f, indent=4, ensure_ascii=False)
    
    return {
        "method": "k_same",
        "elapsed": elapsed,
        "output_dir": out_dir,
        "processed": processed,
        "k_value": k_value,
        "n_clusters": n_clusters
    }


def main():
    parser = argparse.ArgumentParser(description="消融实验：改进方法 vs 原始3DDFA vs k-same")
    
    # 基本参数
    parser.add_argument("--dataset-name", type=str, default="Celeb-A", help="数据集名称")
    parser.add_argument("--dataset-root", type=str, default="dataset", help="数据集根目录")
    parser.add_argument("--out-root", type=str, default="results/ablation", help="输出根目录")
    
    # 评估参数
    parser.add_argument("--face-model", type=str, default="facenet", 
                       choices=["opencv", "facenet", "deepface"], help="人脸识别模型")
    parser.add_argument("--emotion-model", type=str, default="fer",
                       choices=["opencv", "fer", "deepface"], help="表情识别模型")
    parser.add_argument("--threshold", type=float, default=0.6, help="人脸识别阈值")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    
    # 方法A参数（改进匿名化）
    parser.add_argument("--noise", type=float, default=0.20, help="方法A: 全局噪声水平")
    parser.add_argument("--component-noise", type=float, default=0.08, help="方法A: 组件噪声")
    parser.add_argument("--exp-blend", type=float, default=0.9, help="方法A: 表情保留权重")
    parser.add_argument("--render-alpha-a", type=float, default=0.8, help="方法A: 渲染融合度")
    
    # 方法B参数
    parser.add_argument("--render-alpha-b", type=float, default=0.6, help="方法B: 渲染融合度")
    
    # 方法C参数（k-same）
    parser.add_argument("--k-value", type=int, default=5, help="方法C: k-same的k值（每组人脸数量）")
    
    args = parser.parse_args()
    
    dataset_path = os.path.join(args.dataset_root, args.dataset_name)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"数据集不存在: {dataset_path}")
    
    out_root = Path(args.out_root) / args.dataset_name
    out_a = out_root / "method_a_improved"
    out_b = out_root / "method_b_original_3ddfa"
    out_c = out_root / "method_c_k_same"
    
    print("="*70)
    print("消融实验：三种人脸匿名化方法对比")
    print("="*70)
    print(f"数据集: {dataset_path}")
    print(f"输出目录: {out_root}")
    
    # ===== 运行三种方法 =====
    print("\n" + "="*70)
    results_a = run_method_a_improved(
        dataset_path, str(out_a),
        noise=args.noise,
        component_noise=args.component_noise,
        exp_blend=args.exp_blend,
        render_alpha=args.render_alpha_a
    )
    
    print("\n" + "="*70)
    results_b = run_method_b_original_3ddfa(
        dataset_path, str(out_b),
        render_alpha=args.render_alpha_b
    )
    
    print("\n" + "="*70)
    results_c = run_method_c_k_same(
        dataset_path, str(out_c),
        k_value=args.k_value,
        device=args.device
    )
    
    # ===== 评估三种方法 =====
    print("\n" + "="*70)
    print("开始评估...")
    print("="*70)
    
    print("\n评估方法A...")
    eval_a = evaluate_all(dataset_path, results_a["output_dir"], 
                         args.face_model, args.emotion_model, args.threshold, args.device)
    
    print("\n评估方法B...")
    eval_b = evaluate_all(dataset_path, results_b["output_dir"],
                         args.face_model, args.emotion_model, args.threshold, args.device)
    
    print("\n评估方法C...")
    eval_c = evaluate_all(dataset_path, results_c["output_dir"],
                         args.face_model, args.emotion_model, args.threshold, args.device)
    
    # 计算平均耗时
    count_a = eval_a["face_recognition"]["total_count"] if eval_a["face_recognition"] else 0
    count_b = eval_b["face_recognition"]["total_count"] if eval_b["face_recognition"] else 0
    count_c = eval_c["face_recognition"]["total_count"] if eval_c["face_recognition"] else 0
    
    results_a["avg_time"] = results_a["elapsed"] / count_a if count_a else 0.0
    results_b["avg_time"] = results_b["elapsed"] / count_b if count_b else 0.0
    results_c["avg_time"] = results_c["elapsed"] / count_c if count_c else 0.0
    
    # ===== 保存完整结果 =====
    final_results = {
        "dataset": dataset_path,
        "method_a_improved": {
            "processing": results_a,
            "evaluation": eval_a
        },
        "method_b_original_3ddfa": {
            "processing": results_b,
            "evaluation": eval_b
        },
        "method_c_k_same": {
            "processing": results_c,
            "evaluation": eval_c
        }
    }
    
    os.makedirs(out_root, exist_ok=True)
    with open(out_root / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)
    
    # ===== 打印对比摘要 =====
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
    
    print("\n" + "="*70)
    print("消融实验对比摘要")
    print("="*70)
    print(f"数据集: {dataset_path}")
    
    print("\n- 方法A (改进匿名化):")
    print(f"  识别: {fmt_fr(eval_a['face_recognition'])}")
    print(f"  表情: {fmt_er(eval_a['emotion_recognition'])}")
    print(f"  质量: {fmt_iq(eval_a['image_quality'])}")
    print(f"  PRF: P={eval_a['prf']['precision']:.2%}, R={eval_a['prf']['recall']:.2%}, F1={eval_a['prf']['f1']:.2%}")
    print(f"  速度: {results_a['avg_time']:.3f}s/张")
    
    print("\n- 方法B (原始3DDFA):")
    print(f"  识别: {fmt_fr(eval_b['face_recognition'])}")
    print(f"  表情: {fmt_er(eval_b['emotion_recognition'])}")
    print(f"  质量: {fmt_iq(eval_b['image_quality'])}")
    print(f"  PRF: P={eval_b['prf']['precision']:.2%}, R={eval_b['prf']['recall']:.2%}, F1={eval_b['prf']['f1']:.2%}")
    print(f"  速度: {results_b['avg_time']:.3f}s/张")
    
    print("\n- 方法C (k-same算法):")
    print(f"  识别: {fmt_fr(eval_c['face_recognition'])}")
    print(f"  表情: {fmt_er(eval_c['emotion_recognition'])}")
    print(f"  质量: {fmt_iq(eval_c['image_quality'])}")
    print(f"  PRF: P={eval_c['prf']['precision']:.2%}, R={eval_c['prf']['recall']:.2%}, F1={eval_c['prf']['f1']:.2%}")
    print(f"  速度: {results_c['avg_time']:.3f}s/张")
    print(f"  k值: {results_c.get('k_value', 'N/A')}, 聚类数: {results_c.get('n_clusters', 'N/A')}")
    
    print("\n" + "="*70)
    print("说明：")
    print("- 识别率(R): 越低越好，表示匿名化效果越强")
    print("- 表情保持率(P): 越高越好，表示表情信息保留越完整")
    print("- 图像质量(PSNR/SSIM): 越高越好，表示视觉质量越接近原图")
    print("="*70)
    print(f"\n详细结果已保存: {out_root / 'ablation_results.json'}")
    print("="*70)


if __name__ == "__main__":
    main()
