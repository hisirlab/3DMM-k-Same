# coding: utf-8
"""
实验结果可视化脚本
生成清晰的对比图表：
1. 不同数据集上的Ours vs Baseline对比
2. 消融实验：Full Method vs 3DDFA-V2 vs K-Same
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import sys
sys.path.append(r"D:\Python311\Lib\site-packages")
import seaborn as sns

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# ===================== 数据定义 =====================

# 表1：Ours vs Baseline 在三个数据集上的对比
cross_dataset_data = {
    'Celeb-A': {
        'Ours': {
            'identity_rate': 25.93,
            'identity_sim': 0.551,
            'expression_rate': 69.05,
            'expression_sim': 0.816,
            'psnr': 20.53,
            'ssim': 0.868,
            'time': 0.036
        },
        'Baseline': {
            'identity_rate': 99.06,
            'identity_sim': 0.797,
            'expression_rate': 11.11,
            'expression_sim': 0.413,
            'psnr': 15.13,
            'ssim': 0.825,
            'time': 0.010
        }
    },
    'FFHQ': {
        'Ours': {
            'identity_rate': 32.59,
            'identity_sim': 0.570,
            'expression_rate': 61.36,
            'expression_sim': 0.756,
            'psnr': 17.82,
            'ssim': 0.814,
            'time': 0.039
        },
        'Baseline': {
            'identity_rate': 98.37,
            'identity_sim': 0.822,
            'expression_rate': 7.28,
            'expression_sim': 0.263,
            'psnr': 14.84,
            'ssim': 0.810,
            'time': 0.021
        }
    },
    'LFW': {
        'Ours': {
            'identity_rate': 22.43,
            'identity_sim': 0.540,
            'expression_rate': 54.46,
            'expression_sim': 0.717,
            'psnr': 21.32,
            'ssim': 0.901,
            'time': 0.038
        },
        'Baseline': {
            'identity_rate': 98.95,
            'identity_sim': 0.817,
            'expression_rate': 20.00,
            'expression_sim': 0.541,
            'psnr': 18.23,
            'ssim': 0.887,
            'time': 0.022
        }
    }
}

# 表2：消融实验（在Celeb-A上）
ablation_data = {
    'Full Method': {
        'identity_rate': 25.93,
        'identity_sim': 0.551,
        'expression_rate': 69.05,
        'expression_sim': 0.816,
        'psnr': 20.53,
        'ssim': 0.868,
        'time': 0.036
    },
    '3DDFA-V2': {
        'identity_rate': 97.96,
        'identity_sim': 0.740,
        'expression_rate': 72.22,
        'expression_sim': 0.859,
        'psnr': 22.38,
        'ssim': 0.912,
        'time': 0.043
    },
    'K-Same': {
        'identity_rate': 46.67,
        'identity_sim': 0.592,
        'expression_rate': 38.10,
        'expression_sim': 0.605,
        'psnr': 22.73,
        'ssim': 0.862,
        'time': 0.061
    }
}


# ===================== 绘图函数 =====================

def plot_cross_dataset_comparison():
    """绘制跨数据集对比图（Ours vs Baseline）- 3张独立图"""
    
    datasets = ['Celeb-A', 'FFHQ', 'LFW']
    colors_ours = '#2E86AB'  # 蓝色
    colors_baseline = '#A23B72'  # 紫红色
    
    # 准备数据：横坐标为 Ours, Baseline, Ours, Baseline, Ours, Baseline
    x_labels = []
    for dataset in datasets:
        x_labels.extend(['Ours', 'Baseline'])
    x_pos = np.arange(len(x_labels))
    
    # ========== 图1：Identity Rate（柱状图） + Average Similarity（折线图） ==========
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    # 左Y轴：Identity Rate（柱状图）
    id_rates = []
    id_colors = []
    for dataset in datasets:
        id_rates.append(cross_dataset_data[dataset]['Ours']['identity_rate'])
        id_colors.append(colors_ours)
        id_rates.append(cross_dataset_data[dataset]['Baseline']['identity_rate'])
        id_colors.append(colors_baseline)
    
    bars = ax1.bar(x_pos, id_rates, color=id_colors, alpha=0.7, width=0.6)
    ax1.set_ylabel('Identity Recognition Rate (%)', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Methods across Datasets', fontweight='bold', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    ax1.set_ylim(0, 110)
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数据集分隔线
    for i in [1.5, 3.5]:
        ax1.axvline(x=i, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # 添加数据集标签
    dataset_positions = [0.5, 2.5, 4.5]
    for pos, dataset in zip(dataset_positions, datasets):
        ax1.text(pos, 105, dataset, ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 添加柱状图数值标签
    for i, (bar, rate) in enumerate(zip(bars, id_rates)):
        ax1.text(bar.get_x() + bar.get_width()/2., rate + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 右Y轴：Average Similarity（折线图）
    ax1_right = ax1.twinx()
    id_sims = []
    for dataset in datasets:
        id_sims.append(cross_dataset_data[dataset]['Ours']['identity_sim'])
        id_sims.append(cross_dataset_data[dataset]['Baseline']['identity_sim'])
    
    line = ax1_right.plot(x_pos, id_sims, 'o-', color='#E63946', linewidth=2.5, 
                          markersize=8, label='Avg Similarity')
    ax1_right.set_ylabel('Average Similarity', fontweight='bold', fontsize=12, color='#E63946')
    ax1_right.tick_params(axis='y', labelcolor='#E63946')
    ax1_right.set_ylim(0, 1.0)
    
    # 添加折线数值标签
    for i, (x, y) in enumerate(zip(x_pos, id_sims)):
        ax1_right.text(x, y + 0.03, f'{y:.3f}', ha='center', va='bottom', 
                      fontsize=9, color='#E63946', fontweight='bold')
    
    ax1.set_title('Identity Recognition Rate & Average Similarity\n(Lower is Better)', 
                  fontweight='bold', fontsize=14, pad=20)
    
    # 添加图例
    bar_ours = plt.Rectangle((0,0),1,1, fc=colors_ours, alpha=0.7)
    bar_baseline = plt.Rectangle((0,0),1,1, fc=colors_baseline, alpha=0.7)
    ax1.legend([bar_ours, bar_baseline, line[0]], 
              ['Ours (Rate)', 'Baseline (Rate)', 'Avg Similarity'], 
              loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/cross_dataset_identity.png', dpi=300, bbox_inches='tight', transparent=True)
    print("✓ 身份识别对比图已保存: results/cross_dataset_identity.png")
    plt.close()
    
    # ========== 图2：Expression Rate（柱状图） + Average Similarity（折线图） ==========
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # 左Y轴：Expression Rate（柱状图）
    exp_rates = []
    exp_colors = []
    for dataset in datasets:
        exp_rates.append(cross_dataset_data[dataset]['Ours']['expression_rate'])
        exp_colors.append(colors_ours)
        exp_rates.append(cross_dataset_data[dataset]['Baseline']['expression_rate'])
        exp_colors.append(colors_baseline)
    
    bars = ax2.bar(x_pos, exp_rates, color=exp_colors, alpha=0.7, width=0.6)
    ax2.set_ylabel('Expression Preservation Rate (%)', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Methods across Datasets', fontweight='bold', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    ax2.set_ylim(0, 80)
    ax2.grid(axis='y', alpha=0.3)
    
    # 分隔线和数据集标签
    for i in [1.5, 3.5]:
        ax2.axvline(x=i, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    for pos, dataset in zip(dataset_positions, datasets):
        ax2.text(pos, 75, dataset, ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 柱状图数值标签（Ours的数字位置调高，避免与折线重叠）
    for i, (bar, rate) in enumerate(zip(bars, exp_rates)):
        # 每隔2个是Ours（索引0,2,4），数字调高
        offset = 4.0 if i % 2 == 0 else 1.5
        ax2.text(bar.get_x() + bar.get_width()/2., rate + offset,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 右Y轴：Expression Similarity（折线图）
    ax2_right = ax2.twinx()
    exp_sims = []
    for dataset in datasets:
        exp_sims.append(cross_dataset_data[dataset]['Ours']['expression_sim'])
        exp_sims.append(cross_dataset_data[dataset]['Baseline']['expression_sim'])
    
    line = ax2_right.plot(x_pos, exp_sims, 'o-', color='#06A77D', linewidth=2.5,
                          markersize=8, label='Avg Similarity')
    ax2_right.set_ylabel('Average Similarity', fontweight='bold', fontsize=12, color='#06A77D')
    ax2_right.tick_params(axis='y', labelcolor='#06A77D')
    ax2_right.set_ylim(0, 1.0)
    
    # 折线数值标签
    for x, y in zip(x_pos, exp_sims):
        ax2_right.text(x, y + 0.03, f'{y:.3f}', ha='center', va='bottom',
                      fontsize=9, color='#06A77D', fontweight='bold')
    
    ax2.set_title('Expression Preservation Rate & Average Similarity\n(Higher is Better)',
                  fontweight='bold', fontsize=14, pad=20)
    
    # 图例（右上角内部）
    bar_ours = plt.Rectangle((0,0),1,1, fc=colors_ours, alpha=0.7)
    bar_baseline = plt.Rectangle((0,0),1,1, fc=colors_baseline, alpha=0.7)
    ax2.legend([bar_ours, bar_baseline, line[0]],
              ['Ours (Rate)', 'Baseline (Rate)', 'Avg Similarity'],
              loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/cross_dataset_expression.png', dpi=300, bbox_inches='tight', transparent=True)
    print("✓ 表情保持对比图已保存: results/cross_dataset_expression.png")
    plt.close()
    
    # ========== 图3：PSNR（柱状图） + SSIM（折线图） ==========
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    
    # 左Y轴：PSNR（柱状图）
    psnrs = []
    psnr_colors = []
    for dataset in datasets:
        psnrs.append(cross_dataset_data[dataset]['Ours']['psnr'])
        psnr_colors.append(colors_ours)
        psnrs.append(cross_dataset_data[dataset]['Baseline']['psnr'])
        psnr_colors.append(colors_baseline)
    
    bars = ax3.bar(x_pos, psnrs, color=psnr_colors, alpha=0.7, width=0.6)
    ax3.set_ylabel('PSNR (dB)', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Methods across Datasets', fontweight='bold', fontsize=12)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x_labels, rotation=45, ha='right')
    ax3.set_ylim(0, 25)
    ax3.grid(axis='y', alpha=0.3)
    
    # 分隔线和数据集标签
    for i in [1.5, 3.5]:
        ax3.axvline(x=i, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    for pos, dataset in zip(dataset_positions, datasets):
        ax3.text(pos, 23.5, dataset, ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 柱状图数值标签
    for bar, psnr in zip(bars, psnrs):
        ax3.text(bar.get_x() + bar.get_width()/2., psnr + 0.5,
                f'{psnr:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 右Y轴：SSIM（折线图）
    ax3_right = ax3.twinx()
    ssims = []
    for dataset in datasets:
        ssims.append(cross_dataset_data[dataset]['Ours']['ssim'])
        ssims.append(cross_dataset_data[dataset]['Baseline']['ssim'])
    
    line = ax3_right.plot(x_pos, ssims, 'o-', color='#F77F00', linewidth=2.5,
                          markersize=8, label='SSIM')
    ax3_right.set_ylabel('SSIM Score', fontweight='bold', fontsize=12, color='#F77F00')
    ax3_right.tick_params(axis='y', labelcolor='#F77F00')
    ax3_right.set_ylim(0.75, 0.95)
    
    # 折线数值标签
    for x, y in zip(x_pos, ssims):
        ax3_right.text(x, y + 0.005, f'{y:.3f}', ha='center', va='bottom',
                      fontsize=9, color='#F77F00', fontweight='bold')
    
    ax3.set_title('Image Quality: PSNR & SSIM\n(Higher is Better)',
                  fontweight='bold', fontsize=14, pad=20)
    
    # 图例（右上角内部）
    bar_ours = plt.Rectangle((0,0),1,1, fc=colors_ours, alpha=0.7)
    bar_baseline = plt.Rectangle((0,0),1,1, fc=colors_baseline, alpha=0.7)
    ax3.legend([bar_ours, bar_baseline, line[0]],
              ['Ours (PSNR)', 'Baseline (PSNR)', 'SSIM'],
              loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/cross_dataset_quality.png', dpi=300, bbox_inches='tight', transparent=True)
    print("✓ 图像质量对比图已保存: results/cross_dataset_quality.png")
    plt.close()


def plot_ablation_study():
    """绘制消融实验对比图（Full Method vs 3DDFA-V2 vs K-Same）- 3张独立图"""
    
    methods = ['Full Method', '3DDFA-V2', 'K-Same']
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # 蓝、紫红、橙
    x_pos = np.arange(len(methods))
    
    # ========== 图1：Identity Rate（柱状图） + Average Similarity（折线图） ==========
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # 左Y轴：Identity Rate（柱状图）
    id_rates = [ablation_data[m]['identity_rate'] for m in methods]
    bars = ax1.bar(x_pos, id_rates, color=colors, alpha=0.7, width=0.6)
    ax1.set_ylabel('Identity Recognition Rate (%)', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Methods', fontweight='bold', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, fontsize=11)
    ax1.set_ylim(0, 110)
    ax1.grid(axis='y', alpha=0.3)
    
    # 柱状图数值标签
    for bar, rate in zip(bars, id_rates):
        ax1.text(bar.get_x() + bar.get_width()/2., rate + 2,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 右Y轴：Average Similarity（折线图）
    ax1_right = ax1.twinx()
    id_sims = [ablation_data[m]['identity_sim'] for m in methods]
    line = ax1_right.plot(x_pos, id_sims, 'o-', color='#E63946', linewidth=2.5,
                          markersize=10, label='Avg Similarity')
    ax1_right.set_ylabel('Average Similarity', fontweight='bold', fontsize=12, color='#E63946')
    ax1_right.tick_params(axis='y', labelcolor='#E63946')
    ax1_right.set_ylim(0, 1.0)
    
    # 折线数值标签
    for x, y in zip(x_pos, id_sims):
        ax1_right.text(x, y + 0.04, f'{y:.3f}', ha='center', va='bottom',
                      fontsize=10, color='#E63946', fontweight='bold')
    
    ax1.set_title('Ablation Study: Identity Recognition Rate & Average Similarity\n(Lower is Better)',
                  fontweight='bold', fontsize=14, pad=15)
    
    # 图例
    bars_legend = [plt.Rectangle((0,0),1,1, fc=c, alpha=0.7) for c in colors]
    ax1.legend(bars_legend + line, methods + ['Avg Similarity'],
              loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/ablation_identity.png', dpi=300, bbox_inches='tight', transparent=True)
    print("✓ 消融实验-身份识别已保存: results/ablation_identity.png")
    plt.close()
    
    # ========== 图2：Expression Rate（柱状图） + Average Similarity（折线图） ==========
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # 左Y轴：Expression Rate（柱状图）
    exp_rates = [ablation_data[m]['expression_rate'] for m in methods]
    bars = ax2.bar(x_pos, exp_rates, color=colors, alpha=0.7, width=0.6)
    ax2.set_ylabel('Expression Preservation Rate (%)', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Methods', fontweight='bold', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(methods, fontsize=11)
    ax2.set_ylim(0, 80)
    ax2.grid(axis='y', alpha=0.3)
    
    # 柱状图数值标签
    for bar, rate in zip(bars, exp_rates):
        ax2.text(bar.get_x() + bar.get_width()/2., rate + 1.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 右Y轴：Expression Similarity（折线图）
    ax2_right = ax2.twinx()
    exp_sims = [ablation_data[m]['expression_sim'] for m in methods]
    line = ax2_right.plot(x_pos, exp_sims, 'o-', color='#06A77D', linewidth=2.5,
                          markersize=10, label='Avg Similarity')
    ax2_right.set_ylabel('Average Similarity', fontweight='bold', fontsize=12, color='#06A77D')
    ax2_right.tick_params(axis='y', labelcolor='#06A77D')
    ax2_right.set_ylim(0, 1.0)
    
    # 折线数值标签
    for x, y in zip(x_pos, exp_sims):
        ax2_right.text(x, y + 0.04, f'{y:.3f}', ha='center', va='bottom',
                      fontsize=10, color='#06A77D', fontweight='bold')
    
    ax2.set_title('Ablation Study: Expression Preservation Rate & Average Similarity\n(Higher is Better)',
                  fontweight='bold', fontsize=14, pad=15)
    
    # 图例（右上角内部）
    bars_legend = [plt.Rectangle((0,0),1,1, fc=c, alpha=0.7) for c in colors]
    ax2.legend(bars_legend + line, methods + ['Avg Similarity'],
              loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/ablation_expression.png', dpi=300, bbox_inches='tight', transparent=True)
    print("✓ 消融实验-表情保持已保存: results/ablation_expression.png")
    plt.close()
    
    # ========== 图3：PSNR（柱状图） + SSIM（折线图） ==========
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    # 左Y轴：PSNR（柱状图）
    psnrs = [ablation_data[m]['psnr'] for m in methods]
    bars = ax3.bar(x_pos, psnrs, color=colors, alpha=0.7, width=0.6)
    ax3.set_ylabel('PSNR (dB)', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Methods', fontweight='bold', fontsize=12)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(methods, fontsize=11)
    ax3.set_ylim(0, 25)
    ax3.grid(axis='y', alpha=0.3)
    
    # 柱状图数值标签
    for bar, psnr in zip(bars, psnrs):
        ax3.text(bar.get_x() + bar.get_width()/2., psnr + 0.5,
                f'{psnr:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 右Y轴：SSIM（折线图）
    ax3_right = ax3.twinx()
    ssims = [ablation_data[m]['ssim'] for m in methods]
    line = ax3_right.plot(x_pos, ssims, 'o-', color='#F77F00', linewidth=2.5,
                          markersize=10, label='SSIM')
    ax3_right.set_ylabel('SSIM Score', fontweight='bold', fontsize=12, color='#F77F00')
    ax3_right.tick_params(axis='y', labelcolor='#F77F00')
    ax3_right.set_ylim(0.8, 0.95)
    
    # 折线数值标签
    for x, y in zip(x_pos, ssims):
        ax3_right.text(x, y + 0.008, f'{y:.3f}', ha='center', va='bottom',
                      fontsize=10, color='#F77F00', fontweight='bold')
    
    ax3.set_title('Ablation Study: Image Quality (PSNR & SSIM)\n(Higher is Better)',
                  fontweight='bold', fontsize=14, pad=15)
    
    # 图例（字体缩小）
    bars_legend = [plt.Rectangle((0,0),1,1, fc=c, alpha=0.7) for c in colors]
    ax3.legend(bars_legend + line, methods + ['SSIM'],
              loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/ablation_quality.png', dpi=300, bbox_inches='tight', transparent=True)
    print("✓ 消融实验-图像质量已保存: results/ablation_quality.png")
    plt.close()


def plot_radar_chart():
    """绘制雷达图：综合展示各方法的多维度性能"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), subplot_kw=dict(projection='polar'))
    fig.suptitle('Comprehensive Performance Comparison', fontsize=16, fontweight='bold')
    
    # 图1：跨数据集 Ours vs Baseline（取Celeb-A数据）
    categories = ['Privacy\n(↓ID Rate)', 'Expression\nPreservation', 
                 'PSNR', 'SSIM', 'Efficiency\n(↓Time)']
    N = len(categories)
    
    # 归一化数据到0-1（越大越好）
    ours_celeba = cross_dataset_data['Celeb-A']['Ours']
    baseline_celeba = cross_dataset_data['Celeb-A']['Baseline']
    
    ours_values = [
        1 - ours_celeba['identity_rate'] / 100,  # 身份识别率：越低越好，所以取反
        ours_celeba['expression_rate'] / 100,     # 表情保持：越高越好
        ours_celeba['psnr'] / 25,                 # PSNR：归一化到0-1
        ours_celeba['ssim'],                      # SSIM：本身0-1
        1 - ours_celeba['time'] / 0.05           # 效率：越快越好，所以取反
    ]
    
    baseline_values = [
        1 - baseline_celeba['identity_rate'] / 100,
        baseline_celeba['expression_rate'] / 100,
        baseline_celeba['psnr'] / 25,
        baseline_celeba['ssim'],
        1 - baseline_celeba['time'] / 0.05
    ]
    
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    ours_values += ours_values[:1]
    baseline_values += baseline_values[:1]
    angles += angles[:1]
    
    ax1.plot(angles, ours_values, 'o-', linewidth=2, label='Ours', color='#2E86AB')
    ax1.fill(angles, ours_values, alpha=0.25, color='#2E86AB')
    ax1.plot(angles, baseline_values, 'o-', linewidth=2, label='Baseline', color='#A23B72')
    ax1.fill(angles, baseline_values, alpha=0.25, color='#A23B72')
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, fontsize=10)
    ax1.set_ylim(0, 1)
    ax1.set_title('Ours vs Baseline (Celeb-A)', fontsize=12, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax1.grid(True)
    
    # 图2：消融实验（Full Method vs 3DDFA vs K-Same）
    full_values = [
        1 - ablation_data['Full Method']['identity_rate'] / 100,
        ablation_data['Full Method']['expression_rate'] / 100,
        ablation_data['Full Method']['psnr'] / 25,
        ablation_data['Full Method']['ssim'],
        1 - ablation_data['Full Method']['time'] / 0.07
    ]
    
    tddfa_values = [
        1 - ablation_data['3DDFA-V2']['identity_rate'] / 100,
        ablation_data['3DDFA-V2']['expression_rate'] / 100,
        ablation_data['3DDFA-V2']['psnr'] / 25,
        ablation_data['3DDFA-V2']['ssim'],
        1 - ablation_data['3DDFA-V2']['time'] / 0.07
    ]
    
    ksame_values = [
        1 - ablation_data['K-Same']['identity_rate'] / 100,
        ablation_data['K-Same']['expression_rate'] / 100,
        ablation_data['K-Same']['psnr'] / 25,
        ablation_data['K-Same']['ssim'],
        1 - ablation_data['K-Same']['time'] / 0.07
    ]
    
    full_values += full_values[:1]
    tddfa_values += tddfa_values[:1]
    ksame_values += ksame_values[:1]
    
    ax2.plot(angles, full_values, 'o-', linewidth=2, label='Full Method', color='#2E86AB')
    ax2.fill(angles, full_values, alpha=0.25, color='#2E86AB')
    ax2.plot(angles, tddfa_values, 'o-', linewidth=2, label='3DDFA-V2', color='#A23B72')
    ax2.fill(angles, tddfa_values, alpha=0.25, color='#A23B72')
    ax2.plot(angles, ksame_values, 'o-', linewidth=2, label='K-Same', color='#F18F01')
    ax2.fill(angles, ksame_values, alpha=0.25, color='#F18F01')
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.set_title('Ablation Study (Celeb-A)', fontsize=12, fontweight='bold', pad=20)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/radar_comparison.png', dpi=300, bbox_inches='tight', transparent=True)
    print("✓ 雷达图已保存: results/radar_comparison.png")
    plt.show()


def plot_ablation_radar():
    """单独绘制消融实验的雷达图"""
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # 定义指标
    categories = ['Privacy\n(↓ID Rate)', 'Expression\nPreservation', 
                 'PSNR', 'SSIM', 'Efficiency\n(↓Time)']
    N = len(categories)
    
    # 归一化数据到0-1（越大越好）
    full_values = [
        1 - ablation_data['Full Method']['identity_rate'] / 100,
        ablation_data['Full Method']['expression_rate'] / 100,
        ablation_data['Full Method']['psnr'] / 25,
        ablation_data['Full Method']['ssim'],
        1 - ablation_data['Full Method']['time'] / 0.07
    ]
    
    tddfa_values = [
        1 - ablation_data['3DDFA-V2']['identity_rate'] / 100,
        ablation_data['3DDFA-V2']['expression_rate'] / 100,
        ablation_data['3DDFA-V2']['psnr'] / 25,
        ablation_data['3DDFA-V2']['ssim'],
        1 - ablation_data['3DDFA-V2']['time'] / 0.07
    ]
    
    ksame_values = [
        1 - ablation_data['K-Same']['identity_rate'] / 100,
        ablation_data['K-Same']['expression_rate'] / 100,
        ablation_data['K-Same']['psnr'] / 25,
        ablation_data['K-Same']['ssim'],
        1 - ablation_data['K-Same']['time'] / 0.07
    ]
    
    # 闭合雷达图
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    full_values += full_values[:1]
    tddfa_values += tddfa_values[:1]
    ksame_values += ksame_values[:1]
    angles += angles[:1]
    
    # 绘制三条线
    ax.plot(angles, full_values, 'o-', linewidth=2.5, label='Full Method', color='#2E86AB', markersize=8)
    ax.fill(angles, full_values, alpha=0.25, color='#2E86AB')
    
    ax.plot(angles, tddfa_values, 'o-', linewidth=2.5, label='3DDFA-V2', color='#A23B72', markersize=8)
    ax.fill(angles, tddfa_values, alpha=0.25, color='#A23B72')
    
    ax.plot(angles, ksame_values, 'o-', linewidth=2.5, label='K-Same', color='#F18F01', markersize=8)
    ax.fill(angles, ksame_values, alpha=0.25, color='#F18F01')
    
    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Ablation Study Performance Comparison', 
                fontsize=14, fontweight='bold', pad=20)
    
    # 图例放在左上角，贴近雷达图主体
    ax.legend(loc='upper left', bbox_to_anchor=(0.05, 1.0), fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/ablation_radar.png', dpi=300, bbox_inches='tight', transparent=True)
    print("✓ 消融实验雷达图已保存: results/ablation_radar.png")
    plt.close()


def plot_heatmap():
    """绘制热力图：展示所有方法在所有指标上的表现"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle('Performance Heatmap (Normalized)', fontsize=16, fontweight='bold')
    
    # 热力图1：跨数据集对比
    datasets = ['Celeb-A', 'FFHQ', 'LFW']
    metrics = ['Privacy\n(↓ID Rate)', 'Expression\nPreservation', 'PSNR', 'SSIM', 'Efficiency']
    
    # 构建矩阵（Ours vs Baseline）
    matrix1 = np.zeros((len(datasets) * 2, len(metrics)))
    
    for i, dataset in enumerate(datasets):
        ours = cross_dataset_data[dataset]['Ours']
        baseline = cross_dataset_data[dataset]['Baseline']
        
        # Ours
        matrix1[i*2, 0] = 1 - ours['identity_rate'] / 100
        matrix1[i*2, 1] = ours['expression_rate'] / 100
        matrix1[i*2, 2] = ours['psnr'] / 25
        matrix1[i*2, 3] = ours['ssim']
        matrix1[i*2, 4] = 1 - ours['time'] / 0.05
        
        # Baseline
        matrix1[i*2+1, 0] = 1 - baseline['identity_rate'] / 100
        matrix1[i*2+1, 1] = baseline['expression_rate'] / 100
        matrix1[i*2+1, 2] = baseline['psnr'] / 25
        matrix1[i*2+1, 3] = baseline['ssim']
        matrix1[i*2+1, 4] = 1 - baseline['time'] / 0.05
    
    row_labels1 = []
    for d in datasets:
        row_labels1.append(f'{d}\n(Ours)')
        row_labels1.append(f'{d}\n(Baseline)')
    
    sns.heatmap(matrix1, annot=True, fmt='.2f', cmap='RdYlGn', 
                xticklabels=metrics, yticklabels=row_labels1,
                cbar_kws={'label': 'Normalized Score (0-1)'}, 
                ax=ax1, vmin=0, vmax=1, linewidths=0.5)
    ax1.set_title('Cross-Dataset Comparison', fontweight='bold', pad=15)
    
    # 热力图2：消融实验
    methods = ['Full Method', '3DDFA-V2', 'K-Same']
    matrix2 = np.zeros((len(methods), len(metrics)))
    
    for i, method in enumerate(methods):
        data = ablation_data[method]
        matrix2[i, 0] = 1 - data['identity_rate'] / 100
        matrix2[i, 1] = data['expression_rate'] / 100
        matrix2[i, 2] = data['psnr'] / 25
        matrix2[i, 3] = data['ssim']
        matrix2[i, 4] = 1 - data['time'] / 0.07
    
    sns.heatmap(matrix2, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=metrics, yticklabels=methods,
                cbar_kws={'label': 'Normalized Score (0-1)'},
                ax=ax2, vmin=0, vmax=1, linewidths=0.5)
    ax2.set_title('Ablation Study (Celeb-A)', fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig('results/heatmap_comparison.png', dpi=300, bbox_inches='tight', transparent=True)
    print("✓ 热力图已保存: results/heatmap_comparison.png")
    plt.show()


# ===================== 主函数 =====================

def main():
    import os
    os.makedirs('results', exist_ok=True)
    
    print("="*70)
    print("生成实验结果可视化图表")
    print("="*70)
    
    print("\n1. 生成跨数据集对比图（Ours vs Baseline）...")
    plot_cross_dataset_comparison()
    
    print("\n2. 生成消融实验对比图（Full Method vs 3DDFA-V2 vs K-Same）...")
    plot_ablation_study()
    
    print("\n3. 生成雷达图（综合性能对比）...")
    plot_radar_chart()
    
    print("\n4. 生成消融实验单独雷达图...")
    plot_ablation_radar()
    
    print("\n5. 生成热力图（归一化性能矩阵）...")
    plot_heatmap()
    
    print("\n" + "="*70)
    print("所有图表已生成完成！")
    print("保存位置：results/ 目录")
    print("格式：PNG (高清, 透明背景)")
    print("="*70)


if __name__ == "__main__":
    main()