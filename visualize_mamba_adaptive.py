"""
Mamba-Adaptive模型可视化脚本
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path

from config_mamba_adaptive import MambaAdaptiveConfig
from data.dataset import create_dataloaders
from models.mamba_adaptive_scanpath import MambaAdaptiveScanpath
from metrics.scanpath_metrics import compute_all_metrics, compute_dtw
import pickle


def visualize_scanpath(image, true_scanpath, pred_scanpath, save_path, sample_idx):
    """可视化单个样本的扫描路径"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 转换图像格式
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # 确保图像格式是 (C, H, W)，然后转置为 (H, W, C)
    if image.ndim == 3:
        # 检查是否是 (H, W, C) 格式（最后一个维度是3或1）
        if image.shape[-1] in [1, 3]:
            # 已经是 (H, W, C) 格式，不需要转置
            pass
        else:
            # 是 (C, H, W) 格式，需要转置
            image = np.transpose(image, (1, 2, 0))
    
    # 如果图像是归一化到 [-1, 1]，转换到 [0, 1]
    if image.min() < 0:
        image = (image + 1.0) / 2.0
    image = np.clip(image, 0, 1)

    # 左图：真实路径
    axes[0].imshow(image)
    axes[0].plot(true_scanpath[:, 0] * image.shape[1],
                 true_scanpath[:, 1] * image.shape[0],
                 'g-', linewidth=2, alpha=0.6, label='True Path')
    axes[0].scatter(true_scanpath[:, 0] * image.shape[1],
                    true_scanpath[:, 1] * image.shape[0],
                    c=range(len(true_scanpath)), cmap='Greens', s=100, zorder=5)
    axes[0].plot(true_scanpath[0, 0] * image.shape[1],
                 true_scanpath[0, 1] * image.shape[0],
                 'r*', markersize=20, label='Start')
    axes[0].set_title(f'Sample {sample_idx} - Ground Truth')
    axes[0].legend()
    axes[0].axis('off')

    # 右图：预测路径
    axes[1].imshow(image)
    axes[1].plot(pred_scanpath[:, 0] * image.shape[1],
                 pred_scanpath[:, 1] * image.shape[0],
                 'b-', linewidth=2, alpha=0.6, label='Predicted Path')
    axes[1].scatter(pred_scanpath[:, 0] * image.shape[1],
                    pred_scanpath[:, 1] * image.shape[0],
                    c=range(len(pred_scanpath)), cmap='Blues', s=100, zorder=5)
    axes[1].plot(pred_scanpath[0, 0] * image.shape[1],
                 pred_scanpath[0, 1] * image.shape[0],
                 'r*', markersize=20, label='Start')
    axes[1].set_title(f'Sample {sample_idx} - Prediction (Mamba-Adaptive)')
    axes[1].legend()
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_test_results(num_samples=10):
    """可视化测试集结果 - 显示不同的图像样本"""
    config = MambaAdaptiveConfig()

    # 创建输出目录
    output_dir = Path('./visualization_results_adaptive')
    output_dir.mkdir(exist_ok=True)

    print("加载数据...")
    # 直接加载原始数据，以便访问所有扫描路径
    with open(config.processed_data_path, 'rb') as f:
        data_dict = pickle.load(f)
    test_data = data_dict['test']
    test_keys = list(test_data.keys())
    
    # 创建测试数据集，用于获取预处理后的数据
    from data.dataset import Salient360ScanpathDataset
    test_dataset = Salient360ScanpathDataset(config.processed_data_path, 'test', config.seq_len)

    print("加载最佳模型...")
    model = MambaAdaptiveScanpath(config).to(config.device)

    checkpoint_path = os.path.join(config.checkpoint_dir, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f"错误：找不到最佳模型 {checkpoint_path}")
        print("请先训练模型")
        return

    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"成功加载模型 (Epoch {checkpoint['epoch']})")
    print(f"最佳损失: {checkpoint.get('best_loss', 'N/A')}")

    # 可视化样本
    print(f"\n开始可视化 {num_samples} 个不同的测试样本...")

    total_error_best = 0  # 与最佳匹配的误差
    total_error_avg = 0   # 与所有路径的平均误差
    total_lev_best = 0    # 最佳匹配的LEV
    total_lev_avg = 0     # 平均LEV
    total_dtw_best = 0    # 最佳匹配的DTW
    total_dtw_avg = 0     # 平均DTW
    total_rec_best = 0    # 最佳匹配的REC
    total_rec_avg = 0     # 平均REC

    # 确保选择不同的样本（均匀分布）
    sample_indices = np.linspace(0, len(test_keys)-1, num_samples, dtype=int)

    with torch.no_grad():
        for sample_count, idx in enumerate(sample_indices):
            key = test_keys[idx]
            sample = test_data[key]
            
            # 加载图像
            img = sample['image']
            if isinstance(img, torch.Tensor):
                image_tensor = img.float().unsqueeze(0).to(config.device)
                # 保存原始tensor用于可视化函数（函数内部会处理格式转换）
                image_np = img  # 保持为tensor，让visualize_scanpath函数处理
            else:
                image_tensor = torch.from_numpy(img).float().unsqueeze(0).to(config.device)
                image_np = torch.from_numpy(img).float()  # 转换为tensor，保持(C, H, W)格式

            # 处理扫描路径数据
            scanpaths_2d = sample['scanpaths_2d']
            if scanpaths_2d.dtype == np.object_:
                scanpaths_list = []
                for i in range(len(scanpaths_2d)):
                    sp = np.array(scanpaths_2d[i], dtype=np.float32)
                    if sp.shape[1] == 3:
                        sp = sp[:, :2]
                    scanpaths_list.append(sp)
                all_true_scanpaths = np.array(scanpaths_list)
            else:
                if scanpaths_2d.shape[2] == 3:
                    all_true_scanpaths = scanpaths_2d[:, :, :2]
                else:
                    all_true_scanpaths = scanpaths_2d.copy()
            
            # 归一化所有真实路径（如果需要）
            normalized_paths = []
            for sp in all_true_scanpaths:
                sp = sp.copy()
                if sp.max() > 1.0 or sp.min() < 0.0:
                    if sp[:, 0].min() < -100:
                        sp[:, 0] = (sp[:, 0] + 180.0) / 360.0
                        sp[:, 1] = (sp[:, 1] + 90.0) / 180.0
                    else:
                        sp_min = sp.min(axis=0, keepdims=True)
                        sp_max = sp.max(axis=0, keepdims=True)
                        sp_range = sp_max - sp_min
                        sp_range[sp_range < 1e-8] = 1.0
                        sp = (sp - sp_min) / sp_range
                # 裁剪或填充到seq_len
                if sp.shape[0] > config.seq_len:
                    sp = sp[:config.seq_len]
                elif sp.shape[0] < config.seq_len:
                    pad_length = config.seq_len - sp.shape[0]
                    sp = np.concatenate([sp, sp[-1:].repeat(pad_length, axis=0)], axis=0)
                sp = np.clip(sp, 0.0, 1.0)
                normalized_paths.append(sp)
            all_true_scanpaths = np.array(normalized_paths)

            # 推理模式预测 - 不使用Teacher Forcing
            # 显式设置enable_early_stop=False，确保返回3个值
            result = model(image_tensor, gt_scanpaths=None, teacher_forcing_ratio=0.0, enable_early_stop=False)
            # 安全解包：无论返回3个还是5个值，都只取前3个
            pred_scanpath = result[0]
            _ = result[1]  # mus (不需要使用)
            _ = result[2]  # logvars (不需要使用)
            pred_scanpath_np = pred_scanpath[0].cpu().numpy()

            # 与所有真实路径计算距离，选择最佳匹配
            num_paths = len(all_true_scanpaths)
            dtw_scores = []
            for true_sp in all_true_scanpaths:
                dtw_val = compute_dtw(pred_scanpath_np, true_sp, image_size=config.image_size)
                dtw_scores.append(dtw_val)
            
            best_idx = np.argmin(dtw_scores)
            best_true_scanpath = all_true_scanpaths[best_idx]
            
            # 计算最佳匹配的指标
            metrics_best = compute_all_metrics(pred_scanpath_np, best_true_scanpath, 
                                               image_size=config.image_size,
                                               grid_size=32)
            error_best = np.linalg.norm(pred_scanpath_np - best_true_scanpath, axis=1).mean()
            
            # 计算平均指标（与所有路径的平均值）
            metrics_all = []
            for true_sp in all_true_scanpaths:
                metrics_all.append(compute_all_metrics(pred_scanpath_np, true_sp, 
                                                      image_size=config.image_size,
                                                      grid_size=32))
            metrics_avg = {
                'LEV': np.mean([m['LEV'] for m in metrics_all]),
                'DTW': np.mean([m['DTW'] for m in metrics_all]),
                'REC': np.mean([m['REC'] for m in metrics_all])
            }
            error_avg = np.mean([np.linalg.norm(pred_scanpath_np - true_sp, axis=1).mean() 
                                for true_sp in all_true_scanpaths])

            # 累积指标
            total_error_best += error_best
            total_error_avg += error_avg
            total_lev_best += metrics_best['LEV']
            total_lev_avg += metrics_avg['LEV']
            total_dtw_best += metrics_best['DTW']
            total_dtw_avg += metrics_avg['DTW']
            total_rec_best += metrics_best['REC']
            total_rec_avg += metrics_avg['REC']

            # 可视化（使用最佳匹配的真实路径）
            save_path = output_dir / f'sample_{sample_count:03d}_idx{idx}.png'
            visualize_scanpath(image_np, best_true_scanpath, pred_scanpath_np, save_path, sample_count)

            print(f"样本 {sample_count} (数据集索引 {idx}, 共 {num_paths} 条真实路径):")
            print(f"  最佳匹配 (Best Match):")
            print(f"    位置误差 = {error_best:.4f}")
            print(f"    LEV = {metrics_best['LEV']:.2f}, DTW = {metrics_best['DTW']:.4f}, REC = {metrics_best['REC']:.4f}")
            print(f"  平均指标 (Average over {num_paths} paths):")
            print(f"    位置误差 = {error_avg:.4f}")
            print(f"    LEV = {metrics_avg['LEV']:.2f}, DTW = {metrics_avg['DTW']:.4f}, REC = {metrics_avg['REC']:.4f}")

            # 打印预测统计
            pred_mean = pred_scanpath_np.mean(axis=0)
            pred_std = pred_scanpath_np.std(axis=0)
            print(f"  预测路径均值: x={pred_mean[0]:.4f}, y={pred_mean[1]:.4f}")
            print(f"  预测路径标准差: x={pred_std[0]:.6f}, y={pred_std[1]:.6f}")

    # 打印平均指标
    print(f"\n{'='*60}")
    print(f"平均评估指标 (基于 {num_samples} 个样本):")
    print(f"{'='*60}")
    print(f"\n【最佳匹配 (Best Match)】- 与最相似的真实路径比较:")
    print(f"  位置误差 (Position Error): {total_error_best / num_samples:.4f}")
    print(f"  LEV (Levenshtein Distance): {total_lev_best / num_samples:.2f} (越小越好)")
    print(f"  DTW (Dynamic Time Warping): {total_dtw_best / num_samples:.4f} (越小越好)")
    print(f"  REC (Recurrence/IoU): {total_rec_best / num_samples:.4f} (越大越好)")
    print(f"\n【平均指标 (Average)】- 与所有真实路径的平均值:")
    print(f"  位置误差 (Position Error): {total_error_avg / num_samples:.4f}")
    print(f"  LEV (Levenshtein Distance): {total_lev_avg / num_samples:.2f} (越小越好)")
    print(f"  DTW (Dynamic Time Warping): {total_dtw_avg / num_samples:.4f} (越小越好)")
    print(f"  REC (Recurrence/IoU): {total_rec_avg / num_samples:.4f} (越大越好)")
    print(f"{'='*60}")
    print(f"\n可视化结果保存在: {output_dir.absolute()}")


if __name__ == '__main__':
    visualize_test_results(num_samples=10)
