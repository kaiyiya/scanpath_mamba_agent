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
from metrics.scanpath_metrics import compute_all_metrics, compute_all_metrics_extended, compute_dtw
import pickle


def visualize_multiple_scanpaths(image, pred_scanpaths, save_path, sample_idx, max_paths_to_plot=50):
    """
    可视化多条生成的扫描路径（类似ScanDMM的方式）
    
    Args:
        image: 输入图像
        pred_scanpaths: 多条预测路径 (N_paths, T, 2)
        save_path: 保存路径
        sample_idx: 样本索引
        max_paths_to_plot: 最多显示多少条路径（避免过于拥挤）
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # 转换图像格式
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # 确保图像格式是 (C, H, W)，然后转置为 (H, W, C)
    if image.ndim == 3:
        if image.shape[-1] not in [1, 3]:
            image = np.transpose(image, (1, 2, 0))
    
    # 如果图像是归一化到 [-1, 1]，转换到 [0, 1]
    if image.min() < 0:
        image = (image + 1.0) / 2.0
    image = np.clip(image, 0, 1)

    # 显示图像
    ax.imshow(image)
    
    # 限制显示的路径数量，避免过于拥挤
    n_paths_to_plot = min(len(pred_scanpaths), max_paths_to_plot)
    paths_to_plot = pred_scanpaths[:n_paths_to_plot]
    
    # 绘制多条路径，使用不同的颜色和透明度
    for i, pred_path in enumerate(paths_to_plot):
        # 使用渐变色和透明度，让多条路径看起来更清晰
        alpha = 0.4 / np.sqrt(n_paths_to_plot)  # 透明度随路径数量调整
        color = plt.cm.viridis(i / max(n_paths_to_plot, 1))
        
        ax.plot(pred_path[:, 0] * image.shape[1],
                pred_path[:, 1] * image.shape[0],
                '-', linewidth=1.5, alpha=alpha, color=color)
        
        # 标记起始点
        if i < 20:  # 只标记前20条的起始点，避免过于拥挤
            ax.plot(pred_path[0, 0] * image.shape[1],
                   pred_path[0, 1] * image.shape[0],
                   'r*', markersize=8, alpha=0.7)
    
    # 计算并显示所有路径的平均路径（用粗线标出）
    if len(pred_scanpaths) > 0:
        mean_path = np.mean(pred_scanpaths, axis=0)
        ax.plot(mean_path[:, 0] * image.shape[1],
                mean_path[:, 1] * image.shape[0],
                'r-', linewidth=3, alpha=0.9, label=f'Mean Path (n={len(pred_scanpaths)})')
        ax.plot(mean_path[0, 0] * image.shape[1],
                mean_path[0, 1] * image.shape[0],
                'r*', markersize=15, label='Mean Start')
    
    ax.set_title(f'Sample {sample_idx} - Generated Scanpaths ({len(pred_scanpaths)} paths, showing {n_paths_to_plot})')
    ax.legend()
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_test_results(num_scanpaths_per_image=200, max_paths_to_plot=50):
    """
    可视化所有测试集结果 - 每张图生成多条扫描路径
    
    Args:
        num_scanpaths_per_image: 每张图像生成的扫描路径数量（默认200，类似ScanDMM）
        max_paths_to_plot: 可视化时最多显示多少条路径（默认50，避免过于拥挤）
    """
    config = MambaAdaptiveConfig()

    # 创建输出目录
    output_dir = Path('./visualization_results_adaptive_multi')
    output_dir.mkdir(exist_ok=True)

    print("加载数据...")
    # 直接加载原始数据，以便访问所有扫描路径
    with open(config.processed_data_path, 'rb') as f:
        data_dict = pickle.load(f)
    test_data = data_dict['test']
    test_keys = list(test_data.keys())
    
    print(f"测试集共有 {len(test_keys)} 个图像")
    
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
    print(f"每张图像将生成 {num_scanpaths_per_image} 条扫描路径")

    # 遍历所有测试样本
    print(f"\n开始处理所有测试样本（共 {len(test_keys)} 个）...")

    # 用于累积所有样本的平均指标
    all_best_levs = []  # 每个样本的最佳LEV
    all_avg_levs = []   # 每个样本的平均LEV
    all_best_dtws = []  # 每个样本的最佳DTW
    all_avg_dtws = []   # 每个样本的平均DTW
    all_best_recs = []  # 每个样本的最佳REC
    all_avg_recs = []   # 每个样本的平均REC

    with torch.no_grad():
        for sample_idx, key in enumerate(test_keys):
            sample = test_data[key]
            
            print(f"\n处理样本 {sample_idx + 1}/{len(test_keys)}: {key}")
            
            # 加载图像
            img = sample['image']
            if isinstance(img, torch.Tensor):
                image_tensor = img.float().unsqueeze(0).to(config.device)
                image_np = img  # 保持为tensor
            else:
                image_tensor = torch.from_numpy(img).float().unsqueeze(0).to(config.device)
                image_np = torch.from_numpy(img).float()

            # 处理真实扫描路径数据（用于评估）
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

            # ========== 生成多条扫描路径（类似ScanDMM） ==========
            print(f"  生成 {num_scanpaths_per_image} 条扫描路径...")
            all_pred_scanpaths = []
            
            # 为了增加多样性，可以在每次生成时使用略微不同的温度
            for path_idx in range(num_scanpaths_per_image):
                # 可以使用略微不同的温度来增加多样性
                # temperature在1.0附近轻微波动，增加采样多样性
                temperature = 1.0 + np.random.uniform(-0.1, 0.1)  # 轻微变化
                
                result = model(image_tensor, gt_scanpaths=None, teacher_forcing_ratio=0.0, 
                              temperature=temperature, enable_early_stop=False)
                
                pred_scanpath = result[0]
                pred_scanpath_np = pred_scanpath[0].cpu().numpy()
                all_pred_scanpaths.append(pred_scanpath_np)
                
                # 进度提示
                if (path_idx + 1) % 50 == 0:
                    print(f"    已生成 {path_idx + 1}/{num_scanpaths_per_image} 条路径")
            
            all_pred_scanpaths = np.array(all_pred_scanpaths)  # (N_paths, T, 2)
            print(f"  完成生成，共 {len(all_pred_scanpaths)} 条路径")

            # ========== 计算评估指标（与所有真实路径比较） ==========
            # 对每条预测路径，找到最佳匹配的真实路径
            best_levs = []
            best_dtws = []
            best_recs = []
            
            avg_levs = []
            avg_dtws = []
            avg_recs = []

            for pred_path in all_pred_scanpaths:
                # 计算与所有真实路径的指标（使用扩展指标）
                metrics_list = []
                for true_sp in all_true_scanpaths:
                    # 获取显著性图（如果有）
                    saliency = sample.get('saliency_map')
                    if saliency is not None and isinstance(saliency, torch.Tensor):
                        saliency = saliency.squeeze().cpu().numpy()

                    metrics = compute_all_metrics_extended(pred_path, true_sp,
                                                          saliency_map=saliency,
                                                          image_size=config.image_size,
                                                          grid_size=32)
                    metrics_list.append(metrics)

                # 最佳匹配（最小DTW）
                best_idx = np.argmin([m['DTW'] for m in metrics_list])
                best_metrics = metrics_list[best_idx]
                best_levs.append(best_metrics['LEV'])
                best_dtws.append(best_metrics['DTW'])
                best_recs.append(best_metrics['REC'])

                # 平均指标
                avg_levs.append(np.mean([m['LEV'] for m in metrics_list]))
                avg_dtws.append(np.mean([m['DTW'] for m in metrics_list]))
                avg_recs.append(np.mean([m['REC'] for m in metrics_list]))

            # 计算该样本的平均指标（所有预测路径的平均值）
            sample_best_lev = np.mean(best_levs)
            sample_avg_lev = np.mean(avg_levs)
            sample_best_dtw = np.mean(best_dtws)
            sample_avg_dtw = np.mean(avg_dtws)
            sample_best_rec = np.mean(best_recs)
            sample_avg_rec = np.mean(avg_recs)

            # 计算新增指标（使用最佳匹配的真实路径）
            best_true_sp = all_true_scanpaths[best_idx]
            saliency = sample.get('saliency_map')
            if saliency is not None and isinstance(saliency, torch.Tensor):
                saliency = saliency.squeeze().cpu().numpy()

            extended_metrics = compute_all_metrics_extended(
                all_pred_scanpaths[0],  # 使用第一条预测路径
                best_true_sp,
                saliency_map=saliency,
                image_size=config.image_size
            )

            # 累积指标
            all_best_levs.append(sample_best_lev)
            all_avg_levs.append(sample_avg_lev)
            all_best_dtws.append(sample_best_dtw)
            all_avg_dtws.append(sample_avg_dtw)
            all_best_recs.append(sample_best_rec)
            all_avg_recs.append(sample_avg_rec)

            # ========== 可视化多条生成的路径 ==========
            save_path = output_dir / f'sample_{sample_idx:03d}_{key}.png'
            visualize_multiple_scanpaths(image_np, all_pred_scanpaths, save_path, 
                                        sample_idx, max_paths_to_plot=max_paths_to_plot)

            # 打印统计信息
            print(f"  评估结果（基于 {len(all_pred_scanpaths)} 条预测路径）:")
            print(f"    最佳匹配平均值:")
            print(f"      LEV = {sample_best_lev:.2f}, DTW = {sample_best_dtw:.4f}, REC = {sample_best_rec:.4f}")
            print(f"    与所有真实路径的平均值:")
            print(f"      LEV = {sample_avg_lev:.2f}, DTW = {sample_avg_dtw:.4f}, REC = {sample_avg_rec:.4f}")
            print(f"    新增指标（最佳匹配）:")
            print(f"      SIM = {extended_metrics.get('SIM', 0):.4f} (越小越好)")
            print(f"      MM_Vector = {extended_metrics.get('MM_Vector', 0):.4f} (方向相似度, 0-1)")
            print(f"      MM_Length = {extended_metrics.get('MM_Length', 0):.4f} (步长相似度, 0-1)")
            print(f"      MM_Position = {extended_metrics.get('MM_Position', 0):.4f} (位置相似度, 0-1)")
            if 'NSS' in extended_metrics:
                print(f"      NSS = {extended_metrics['NSS']:.4f} (显著性得分)")
                print(f"      CC = {extended_metrics['CC']:.4f} (相关系数, 0-1)")
                print(f"      SalCoverage = {extended_metrics['SalCoverage']:.4f} (显著性覆盖率, 0-1)")

            # 计算生成路径的统计信息
            all_paths_flat = all_pred_scanpaths.reshape(-1, 2)
            pred_mean = all_paths_flat.mean(axis=0)
            pred_std = all_paths_flat.std(axis=0)
            print(f"    生成路径统计:")
            print(f"      均值: x={pred_mean[0]:.4f}, y={pred_mean[1]:.4f}")
            print(f"      标准差: x={pred_std[0]:.4f}, y={pred_std[1]:.4f}")

    # 打印最终的平均指标
    print(f"\n{'='*60}")
    print(f"最终评估指标 (基于所有 {len(test_keys)} 个测试样本)")
    print(f"每张图像生成 {num_scanpaths_per_image} 条扫描路径")
    print(f"{'='*60}")
    print(f"\n【最佳匹配 (Best Match)】- 每条预测路径与最相似的真实路径比较:")
    print(f"  平均 LEV (Levenshtein Distance): {np.mean(all_best_levs):.2f} ± {np.std(all_best_levs):.2f} (越小越好)")
    print(f"  平均 DTW (Dynamic Time Warping): {np.mean(all_best_dtws):.4f} ± {np.std(all_best_dtws):.4f} (越小越好)")
    print(f"  平均 REC (Recurrence/IoU): {np.mean(all_best_recs):.4f} ± {np.std(all_best_recs):.4f} (越大越好)")
    print(f"\n【平均指标 (Average)】- 每条预测路径与所有真实路径的平均值:")
    print(f"  平均 LEV (Levenshtein Distance): {np.mean(all_avg_levs):.2f} ± {np.std(all_avg_levs):.2f} (越小越好)")
    print(f"  平均 DTW (Dynamic Time Warping): {np.mean(all_avg_dtws):.4f} ± {np.std(all_avg_dtws):.4f} (越小越好)")
    print(f"  平均 REC (Recurrence/IoU): {np.mean(all_avg_recs):.4f} ± {np.std(all_avg_recs):.4f} (越大越好)")
    print(f"{'='*60}")
    print(f"\n可视化结果保存在: {output_dir.absolute()}")


if __name__ == '__main__':
    # 每张图像生成200条扫描路径（类似ScanDMM），最多显示50条路径
    visualize_test_results(num_scanpaths_per_image=200, max_paths_to_plot=50)
