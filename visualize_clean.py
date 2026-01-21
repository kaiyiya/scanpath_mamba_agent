"""
清晰的可视化脚本：每张图片显示一条路径，排列成网格
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config_mamba_adaptive import MambaAdaptiveConfig
from data.dataset import Salient360ScanpathDataset
from models.mamba_adaptive_scanpath import MambaAdaptiveScanpath
from metrics.scanpath_metrics import compute_all_metrics_extended


def visualize_clean_grid(num_samples=None, num_paths_per_sample=20):
    """
    生成清晰的网格可视化

    Args:
        num_samples: 评估多少个测试样本（None表示所有样本）
        num_paths_per_sample: 每个样本生成多少条路径
    """
    config = MambaAdaptiveConfig()
    device = config.device
    output_dir = Path('logs/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据和模型
    print("加载测试数据...")
    test_dataset = Salient360ScanpathDataset(
        config.processed_data_path, 'test', config.seq_len, augment=False
    )

    print("加载模型...")
    model = MambaAdaptiveScanpath(config).to(device)
    checkpoint_path = Path(config.checkpoint_dir) / 'best_model.pth'

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"成功加载模型 (Epoch {checkpoint['epoch']})\n")
    else:
        print("未找到模型!")
        return

    model.eval()

    # 获取所有唯一的图像key
    unique_keys = []
    seen_keys = set()
    for sample in test_dataset.samples:
        key = sample['key']
        if key not in seen_keys:
            unique_keys.append(key)
            seen_keys.add(key)

    # 如果指定了num_samples，则只处理前num_samples个
    if num_samples is not None:
        unique_keys = unique_keys[:num_samples]

    total_samples = len(unique_keys)
    print(f"\n将为 {total_samples} 个测试样本生成可视化\n")

    for sample_idx, key in enumerate(unique_keys):
        print(f"处理样本 {sample_idx + 1}/{total_samples}: {key}")

        # 获取该图像的样本
        image_samples = [s for s in test_dataset.samples if s['key'] == key]
        first_sample_idx = test_dataset.samples.index(image_samples[0])
        sample = test_dataset[first_sample_idx]

        image = sample['image'].unsqueeze(0).to(device)
        image_np = sample['image'].permute(1, 2, 0).cpu().numpy()

        # 生成多条路径
        pred_paths = []
        print(f"  生成 {num_paths_per_sample} 条路径...", end='', flush=True)
        with torch.no_grad():
            for _ in range(num_paths_per_sample):
                pred_scanpath, _, _ = model(image, gt_scanpaths=None,
                                            teacher_forcing_ratio=0.0,
                                            enable_early_stop=False,
                                            use_gt_start=False)
                pred_paths.append(pred_scanpath[0].cpu().numpy())
        print(" 完成")

        # 创建网格布局：4行5列
        print(f"  绘制可视化...", end='', flush=True)
        rows, cols = 4, 5
        fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
        fig.suptitle(f'Sample {sample_idx + 1}/{total_samples}: {key} - {num_paths_per_sample} Generated Paths',
                     fontsize=16, y=0.995)

        for path_idx in range(num_paths_per_sample):
            row = path_idx // cols
            col = path_idx % cols
            ax = axes[row, col]

            # 显示图像
            ax.imshow(image_np)

            # 绘制路径
            pred_path = pred_paths[path_idx]
            h, w = config.image_size

            # 转换为像素坐标
            pred_pixels = pred_path.copy()
            pred_pixels[:, 0] *= w
            pred_pixels[:, 1] *= h

            # 绘制路径线
            ax.plot(pred_pixels[:, 0], pred_pixels[:, 1],
                    'r-', linewidth=2, alpha=0.7, label='Predicted')

            # 绘制注视点
            ax.scatter(pred_pixels[:, 0], pred_pixels[:, 1],
                       c=range(len(pred_pixels)), cmap='hot',
                       s=100, alpha=0.8, edgecolors='white', linewidths=1.5)

            # 标记起点和终点
            ax.scatter(pred_pixels[0, 0], pred_pixels[0, 1],
                       c='green', s=200, marker='o',
                       edgecolors='white', linewidths=2, label='Start', zorder=10)
            ax.scatter(pred_pixels[-1, 0], pred_pixels[-1, 1],
                       c='blue', s=200, marker='s',
                       edgecolors='white', linewidths=2, label='End', zorder=10)

            ax.set_title(f'Path {path_idx + 1}', fontsize=10)
            ax.axis('off')

            # 只在第一个子图显示图例
            if path_idx == 0:
                ax.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        save_path = output_dir / f'sample_{sample_idx:03d}_{key}_grid.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f" 完成")
        print(f"  保存到: {save_path.name}")

        # 计算指标（使用第一条路径）
        pred_path = pred_paths[0]
        all_gt_paths = [s['scanpath'] for s in image_samples]

        # 获取显著性图
        saliency = sample.get('saliency_map')
        if saliency is not None and isinstance(saliency, torch.Tensor):
            saliency = saliency.squeeze().cpu().numpy()

        # 计算与最佳GT的指标
        metrics_list = []
        for gt_path in all_gt_paths:
            metrics = compute_all_metrics_extended(
                pred_path, gt_path,
                saliency_map=saliency,
                image_size=config.image_size
            )
            metrics_list.append(metrics)

        best_idx = np.argmin([m['DTW'] for m in metrics_list])
        best_metrics = metrics_list[best_idx]

        print(f"  指标: LEV={best_metrics['LEV']:.1f}, SIM={best_metrics['SIM']:.3f}, "
              f"MM_Pos={best_metrics['MM_Position']:.3f}, "
              f"SalCov={best_metrics.get('SalCoverage', 0):.2f}")

    print(f"\n所有可视化已保存到: {output_dir.absolute()}")
    print(f"共生成 {total_samples} 张可视化图片")


if __name__ == '__main__':
    # 默认生成所有测试样本的可视化
    # 如果只想生成部分样本，可以设置 num_samples=5
    visualize_clean_grid(num_samples=None, num_paths_per_sample=20)
