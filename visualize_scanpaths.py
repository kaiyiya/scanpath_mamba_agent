"""
可视化预测路径和真实路径的对比
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config_mamba_adaptive import MambaAdaptiveConfig
from data.dataset import Salient360ScanpathDataset
from models.mamba_adaptive_scanpath import MambaAdaptiveScanpath


def visualize_scanpaths(num_samples=5):
    """可视化预测路径和真实路径"""
    config = MambaAdaptiveConfig()
    device = config.device

    # 加载测试数据
    print("加载测试数据...")
    test_dataset = Salient360ScanpathDataset(
        config.processed_data_path,
        'test',
        config.seq_len,
        augment=False
    )

    # 加载模型
    print("加载模型...")
    model = MambaAdaptiveScanpath(config).to(device)
    checkpoint_path = Path(config.checkpoint_dir) / 'best_model.pth'

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"成功加载模型")
    else:
        print("未找到模型checkpoint!")
        return

    model.eval()

    # 创建输出目录
    output_dir = Path('logs') / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 可视化前几个样本
    for idx in range(min(num_samples, len(test_dataset))):
        sample = test_dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)
        gt_path = sample['scanpath'].numpy()
        key = sample['key']

        # 预测
        with torch.no_grad():
            pred_scanpath, _, _ = model(
                image,
                gt_scanpaths=None,
                teacher_forcing_ratio=0.0,
                enable_early_stop=False,
                use_gt_start=False
            )

        pred_path = pred_scanpath[0].cpu().numpy()
        pred_path = np.clip(pred_path, 0.0, 1.0)

        # 转换为像素坐标
        h, w = config.image_size
        pred_pixels = pred_path.copy()
        pred_pixels[:, 0] = pred_pixels[:, 0] * w
        pred_pixels[:, 1] = pred_pixels[:, 1] * h

        gt_pixels = gt_path.copy()
        gt_pixels[:, 0] = gt_pixels[:, 0] * w
        gt_pixels[:, 1] = gt_pixels[:, 1] * h

        # 创建图像
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # 左图：归一化坐标空间
        ax1 = axes[0]
        ax1.plot(gt_path[:, 0], gt_path[:, 1], 'b-o', label='真实路径', markersize=4, linewidth=2)
        ax1.plot(pred_path[:, 0], pred_path[:, 1], 'r-s', label='预测路径', markersize=4, linewidth=2, alpha=0.7)
        ax1.scatter(gt_path[0, 0], gt_path[0, 1], c='green', s=100, marker='*', label='起始点', zorder=5)
        ax1.scatter(pred_path[0, 0], pred_path[0, 1], c='orange', s=100, marker='*', label='预测起始点', zorder=5)
        ax1.set_xlabel('X坐标 (归一化)', fontsize=12)
        ax1.set_ylabel('Y坐标 (归一化)', fontsize=12)
        ax1.set_title(f'样本 {idx+1}: {key}\n归一化坐标空间', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        ax1.set_aspect('equal')

        # 右图：像素坐标空间
        ax2 = axes[1]
        ax2.plot(gt_pixels[:, 0], gt_pixels[:, 1], 'b-o', label='真实路径', markersize=4, linewidth=2)
        ax2.plot(pred_pixels[:, 0], pred_pixels[:, 1], 'r-s', label='预测路径', markersize=4, linewidth=2, alpha=0.7)
        ax2.scatter(gt_pixels[0, 0], gt_pixels[0, 1], c='green', s=100, marker='*', label='起始点', zorder=5)
        ax2.scatter(pred_pixels[0, 0], pred_pixels[0, 1], c='orange', s=100, marker='*', label='预测起始点', zorder=5)
        
        # 添加12像素阈值圆圈（REC阈值）
        for i in range(len(gt_pixels)):
            circle = plt.Circle((gt_pixels[i, 0], gt_pixels[i, 1]), 12, color='blue', fill=False, linestyle='--', alpha=0.3)
            ax2.add_patch(circle)
        
        ax2.set_xlabel('X坐标 (像素)', fontsize=12)
        ax2.set_ylabel('Y坐标 (像素)', fontsize=12)
        ax2.set_title(f'像素坐标空间 (REC阈值=12像素)', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, w])
        ax2.set_ylim([0, h])
        ax2.set_aspect('equal')

        plt.tight_layout()
        
        # 保存图像
        output_path = output_dir / f'scanpath_comparison_{idx+1}_{key}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"保存可视化结果: {output_path}")
        plt.close()

        # 计算距离统计
        distances = np.sqrt(np.sum((pred_pixels - gt_pixels) ** 2, axis=1))
        print(f"\n样本 {idx+1} ({key}):")
        print(f"  平均距离: {distances.mean():.2f} 像素")
        print(f"  最小距离: {distances.min():.2f} 像素")
        print(f"  最大距离: {distances.max():.2f} 像素")
        print(f"  距离<12像素的点数: {(distances < 12).sum()}/{len(distances)}")

    print(f"\n所有可视化结果已保存到: {output_dir.absolute()}")


if __name__ == '__main__':
    visualize_scanpaths(num_samples=5)
