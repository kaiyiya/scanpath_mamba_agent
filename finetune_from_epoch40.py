"""
从第40轮检查点继续训练的脚本
应用所有修复：
1. Y均值偏差修复
2. 重构损失权重提高
3. Teacher Forcing放慢
4. 序列对齐损失（改善LEV）
5. 使用真实起始点
"""
import os
import torch
from train_mamba_adaptive import train
from config_mamba_adaptive import MambaAdaptiveConfig

def finetune_from_epoch_40():
    """从第40轮检查点继续训练"""

    # 检查检查点是否存在
    checkpoint_path = './checkpoints_adaptive/checkpoint_epoch_40.pth'

    if not os.path.exists(checkpoint_path):
        print(f"❌ 检查点不存在: {checkpoint_path}")
        print("可用的检查点:")
        checkpoint_dir = './checkpoints_adaptive'
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            for ckpt in sorted(checkpoints):
                print(f"  - {ckpt}")
        return

    print("="*80)
    print("从第40轮检查点继续训练（应用所有修复）")
    print("="*80)
    print(f"\n加载检查点: {checkpoint_path}")

    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    print(f"检查点信息:")
    print(f"  - 轮次: {checkpoint['epoch']}")
    if 'best_loss' in checkpoint:
        print(f"  - 最佳损失: {checkpoint['best_loss']:.4f}")
    if 'best_position_error' in checkpoint:
        print(f"  - 最佳位置误差: {checkpoint['best_position_error']:.4f}")

    print("\n应用的修复:")
    print("  ✅ Y均值偏差修复（惩罚偏离0.5的任何方向）")
    print("  ✅ 重构损失权重提高（1.0 → 2.0）")
    print("  ✅ KL散度权重降低（0.005 → 0.001）")
    print("  ✅ Teacher Forcing放慢（0.7→0.1/100轮 改为 0.7→0.2/150轮）")
    print("  ✅ 序列对齐损失（权重2.0-3.0，改善LEV指标）")
    print("  ✅ 使用真实起始点（use_gt_start=True）")
    print("  ✅ 前5步保持高Teacher Forcing（+0.3）")

    print("\n开始训练...")
    print("="*80)

    # 调用训练函数（会自动加载检查点）
    # 注意：需要修改train_mamba_adaptive.py以支持从指定检查点继续
    train()

if __name__ == '__main__':
    finetune_from_epoch_40()
