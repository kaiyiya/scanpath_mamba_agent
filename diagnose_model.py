"""
诊断脚本：检查数据加载和模型预测是否正常
"""
import torch
import numpy as np
from pathlib import Path
from config_mamba_adaptive import MambaAdaptiveConfig
from data.dataset import Salient360ScanpathDataset
from models.mamba_adaptive_scanpath import MambaAdaptiveScanpath


def diagnose_data_and_model():
    """诊断数据和模型"""
    config = MambaAdaptiveConfig()
    device = config.device

    # 加载数据
    print("=" * 70)
    print("1. 检查数据加载")
    print("=" * 70)

    test_dataset = Salient360ScanpathDataset(
        config.processed_data_path, 'test', config.seq_len, augment=False
    )

    sample = test_dataset[0]
    print(f"样本key: {sample['key']}")
    print(f"GT路径形状: {sample['scanpath'].shape}")
    print(f"GT路径范围: x=[{sample['scanpath'][:, 0].min():.3f}, {sample['scanpath'][:, 0].max():.3f}], "
          f"y=[{sample['scanpath'][:, 1].min():.3f}, {sample['scanpath'][:, 1].max():.3f}]")
    print(f"GT路径前5个点:\n{sample['scanpath'][:5]}")

    # 加载模型
    print("\n" + "=" * 70)
    print("2. 检查模型预测")
    print("=" * 70)

    model = MambaAdaptiveScanpath(config).to(device)
    checkpoint_path = Path(config.checkpoint_dir) / 'best_model.pth'

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载模型: Epoch {checkpoint['epoch']}")
    else:
        print("❌ 未找到模型!")
        return

    model.eval()

    # 预测
    image = sample['image'].unsqueeze(0).to(device)
    gt_scanpath = sample['scanpath'].unsqueeze(0).to(device)

    with torch.no_grad():
        # 测试1: 使用GT起始点
        pred_scanpath1, _, _ = model(
            image, gt_scanpaths=None,
            teacher_forcing_ratio=0.0,
            enable_early_stop=False,
            use_gt_start=True
        )
        pred1 = pred_scanpath1[0].cpu().numpy()

        # 测试2: 不使用GT起始点
        pred_scanpath2, _, _ = model(
            image, gt_scanpaths=None,
            teacher_forcing_ratio=0.0,
            enable_early_stop=False,
            use_gt_start=False
        )
        pred2 = pred_scanpath2[0].cpu().numpy()

        # 测试3: 使用Teacher Forcing（应该完美复制GT）
        pred_scanpath3, _, _ = model(
            image, gt_scanpaths=gt_scanpath,
            teacher_forcing_ratio=1.0,
            enable_early_stop=False,
            use_gt_start=True
        )
        pred3 = pred_scanpath3[0].cpu().numpy()

    gt = sample['scanpath'].numpy()

    print("\n【测试1: 使用GT起始点】")
    print(f"预测路径范围: x=[{pred1[:, 0].min():.3f}, {pred1[:, 0].max():.3f}], "
          f"y=[{pred1[:, 1].min():.3f}, {pred1[:, 1].max():.3f}]")
    print(f"预测路径前5个点:\n{pred1[:5]}")
    print(f"与GT的平均距离: {np.mean(np.linalg.norm(pred1 - gt, axis=1)):.4f}")

    print("\n【测试2: 不使用GT起始点】")
    print(f"预测路径范围: x=[{pred2[:, 0].min():.3f}, {pred2[:, 0].max():.3f}], "
          f"y=[{pred2[:, 1].min():.3f}, {pred2[:, 1].max():.3f}]")
    print(f"预测路径前5个点:\n{pred2[:5]}")
    print(f"与GT的平均距离: {np.mean(np.linalg.norm(pred2 - gt, axis=1)):.4f}")

    print("\n【测试3: Teacher Forcing=1.0（应该完美复制GT）】")
    print(f"预测路径范围: x=[{pred3[:, 0].min():.3f}, {pred3[:, 0].max():.3f}], "
          f"y=[{pred3[:, 1].min():.3f}, {pred3[:, 1].max():.3f}]")
    print(f"预测路径前5个点:\n{pred3[:5]}")
    print(f"与GT的平均距离: {np.mean(np.linalg.norm(pred3 - gt, axis=1)):.4f}")

    if np.mean(np.linalg.norm(pred3 - gt, axis=1)) > 0.1:
        print("\n⚠️  警告: Teacher Forcing=1.0时仍然无法复制GT!")
        print("   这说明模型架构或训练有严重问题!")

    # 检查路径多样性
    print("\n" + "=" * 70)
    print("3. 检查路径多样性")
    print("=" * 70)

    pred_std_x = np.std(pred1[:, 0])
    pred_std_y = np.std(pred1[:, 1])
    gt_std_x = np.std(gt[:, 0])
    gt_std_y = np.std(gt[:, 1])

    print(f"预测路径标准差: x={pred_std_x:.4f}, y={pred_std_y:.4f}")
    print(f"GT路径标准差:   x={gt_std_x:.4f}, y={gt_std_y:.4f}")

    if pred_std_x < 0.05 or pred_std_y < 0.05:
        print("\n⚠️  警告: 预测路径标准差过小，可能出现'卡住'现象!")

    # 检查步长
    print("\n" + "=" * 70)
    print("4. 检查步长分布")
    print("=" * 70)

    pred_steps = np.linalg.norm(pred1[1:] - pred1[:-1], axis=1)
    gt_steps = np.linalg.norm(gt[1:] - gt[:-1], axis=1)

    print(f"预测步长: mean={np.mean(pred_steps):.4f}, std={np.std(pred_steps):.4f}, "
          f"min={np.min(pred_steps):.4f}, max={np.max(pred_steps):.4f}")
    print(f"GT步长:   mean={np.mean(gt_steps):.4f}, std={np.std(gt_steps):.4f}, "
          f"min={np.min(gt_steps):.4f}, max={np.max(gt_steps):.4f}")

    if np.mean(pred_steps) < 0.01:
        print("\n⚠️  警告: 预测步长过小，路径几乎不移动!")

    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)


if __name__ == '__main__':
    diagnose_data_and_model()
