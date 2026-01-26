"""
测试方案1：验证位置累积和序列连续性
"""
import torch
import numpy as np
from pathlib import Path
from config_mamba_adaptive import MambaAdaptiveConfig
from data.dataset import Salient360ScanpathDataset
from models.mamba_adaptive_scanpath import MambaAdaptiveScanpath


def test_position_accumulation():
    """测试位置累积和序列连续性"""
    config = MambaAdaptiveConfig()
    device = config.device

    print("=" * 70)
    print("测试方案1：位置累积和序列连续性")
    print("=" * 70)

    # 加载数据
    test_dataset = Salient360ScanpathDataset(
        config.processed_data_path, 'test', config.seq_len, augment=False
    )
    sample = test_dataset[0]

    # 创建模型（随机初始化）
    model = MambaAdaptiveScanpath(config).to(device)
    model.eval()

    image = sample['image'].unsqueeze(0).to(device)
    gt_scanpath = sample['scanpath'].unsqueeze(0).to(device)

    print("\n1. 测试位置范围（应该允许轻微越界）")
    print("-" * 70)

    with torch.no_grad():
        # 生成路径
        pred_scanpath, _, _ = model(
            image, gt_scanpaths=None,
            teacher_forcing_ratio=0.0,
            enable_early_stop=False,
            use_gt_start=False
        )
        pred = pred_scanpath[0].cpu().numpy()

        # 检查位置范围
        x_min, x_max = pred[:, 0].min(), pred[:, 0].max()
        y_min, y_max = pred[:, 1].min(), pred[:, 1].max()

        print(f"  预测路径范围:")
        print(f"    X: [{x_min:.4f}, {x_max:.4f}]")
        print(f"    Y: [{y_min:.4f}, {y_max:.4f}]")

        # 检查是否有NaN
        has_nan = np.isnan(pred).any()
        if has_nan:
            print(f"  ✗ 包含NaN!")
        else:
            print(f"  ✓ 无NaN")

        # 位置应该主要在[0,1]内，但允许轻微越界
        if x_min >= -0.1 and x_max <= 1.1 and y_min >= -0.1 and y_max <= 1.1:
            print(f"  ✓ 位置在合理范围内（允许轻微越界）")
        else:
            print(f"  ⚠️  位置越界过多")

    print("\n2. 测试序列连续性（步长分布）")
    print("-" * 70)

    with torch.no_grad():
        # 生成多条路径
        all_steps = []
        for _ in range(10):
            pred_scanpath, _, _ = model(
                image, gt_scanpaths=None,
                teacher_forcing_ratio=0.0,
                enable_early_stop=False,
                use_gt_start=False
            )
            pred = pred_scanpath[0].cpu().numpy()
            steps = np.linalg.norm(pred[1:] - pred[:-1], axis=1)
            all_steps.extend(steps)

        all_steps = np.array(all_steps)
        gt = sample['scanpath'].numpy()
        gt_steps = np.linalg.norm(gt[1:] - gt[:-1], axis=1)

        print(f"  预测步长统计:")
        print(f"    均值: {np.mean(all_steps):.4f}")
        print(f"    标准差: {np.std(all_steps):.4f}")
        print(f"    最大值: {np.max(all_steps):.4f}")
        print(f"    最小值: {np.min(all_steps):.4f}")

        print(f"\n  GT步长统计:")
        print(f"    均值: {np.mean(gt_steps):.4f}")
        print(f"    标准差: {np.std(gt_steps):.4f}")
        print(f"    最大值: {np.max(gt_steps):.4f}")

        ratio = np.mean(all_steps) / np.mean(gt_steps)
        print(f"\n  预测/GT步长比率: {ratio:.2f}")
        if 0.5 < ratio < 2.0:
            print(f"  ✓ 步长大小与GT接近")
        else:
            print(f"  ⚠️  步长大小与GT差异较大")

    print("\n3. 测试Teacher Forcing（应该能更好地复制GT）")
    print("-" * 70)

    with torch.no_grad():
        # Teacher Forcing=1.0
        pred_scanpath_tf, _, _ = model(
            image, gt_scanpaths=gt_scanpath,
            teacher_forcing_ratio=1.0,
            enable_early_stop=False,
            use_gt_start=True
        )
        pred_tf = pred_scanpath_tf[0].cpu().numpy()
        gt = sample['scanpath'].numpy()

        distance = np.mean(np.linalg.norm(pred_tf - gt, axis=1))
        print(f"  Teacher Forcing=1.0时与GT的平均距离: {distance:.4f}")

        if distance < 0.10:
            print(f"  ✓ 相对位移+宽松裁剪显著改善了序列复制能力")
        elif distance < 0.15:
            print(f"  ✓ 有改善，但仍有提升空间")
        else:
            print(f"  ⚠️  距离仍然较大")

    print("\n4. 测试训练模式（检查梯度）")
    print("-" * 70)

    model.train()

    try:
        # 模拟训练步骤
        pred_scanpath, mu, logvar = model(
            image, gt_scanpaths=gt_scanpath,
            teacher_forcing_ratio=0.5,
            enable_early_stop=False,
            use_gt_start=True
        )

        # 计算简单损失
        loss = torch.nn.functional.mse_loss(pred_scanpath, gt_scanpath)

        # 反向传播
        loss.backward()

        print(f"  ✓ 反向传播成功")
        print(f"  Loss: {loss.item():.4f}")

        # 检查是否有NaN梯度
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                has_nan_grad = True
                print(f"  ✗ {name} 有NaN梯度")
                break

        if not has_nan_grad:
            print(f"  ✓ 所有梯度正常")

    except Exception as e:
        print(f"  ✗ 训练失败: {e}")

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
    print("\n建议:")
    print("1. 如果Teacher Forcing距离<0.10，说明改进有效")
    print("2. 现在可以重新训练，观察LEV指标是否改善")
    print("3. 目标：LEV从50.6降到<40")


if __name__ == '__main__':
    test_position_accumulation()
