"""
快速测试：验证方案A的修改是否正确
"""
import torch
import numpy as np
from pathlib import Path
from config_mamba_adaptive import MambaAdaptiveConfig
from data.dataset import Salient360ScanpathDataset
from models.mamba_adaptive_scanpath import MambaAdaptiveScanpath


def test_plan_a():
    """测试方案A：绝对位置 + 低VAE随机性 + 强序列对齐"""
    config = MambaAdaptiveConfig()
    device = config.device

    print("=" * 70)
    print("方案A快速测试")
    print("=" * 70)

    # 加载数据
    test_dataset = Salient360ScanpathDataset(
        config.processed_data_path, 'test', config.seq_len, augment=False
    )
    sample = test_dataset[0]

    # 创建模型
    model = MambaAdaptiveScanpath(config).to(device)

    image = sample['image'].unsqueeze(0).to(device)
    gt_scanpath = sample['scanpath'].unsqueeze(0).to(device)

    print("\n1. 测试推理模式（应该完全确定性）")
    print("-" * 70)

    model.eval()
    with torch.no_grad():
        # 生成两次，应该完全相同
        pred1, _, _ = model(image, gt_scanpaths=None, teacher_forcing_ratio=0.0)
        pred2, _, _ = model(image, gt_scanpaths=None, teacher_forcing_ratio=0.0)

        diff = torch.abs(pred1 - pred2).max().item()
        print(f"  两次预测的最大差异: {diff:.6f}")

        if diff < 1e-5:
            print(f"  ✓ 推理完全确定性（temperature=0）")
        else:
            print(f"  ⚠️  推理有随机性，可能有问题")

    print("\n2. 测试训练模式（应该有轻微随机性）")
    print("-" * 70)

    model.train()
    with torch.no_grad():
        # 生成两次，应该略有不同（temperature=0.3）
        pred1, _, _ = model(image, gt_scanpaths=None, teacher_forcing_ratio=0.0)
        pred2, _, _ = model(image, gt_scanpaths=None, teacher_forcing_ratio=0.0)

        diff = torch.abs(pred1 - pred2).max().item()
        print(f"  两次预测的最大差异: {diff:.6f}")

        if 1e-5 < diff < 0.1:
            print(f"  ✓ 训练有适度随机性（temperature=0.3）")
        elif diff < 1e-5:
            print(f"  ⚠️  训练完全确定性，temperature可能未生效")
        else:
            print(f"  ⚠️  训练随机性过大")

    print("\n3. 测试位置范围（应该在[0,1]内）")
    print("-" * 70)

    model.eval()
    with torch.no_grad():
        pred, _, _ = model(image, gt_scanpaths=None, teacher_forcing_ratio=0.0)
        pred_np = pred[0].cpu().numpy()

        x_min, x_max = pred_np[:, 0].min(), pred_np[:, 0].max()
        y_min, y_max = pred_np[:, 1].min(), pred_np[:, 1].max()

        print(f"  X范围: [{x_min:.4f}, {x_max:.4f}]")
        print(f"  Y范围: [{y_min:.4f}, {y_max:.4f}]")

        if 0.0 <= x_min and x_max <= 1.0 and 0.0 <= y_min and y_max <= 1.0:
            print(f"  ✓ 位置在有效范围内")
        else:
            print(f"  ✗ 位置超出范围")

    print("\n4. 测试梯度反向传播")
    print("-" * 70)

    model.train()
    try:
        pred, mu, logvar = model(image, gt_scanpaths=gt_scanpath, teacher_forcing_ratio=0.5)
        loss = torch.nn.functional.mse_loss(pred, gt_scanpath)
        loss.backward()

        print(f"  ✓ 反向传播成功")
        print(f"  Loss: {loss.item():.4f}")

        # 检查梯度
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                has_nan_grad = True
                break

        if not has_nan_grad:
            print(f"  ✓ 所有梯度正常")
        else:
            print(f"  ✗ 存在NaN梯度")

    except Exception as e:
        print(f"  ✗ 反向传播失败: {e}")

    print("\n5. 测试position_decoder输出")
    print("-" * 70)

    model.eval()
    with torch.no_grad():
        # 检查position_decoder的最后一层
        last_layer = model.position_decoder[-1]
        print(f"  最后一层类型: {type(last_layer).__name__}")

        if isinstance(last_layer, torch.nn.Linear):
            print(f"  ✓ 使用Linear层（配合Sigmoid）")
        elif isinstance(last_layer, torch.nn.Tanh):
            print(f"  ✗ 仍然使用Tanh层（应该改为Linear）")
        else:
            print(f"  ⚠️  未知的最后一层类型")

    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
    print("\n如果所有测试通过，可以开始训练：")
    print("  python train_mamba_adaptive.py")
    print("\n预期改善：")
    print("  - LEV: 从58.4降低到<45")
    print("  - 训练稳定，无NaN")
    print("  - 序列对齐改善")


if __name__ == '__main__':
    test_plan_a()
