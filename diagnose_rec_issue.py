"""
诊断REC为0的问题
检查预测路径和真实路径的坐标分布
"""
import torch
import numpy as np
import json
from pathlib import Path
from config_mamba_adaptive import MambaAdaptiveConfig
from data.dataset import Salient360ScanpathDataset
from models.mamba_adaptive_scanpath import MambaAdaptiveScanpath
# 直接实现REC计算，避免依赖问题
def compute_rec(pred_scanpath, true_scanpath, threshold=12.0, image_size=(256, 512)):
    """计算REC指标"""
    h, w = image_size
    
    # 转换为像素坐标
    P = pred_scanpath.copy()
    P[:, 0] = P[:, 0] * w
    P[:, 1] = P[:, 1] * h
    P = P.astype(np.float32)
    
    Q = true_scanpath.copy()
    Q[:, 0] = Q[:, 0] * w
    Q[:, 1] = Q[:, 1] * h
    Q = Q.astype(np.float32)
    
    # 截断到相同长度
    min_len = min(P.shape[0], Q.shape[0])
    P = P[:min_len, :2]
    Q = Q[:min_len, :2]
    
    # 计算交叉重现矩阵
    def euclidean(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))
    
    c = np.zeros((min_len, min_len))
    for i in range(min_len):
        for j in range(min_len):
            if euclidean(P[i], Q[j]) < threshold:
                c[i, j] = 1
    
    R = np.triu(c, 1).sum()
    return 100 * (2 * R) / (min_len * (min_len - 1)) if min_len > 1 else 0.0


def diagnose_rec_issue():
    """诊断REC为0的根本原因"""
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
        print(f"成功加载模型 (Epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        print("未找到模型checkpoint!")
        return

    model.eval()

    # 获取前5个样本进行详细分析
    num_samples_to_check = 5
    print(f"\n分析前{num_samples_to_check}个样本...\n")

    all_pred_ranges = []
    all_gt_ranges = []
    all_distances = []
    rec_values = []

    for idx in range(min(num_samples_to_check, len(test_dataset))):
        sample = test_dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)
        gt_path = sample['scanpath'].numpy()

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

        # 计算坐标范围
        pred_x_range = [pred_path[:, 0].min(), pred_path[:, 0].max()]
        pred_y_range = [pred_path[:, 1].min(), pred_path[:, 1].max()]
        gt_x_range = [gt_path[:, 0].min(), gt_path[:, 0].max()]
        gt_y_range = [gt_path[:, 1].min(), gt_path[:, 1].max()]

        all_pred_ranges.append({
            'x': pred_x_range,
            'y': pred_y_range
        })
        all_gt_ranges.append({
            'x': gt_x_range,
            'y': gt_y_range
        })

        # 计算点对点距离（像素坐标）
        h, w = config.image_size
        pred_pixels = pred_path.copy()
        pred_pixels[:, 0] = pred_pixels[:, 0] * w
        pred_pixels[:, 1] = pred_pixels[:, 1] * h

        gt_pixels = gt_path.copy()
        gt_pixels[:, 0] = gt_pixels[:, 0] * w
        gt_pixels[:, 1] = gt_pixels[:, 1] * h

        # 计算每个点的距离
        distances = np.sqrt(np.sum((pred_pixels - gt_pixels) ** 2, axis=1))
        all_distances.append(distances)

        # 计算REC（使用不同阈值）
        rec_12 = compute_rec(pred_path, gt_path, threshold=12.0, image_size=config.image_size)
        rec_24 = compute_rec(pred_path, gt_path, threshold=24.0, image_size=config.image_size)
        rec_48 = compute_rec(pred_path, gt_path, threshold=48.0, image_size=config.image_size)
        rec_values.append({
            'rec_12': rec_12,
            'rec_24': rec_24,
            'rec_48': rec_48
        })

        print(f"样本 {idx + 1}:")
        print(f"  预测路径范围: x=[{pred_x_range[0]:.4f}, {pred_x_range[1]:.4f}], "
              f"y=[{pred_y_range[0]:.4f}, {pred_y_range[1]:.4f}]")
        print(f"  真实路径范围: x=[{gt_x_range[0]:.4f}, {gt_x_range[1]:.4f}], "
              f"y=[{gt_y_range[0]:.4f}, {gt_y_range[1]:.4f}]")
        print(f"  平均点对点距离: {distances.mean():.2f} 像素")
        print(f"  最小点对点距离: {distances.min():.2f} 像素")
        print(f"  最大点对点距离: {distances.max():.2f} 像素")
        print(f"  距离<12像素的点数: {(distances < 12).sum()}/{len(distances)}")
        print(f"  距离<24像素的点数: {(distances < 24).sum()}/{len(distances)}")
        print(f"  距离<48像素的点数: {(distances < 48).sum()}/{len(distances)}")
        print(f"  REC(12px): {rec_12:.4f}")
        print(f"  REC(24px): {rec_24:.4f}")
        print(f"  REC(48px): {rec_48:.4f}")
        print()

    # 汇总统计
    print("=" * 70)
    print("汇总统计:")
    print("=" * 70)
    
    all_distances_flat = np.concatenate(all_distances)
    print(f"\n所有样本的点对点距离统计:")
    print(f"  平均距离: {all_distances_flat.mean():.2f} 像素")
    print(f"  中位数距离: {np.median(all_distances_flat):.2f} 像素")
    print(f"  最小距离: {all_distances_flat.min():.2f} 像素")
    print(f"  最大距离: {all_distances_flat.max():.2f} 像素")
    print(f"  标准差: {all_distances_flat.std():.2f} 像素")
    print(f"\n距离分布:")
    print(f"  <12像素: {(all_distances_flat < 12).sum()}/{len(all_distances_flat)} "
          f"({100*(all_distances_flat < 12).sum()/len(all_distances_flat):.2f}%)")
    print(f"  <24像素: {(all_distances_flat < 24).sum()}/{len(all_distances_flat)} "
          f"({100*(all_distances_flat < 24).sum()/len(all_distances_flat):.2f}%)")
    print(f"  <48像素: {(all_distances_flat < 48).sum()}/{len(all_distances_flat)} "
          f"({100*(all_distances_flat < 48).sum()/len(all_distances_flat):.2f}%)")
    print(f"  <96像素: {(all_distances_flat < 96).sum()}/{len(all_distances_flat)} "
          f"({100*(all_distances_flat < 96).sum()/len(all_distances_flat):.2f}%)")

    # 平均REC值
    avg_rec_12 = np.mean([r['rec_12'] for r in rec_values])
    avg_rec_24 = np.mean([r['rec_24'] for r in rec_values])
    avg_rec_48 = np.mean([r['rec_48'] for r in rec_values])
    print(f"\n平均REC值:")
    print(f"  REC(12px): {avg_rec_12:.4f}")
    print(f"  REC(24px): {avg_rec_24:.4f}")
    print(f"  REC(48px): {avg_rec_48:.4f}")

    # 诊断建议
    print("\n" + "=" * 70)
    print("诊断建议:")
    print("=" * 70)
    
    if all_distances_flat.mean() > 100:
        print("⚠️  平均距离过大（>100像素），预测路径与真实路径偏差很大")
        print("   建议：增加重建损失权重，或检查模型预测逻辑")
    elif all_distances_flat.mean() > 50:
        print("⚠️  平均距离较大（>50像素），预测路径与真实路径有一定偏差")
        print("   建议：适度增加重建损失权重")
    else:
        print("✓ 平均距离在合理范围内")
    
    if (all_distances_flat < 12).sum() == 0:
        print("⚠️  没有任何点在12像素范围内，REC为0")
        if (all_distances_flat < 24).sum() > 0:
            print("   建议：考虑将REC阈值从12像素提高到24像素")
        elif (all_distances_flat < 48).sum() > 0:
            print("   建议：考虑将REC阈值从12像素提高到48像素")
        else:
            print("   建议：需要大幅提高重建损失权重，改善点对点匹配")
    
    # 保存诊断结果
    diagnosis = {
        'avg_distance': float(all_distances_flat.mean()),
        'median_distance': float(np.median(all_distances_flat)),
        'min_distance': float(all_distances_flat.min()),
        'max_distance': float(all_distances_flat.max()),
        'std_distance': float(all_distances_flat.std()),
        'points_within_12px': int((all_distances_flat < 12).sum()),
        'points_within_24px': int((all_distances_flat < 24).sum()),
        'points_within_48px': int((all_distances_flat < 48).sum()),
        'total_points': int(len(all_distances_flat)),
        'avg_rec_12': float(avg_rec_12),
        'avg_rec_24': float(avg_rec_24),
        'avg_rec_48': float(avg_rec_48),
        'sample_ranges': [
            {
                'pred': all_pred_ranges[i],
                'gt': all_gt_ranges[i]
            }
            for i in range(len(all_pred_ranges))
        ]
    }
    
    output_path = Path('logs') / 'rec_diagnosis.json'
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(diagnosis, f, indent=2, ensure_ascii=False)
    
    print(f"\n诊断结果已保存到: {output_path.absolute()}")


if __name__ == '__main__':
    diagnose_rec_issue()
