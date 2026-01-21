"""
修复版评估脚本 - 确保坐标正确
"""
import torch
import numpy as np
from pathlib import Path
from config_mamba_adaptive import MambaAdaptiveConfig
from data.dataset import Salient360ScanpathDataset
from models.mamba_adaptive_scanpath import MambaAdaptiveScanpath
from metrics.scanpath_metrics import compute_all_metrics_extended

def evaluate_model():
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
        print(f"成功加载模型 (Epoch {checkpoint['epoch']})")
    else:
        print("未找到模型checkpoint!")
        return
    
    model.eval()
    
    # 评估10个样本
    num_samples = min(10, len(test_dataset))
    all_metrics = []
    
    print(f"\n开始评估 {num_samples} 个样本...\n")
    
    # 获取唯一的图像key
    unique_keys = []
    seen_keys = set()
    for sample in test_dataset.samples:
        key = sample['key']
        if key not in seen_keys:
            unique_keys.append(key)
            seen_keys.add(key)
            if len(unique_keys) >= num_samples:
                break
    
    for idx, key in enumerate(unique_keys):
        # 获取该图像的所有样本
        image_samples = [s for s in test_dataset.samples if s['key'] == key]
        
        # 使用第一个样本的图像
        first_sample_idx = test_dataset.samples.index(image_samples[0])
        sample = test_dataset[first_sample_idx]
        
        image = sample['image'].unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            pred_scanpath, _, _ = model(image, gt_scanpaths=None, 
                                       teacher_forcing_ratio=0.0,
                                       enable_early_stop=False,
                                       use_gt_start=False)
        
        pred_path = pred_scanpath[0].cpu().numpy()
        
        # 确保坐标在[0,1]范围内
        pred_path = np.clip(pred_path, 0.0, 1.0)
        
        # 调试：打印坐标范围
        if idx == 0:
            print(f"调试信息 - 样本0:")
            print(f"  预测路径范围: x=[{pred_path[:, 0].min():.4f}, {pred_path[:, 0].max():.4f}], "
                  f"y=[{pred_path[:, 1].min():.4f}, {pred_path[:, 1].max():.4f}]")
        
        # 获取所有GT路径
        all_gt_paths = [s['scanpath'].numpy() if isinstance(s['scanpath'], torch.Tensor) else s['scanpath'] 
                        for s in image_samples]
        
        # 确保GT路径也在[0,1]范围
        all_gt_paths = [np.clip(gt, 0.0, 1.0) for gt in all_gt_paths]
        
        # 获取显著性图
        saliency = sample.get('saliency_map')
        if saliency is not None and isinstance(saliency, torch.Tensor):
            saliency = saliency.squeeze().cpu().numpy()
        
        # 计算与所有GT的指标
        metrics_list = []
        for gt_path in all_gt_paths:
            metrics = compute_all_metrics_extended(
                pred_path, gt_path,
                saliency_map=saliency,
                image_size=config.image_size
            )
            metrics_list.append(metrics)
        
        # 找到最佳匹配
        best_idx = np.argmin([m['DTW'] for m in metrics_list])
        best_metrics = metrics_list[best_idx]
        
        # 计算平均指标
        avg_metrics = {}
        for key_name in best_metrics.keys():
            avg_metrics[key_name] = np.mean([m[key_name] for m in metrics_list])
        
        all_metrics.append({
            'best': best_metrics,
            'avg': avg_metrics,
            'num_gt_paths': len(all_gt_paths)
        })
        
        # 打印该样本的结果
        print(f"样本 {idx} (共 {len(all_gt_paths)} 条GT路径):")
        print(f"  最佳匹配:")
        print(f"    LEV={best_metrics['LEV']:.2f}, DTW={best_metrics['DTW']:.2f}, REC={best_metrics['REC']:.2f}")
        print(f"    SIM={best_metrics['SIM']:.4f} (越小越好)")
        print(f"    MM_Vector={best_metrics['MM_Vector']:.3f}, MM_Length={best_metrics['MM_Length']:.3f}, MM_Position={best_metrics['MM_Position']:.3f}")
        if 'NSS' in best_metrics:
            print(f"    NSS={best_metrics['NSS']:.3f}, CC={best_metrics['CC']:.3f}, SalCoverage={best_metrics['SalCoverage']:.3f}")
        print()
    
    # 打印汇总统计
    print("="*70)
    print(f"汇总统计 (基于 {num_samples} 个样本):")
    print("="*70)
    
    print("\n【最佳匹配指标】")
    for metric_name in ['LEV', 'DTW', 'REC', 'SIM', 'MM_Vector', 'MM_Length', 'MM_Position']:
        values = [m['best'][metric_name] for m in all_metrics]
        print(f"  {metric_name:12s}: {np.mean(values):7.3f} ± {np.std(values):6.3f}")
    
    # 显著性指标（如果有）
    if 'NSS' in all_metrics[0]['best']:
        print("\n【显著性指标】")
        for metric_name in ['NSS', 'CC', 'SalCoverage']:
            values = [m['best'][metric_name] for m in all_metrics]
            print(f"  {metric_name:12s}: {np.mean(values):7.3f} ± {np.std(values):6.3f}")
    
    print("\n【平均指标】")
    for metric_name in ['LEV', 'DTW', 'REC', 'SIM']:
        values = [m['avg'][metric_name] for m in all_metrics]
        print(f"  {metric_name:12s}: {np.mean(values):7.3f} ± {np.std(values):6.3f}")
    
    print("="*70)
    
    # 指标解释
    print("\n指标说明:")
    print("  LEV: Levenshtein距离 (越小越好, <20优秀, 20-25良好, >25一般)")
    print("  DTW: 动态时间规整距离 (越小越好, 归一化坐标下<5优秀)")
    print("  REC: 重现率 (越大越好)")
    print("  SIM: 平均欧氏距离 (越小越好, <0.2优秀, 0.2-0.3良好)")
    print("  MM_Vector: 方向相似度 (0-1, 越大越好)")
    print("  MM_Length: 步长相似度 (0-1, 越大越好)")
    print("  MM_Position: 位置相似度 (0-1, 越大越好)")
    if 'NSS' in all_metrics[0]['best']:
        print("  NSS: 归一化显著性得分 (越大越好)")
        print("  CC: 相关系数 (0-1, 越大越好)")
        print("  SalCoverage: 显著性覆盖率 (0-1, 越大越好)")

if __name__ == '__main__':
    evaluate_model()
