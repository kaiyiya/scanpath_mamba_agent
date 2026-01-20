# 模型训练改进方案 V2 - 更激进的改进

## 改进日期
2026-01-20 (第二轮)

## 第一轮结果分析 (Epoch 145)

### ✅ 改善的地方
- **X方向多样性**: 0.07-0.13 → 0.15-0.20 ✓ (达标)
- **REC指标**: 4.60 → 6.00 (改善)

### ⚠️ 仍存在的问题
1. **Y方向多样性严重不足**: 标准差只有0.04-0.07，目标是0.10-0.15
2. **路径过于集中在中心**: y均值几乎都在0.49-0.51
3. **路径不平滑**: 用户反馈路径杂乱无章
4. **位置误差略有退步**: 0.235 → 0.244

## 根本原因分析

### 问题1: Y方向约束力度不够
- 之前Y方向权重只是X方向的2倍，不够
- Y方向阈值0.020太低，需要提高到0.025

### 问题2: 重构损失仍占主导
- 模型学习到"预测中心最安全"的策略
- 需要大幅提高多样性损失的权重

### 问题3: 缺少针对Y方向中心聚集的专门惩罚
- 之前完全移除了中心惩罚
- 但Y方向确实需要惩罚过度集中在0.5附近

## V2 激进改进方案

### ✅ 改进1: Y方向多样性权重从2倍提高到5倍

**位置**: train_mamba_adaptive.py:148-157

```python
# V1: Y方向权重2倍
min_var_y = 0.020
diversity_loss = diversity_loss_x + 2.0 * diversity_loss_y

# V2: Y方向权重5倍，阈值提高
min_var_y = 0.025  # 从0.020提高到0.025
diversity_loss = diversity_loss_x + 5.0 * diversity_loss_y  # 从2.0提高到5.0
```

---

### ✅ 改进2: 分别约束X和Y方向的覆盖范围

**位置**: train_mamba_adaptive.py:130-151

```python
# V1: 统一处理覆盖范围
coverage_loss = torch.mean(((0.3 - pred_range).clamp(min=0.0)) ** 2) * 0.5

# V2: 分别约束X和Y方向
pred_range_x = pred_range[:, 0]
pred_range_y = pred_range[:, 1]

min_range_x = 0.3
min_range_y = 0.25  # Y方向要求更大覆盖

coverage_loss_x = torch.mean(((min_range_x - pred_range_x).clamp(min=0.0)) ** 2)
coverage_loss_y = torch.mean(((min_range_y - pred_range_y).clamp(min=0.0)) ** 2)

# Y方向给予3倍权重
coverage_loss = coverage_loss_x + 3.0 * coverage_loss_y
```

---

### ✅ 改进3: 新增Y方向中心聚集惩罚

**位置**: train_mamba_adaptive.py:185-196

```python
# 新增：专门惩罚Y方向过于集中在0.5附近
pred_mean_y = pred_mean[:, 1]  # Y方向的均值

# 惩罚Y方向均值过于接近0.5
y_center_dist = torch.abs(pred_mean_y - 0.5)
# 当距离小于0.1时（即均值在[0.4, 0.6]），给予惩罚
y_too_centered = (y_center_dist < 0.1).float()
y_center_penalty = torch.mean(y_too_centered * (0.1 - y_center_dist) ** 2) * 10.0

center_penalty = y_center_penalty
center_weight = 0.3 -> 0.5  # 启用Y方向中心惩罚
```

---

### ✅ 改进4: 大幅提高所有正则化权重

**位置**: train_mamba_adaptive.py:234-277

```python
# V1 权重
coverage_weight = 0.05 -> 0.10
diversity_weight = 0.1 -> 0.2
smoothness_weight = 1.0
jump_weight = 0.5

# V2 权重（激进提高）
coverage_weight = 0.3 -> 0.5  # 提高6倍
diversity_weight = 0.5 -> 0.8  # 提高4倍
center_weight = 0.3 -> 0.5    # 新增Y方向中心惩罚
smoothness_weight = 1.5       # 提高50%
jump_weight = 0.8             # 提高60%
direction_weight = 0.5        # 提高67%
acceleration_weight = 0.5     # 提高67%
direction_continuity_weight = 0.5  # 提高67%
batch_diversity_weight = 0.2 -> 0.3  # 提高3倍
```

---

### ✅ 改进5: 提高batch多样性阈值和权重

**位置**: train_mamba_adaptive.py:279-291

```python
# V1
min_batch_diversity = 0.01
batch_diversity_weight = 0.05 -> 0.10

# V2
min_batch_diversity = 0.015  # 提高50%
batch_diversity_weight = 0.2 -> 0.3  # 提高3倍
```

---

## 权重对比总结

| 损失项 | V1 (Epoch 1-80) | V2 (Epoch 1-80) | 提升倍数 |
|--------|----------------|----------------|---------|
| coverage_weight | 0.05 | 0.3 | **6x** |
| diversity_weight | 0.1 | 0.5 | **5x** |
| center_weight | 0.0 | 0.3 | **新增** |
| smoothness_weight | 1.0 | 1.5 | 1.5x |
| jump_weight | 0.5 | 0.8 | 1.6x |
| direction_weight | 0.3 | 0.5 | 1.67x |
| batch_diversity_weight | 0.05 | 0.2 | **4x** |

## 预期改进效果

### 多样性指标
- **Y方向标准差**: 0.04-0.07 → **0.10-0.15** (关键目标)
- X方向标准差: 0.15-0.20 (保持)

### 位置分布
- **Y方向均值**: 0.49-0.51 → **0.35-0.65** (更分散)
- X方向均值: 0.44-0.51 (保持或更分散)

### 路径质量
- **路径平滑度**: 显著改善（权重提高50%）
- **覆盖范围**: Y方向覆盖更大

### 评估指标
- 位置误差: 0.244 → < 0.22
- LEV: 29.70 → < 25
- DTW: 2441 → < 2000
- REC: 6.00 → > 10

## 改进策略说明

### 为什么这么激进？

1. **Y方向问题严重**: 标准差只有0.04-0.07，是X方向的1/3，必须大幅提高约束
2. **中心聚集严重**: y均值都在0.49-0.51，说明模型陷入局部最优
3. **重构损失过强**: 需要大幅提高正则化权重来平衡

### 风险控制

1. **保持平滑性约束**: 同时提高平滑性权重，避免路径过于杂乱
2. **分阶段调整**: 前80 epochs使用中等权重，后续逐渐增加
3. **边界约束放宽**: 允许更大的探索空间

## 下一步

1. **重新训练**: 使用V2配置训练300 epochs
2. **重点监控**:
   - Y方向标准差是否达到0.10+
   - Y方向均值是否更分散
   - 路径是否更平滑
3. **可视化对比**: 对比V1和V2的预测路径
4. **如果仍不够**: 考虑进一步提高Y方向权重到10倍

## 备注

- V2是基于V1 (Epoch 145)的结果进行的激进改进
- 核心策略：大幅提高Y方向相关损失的权重
- 如果V2仍不够，可以考虑修改模型架构（如在Focus网络中增加Y方向的注意力）
