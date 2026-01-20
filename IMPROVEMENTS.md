# 模型训练改进方案实施记录

## 改进日期
2026-01-20

## 问题诊断

### 当前问题
基于Epoch 240的测试结果：
1. **预测路径聚集在图像中心** (x: 0.44-0.55, y: 0.47-0.50)
2. **Y方向多样性严重不足** (标准差仅0.03-0.05)
3. **路径看起来杂乱无章** (平滑性不足)
4. **训练后期震荡** (100 epoch后损失反弹)

### 评估指标
- 位置误差: 0.235 (最佳匹配), 0.370 (平均)
- LEV: 29.80 (接近30，几乎无相似性)
- DTW: 2496 (仍然很高)
- REC: 4.60 (很低)

## 实施的改进方案

### ✅ 方案1: 分别约束X和Y方向的多样性 (关键！)

**位置**: train_mamba_adaptive.py:139-157

**改进内容**:
```python
# 之前：统一处理X和Y方向
min_var = 0.015
diversity_loss = torch.mean(((min_var - pred_var).clamp(min=0.0)) ** 2) * 0.5

# 改进：分别约束X和Y方向
pred_var_x = pred_var[:, 0]
pred_var_y = pred_var[:, 1]

min_var_x = 0.015  # x方向标准差约0.12
min_var_y = 0.020  # y方向标准差约0.14 (提高！)

diversity_loss_x = torch.mean(((min_var_x - pred_var_x).clamp(min=0.0)) ** 2)
diversity_loss_y = torch.mean(((min_var_y - pred_var_y).clamp(min=0.0)) ** 2)

# Y方向给予2倍权重
diversity_loss = diversity_loss_x + 2.0 * diversity_loss_y
```

**预期效果**: Y方向标准差从0.03-0.05提升到0.10-0.15

---

### ✅ 方案2: 增强轨迹平滑性约束

**位置**: train_mamba_adaptive.py:234-277

**改进内容**:
```python
# 之前的权重
smoothness_weight = 0.15
jump_weight = 0.05
direction_weight = 0.05

# 改进后的权重
smoothness_weight = 1.0   # 从0.15提高到1.0
jump_weight = 0.5         # 从0.05提高到0.5
direction_weight = 0.3    # 从0.05提高到0.3
acceleration_weight = 0.3  # 新增
direction_continuity_weight = 0.3  # 新增
```

**预期效果**: 路径更加平滑连贯，不再杂乱无章

---

### ✅ 方案3: 移除中心惩罚

**位置**: train_mamba_adaptive.py:160-163

**改进内容**:
```python
# 之前：惩罚接近中心的预测
very_close_to_center = (mean_center_dist < 0.01).float()
center_penalty = torch.mean(very_close_to_center * (0.01 - mean_center_dist) * 5.0)

# 改进：完全移除中心惩罚
center_penalty = torch.tensor(0.0, device=predicted_scanpaths.device)
point_center_penalty = torch.tensor(0.0, device=predicted_scanpaths.device)
center_weight = 0.0  # 权重设为0
```

**预期效果**: 模型可以自由探索整个图像空间

---

### ✅ 方案4: 放宽边界约束

**位置**: train_mamba_adaptive.py:146-158

**改进内容**:
```python
# 之前：严格的边界约束
boundary_min = 0.05
boundary_max = 0.95
boundary_penalty = ... * 10.0

# 改进：放宽边界，降低惩罚
boundary_min = 0.02  # 从0.05放宽到0.02
boundary_max = 0.98  # 从0.95放宽到0.98
boundary_penalty = ... * 2.0  # 从10.0降低到2.0
```

**预期效果**: 允许预测更接近边缘，增加探索空间

---

### ✅ 方案5: 调整Teacher Forcing策略

**位置**: train_mamba_adaptive.py:93-104

**改进内容**:
```python
# 之前：快速衰减
# Epoch 1-50: 0.5
# Epoch 51-100: 0.5 -> 0.3
# Epoch 100+: 0.3

# 改进：更慢的衰减
if epoch <= 100:
    teacher_forcing_ratio = 0.7 - 0.3 * (epoch / 100.0)  # 0.7 -> 0.4
elif epoch <= 200:
    teacher_forcing_ratio = 0.4 - 0.2 * ((epoch - 100) / 100.0)  # 0.4 -> 0.2
else:
    teacher_forcing_ratio = 0.2
```

**预期效果**: 模型有更多时间学习生成连贯路径

---

### ✅ 方案6: 降低VAE的KL散度权重

**位置**: train_mamba_adaptive.py:125-129

**改进内容**:
```python
# 之前
beta = min(0.01, 0.005 * (1.01 ** (epoch - 1)))

# 改进：进一步降低beta
beta = min(0.005, 0.001 * (1.01 ** (epoch - 1)))  # 从0.005降到0.001
```

**预期效果**: 减少VAE引入的随机性，路径更稳定

---

### ✅ 方案7: 学习率调度 (已实现)

**位置**: train_mamba_adaptive.py:45-51

**当前实现**: 已使用余弦退火
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config.num_epochs,
    eta_min=config.learning_rate * 0.01
)
```

**说明**: 学习率调度已经是最优方案，无需修改

---

## 权重调整总结

### 阶段1 (Epoch 1-80): 重构+平滑性优先
- coverage_weight: 0.05
- diversity_weight: 0.1
- center_weight: 0.0 (移除)
- boundary_weight: 0.3 (降低)
- smoothness_weight: 1.0 (大幅提高)
- jump_weight: 0.5 (提高)
- direction_weight: 0.3 (提高)
- acceleration_weight: 0.3
- direction_continuity_weight: 0.3
- batch_diversity_weight: 0.05

### 阶段2 (Epoch 81-150): 逐渐增加多样性
- coverage_weight: 0.05 -> 0.10
- diversity_weight: 0.1 -> 0.2
- 其他权重保持不变

### 阶段3 (Epoch 151+): 平衡优化
- coverage_weight: 0.10
- diversity_weight: 0.2
- 其他权重保持不变

## 预期改进效果

### 多样性指标
- X方向标准差: 0.07-0.13 → **0.12-0.20**
- Y方向标准差: 0.03-0.05 → **0.10-0.15** (关键改进)

### 位置精度
- 位置误差: 0.235 → **< 0.18**

### 路径质量
- LEV: 29.8 → **< 25**
- DTW: 2496 → **< 2000**
- REC: 4.6 → **> 8**
- 路径平滑度: 显著改善

### 训练稳定性
- 消除100 epoch后的震荡
- 更平滑的收敛曲线

## 下一步

1. **重新训练模型**: 使用改进后的配置训练300 epochs
2. **监控关键指标**:
   - Y方向标准差是否提升
   - 路径是否更平滑
   - 位置误差是否下降
3. **可视化对比**: 对比改进前后的预测路径
4. **调优**: 根据训练结果微调权重

## 备注

- 所有改进都是基于Epoch 240的测试结果分析
- 核心问题是Y方向多样性不足和路径平滑性差
- 改进方案聚焦于这两个关键问题
- 学习率调度已经是最优方案，无需修改
