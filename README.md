# Hierarchical Mamba-Agent Scanpath Model

结合Mamba状态空间模型 + AgentAttention + Glance特征提取的创新架构

## 核心创新

### 1. Mamba序列建模
- **参数共享**: 参数量O(1)，与序列长度无关
- **线性复杂度**: O(T)时间复杂度，支持任意长度序列
- **长期依赖**: 状态空间模型天然捕获长期依赖关系
- **解决AdaptiveNN问题**: 不再为每个时间步创建独立参数

### 2. AgentAttention空间引导
- **避免Focus恶性循环**: 不需要crop patch，直接用agent tokens
- **线性复杂度**: O(N)而不是O(N²)
- **可解释性**: Agent tokens可视化显示关注区域
- **49个agent tokens**: 作为"虚拟注视点"引导空间注意力

### 3. 层次化建模
```
输入图像 (B, 3, 256, 512)
    ↓
[Glance网络] → 全局特征 (B, 196, 384)
    ↓
[AgentAttention] → 空间引导特征 (B, 196, 384)
    ↓
[Mamba序列生成] → 预测轨迹 (B, 30, 2)
```

## 架构优势

| 特性 | AdaptiveNN | Markov模型 | **Mamba-Agent** |
|------|-----------|-----------|----------------|
| 参数量 | O(T) | O(1) | **O(1)** |
| 序列长度 | 固定4步 | 任意 | **任意** |
| 历史依赖 | 完整 | 只看前一步 | **长期依赖** |
| 空间建模 | Focus crop | 空间注意力 | **Agent注意力** |
| 复杂度 | O(T²) | O(T) | **O(T)** |
| 创新性 | 中 | 低 | **高** |

## 快速开始

### 环境要求

```bash
# 安装Mamba SSM
pip install mamba-ssm

# 或者使用稳定安装方法（推荐）
# 1. 查看wheel文件: pip install mamba-ssm --no-cache-dir --verbose
# 2. 下载对应的.whl文件
# 3. pip install mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```

### 训练模型

```bash
cd scanpath_mamba_agent
python train_mamba_agent.py
```

### 可视化结果

```bash
python visualize_mamba_agent.py
```

### 评估模型

```bash
python evaluate_mamba_agent.py
```

## 文件结构

```
scanpath_mamba_agent/
├── models/
│   ├── mamba_agent_scanpath.py       # 主模型实现
│   ├── improved_model_v5.py          # Glance特征提取器
│   └── sphere_conv_optimized.py      # 球面卷积
├── data/
│   └── dataset.py                    # 数据加载器
├── config_mamba_agent.py             # 配置文件
├── train_mamba_agent.py              # 训练脚本
├── visualize_mamba_agent.py          # 可视化脚本
├── evaluate_mamba_agent.py           # 评估脚本
├── checkpoints/                      # 模型检查点
├── logs/                             # 训练日志
└── visualization_results/            # 可视化结果
```

## 配置参数

主要超参数（config_mamba_agent.py）：
- `seq_len`: 30 - 序列长度
- `feature_dim`: 384 - 特征维度
- `d_state`: 256 - SSM状态维度
- `agent_num`: 49 - Agent tokens数量
- `num_heads`: 8 - 注意力头数
- `learning_rate`: 1e-4
- `num_epochs`: 50

## 模型组件详解

### MambaScanpathGenerator
```python
# Mamba序列生成器
# 参数共享，支持任意长度序列
self.mamba = Mamba(
    d_model=384,      # 模型维度
    d_state=256,      # SSM状态扩展因子
    d_conv=4,         # 局部卷积宽度
    expand=2          # 块扩展因子
)
```

### AgentAttention
```python
# Agent注意力机制
# 使用49个agent tokens作为中介
# 降低复杂度到O(N)
self.agent_attention = AgentAttention(
    dim=384,
    num_heads=8,
    agent_num=49,     # 7x7 agent tokens
    window=14         # 特征图大小
)
```

### 完整流程
1. **Glance网络**: 提取全局特征 (B, 196, 384)
2. **AgentAttention**: 空间引导，agent tokens关注显著区域
3. **Mamba序列建模**: 逐步生成30步轨迹
4. **位置预测**: 输出归一化坐标 [0, 1]

## 创新点总结

### 1. 首次将Mamba用于扫描路径预测
- 状态空间模型的序列建模能力
- 线性复杂度支持长序列
- 参数共享避免AdaptiveNN的参数爆炸

### 2. AgentAttention避免Focus恶性循环
- 不需要crop操作
- Agent tokens自动学习关注区域
- 可解释性强（可视化agent位置）

### 3. 层次化建模框架
- 全局层（Glance）：整体感知
- 序列层（Mamba）：时序依赖
- 空间层（AgentAttention）：区域关注

### 4. 针对360度全景图优化
- 球面卷积处理周期性边界
- Agent tokens学习全景图特殊模式

## 论文价值

这个架构可以写成一篇高质量论文：

**标题**: Hierarchical Mamba-Agent Network for Long-Sequence Scanpath Prediction in 360° Images

**主要贡献**:
1. 首次将Mamba状态空间模型用于扫描路径预测
2. AgentAttention机制避免Focus机制的恶性循环
3. 层次化建模框架结合全局感知和局部关注
4. 在360度全景图上的成功应用

**实验对比**:
- vs AdaptiveNN: 支持长序列（30步 vs 4步）
- vs Markov模型: 更强的长期依赖建模能力
- vs Transformer: 线性复杂度，更高效

## 与其他方案对比

### vs 马尔可夫模型（scanpath_markov/）
- **优势**:
  - 更强的长期依赖建模（Mamba vs 只看前一步）
  - Agent注意力更灵活（vs 简单空间注意力）
  - 更高的创新性
- **劣势**:
  - 需要安装mamba-ssm（环境依赖）
  - 模型稍复杂

### vs AdaptiveNN原始架构
- **优势**:
  - 支持任意长度序列（vs 固定4步）
  - 参数量O(1)（vs O(T)）
  - 避免Focus恶性循环
- **保留**:
  - Glance全局特征提取
  - 球面卷积处理全景图

## 预期效果

基于架构设计，预期性能：
- 平均位置误差 (ADE): < 0.30
- 预测多样性: > 1.0
- 轨迹平滑度: 0.03-0.06
- 覆盖范围: X > 0.18, Y > 0.12

## 下一步优化

1. **添加ModernTCN平滑层**: 进一步优化轨迹平滑度
2. **多尺度Agent tokens**: 不同尺度的agent关注不同层次特征
3. **对比学习**: 相似图像产生相似轨迹
4. **显著性图引导**: 结合显著性检测增强可解释性

## 引用

如果使用本架构，请引用：
```
@article{mamba_agent_scanpath_2026,
  title={Hierarchical Mamba-Agent Network for Long-Sequence Scanpath Prediction in 360° Images},
  author={Your Name},
  year={2026}
}
```

## 许可证

MIT License
