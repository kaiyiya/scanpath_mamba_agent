"""
Mamba-Adaptive扫描路径模型配置
结合 Mamba + AdaptiveNN Focus机制
"""
import torch

class MambaAdaptiveConfig:
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据配置
    # 使用H+HE合并数据集（包含头动和眼动数据，不进行旋转增强）
    processed_data_path = '../../adaptive_scanpath/data_processing/processed_data/Salient360_H_HE_processed.pkl'
    image_size = (256, 512)
    seq_len = 30  # 序列长度

    # Glance网络参数
    feature_dim = 384  # Glance特征维度
    feature_size = 14  # 特征图大小 (14x14=196)

    # Focus网络参数（AdaptiveNN核心）
    focus_patch_size = 224  # 局部patch大小

    # Mamba参数
    d_state = 256  # SSM状态维度
    d_conv = 4  # 局部卷积宽度
    expand = 2  # 扩展因子
    
    # 自动停止参数
    enable_early_stop = True  # 是否启用自动停止（推理时）
    stop_threshold = 0.5  # 停止阈值，继续概率低于此值时停止
    min_steps = 5  # 最小步数，至少生成这么多步才允许停止

    # 训练参数
    batch_size = 8
    num_epochs = 300  # 增加到300 epochs（参考ScanDMM的500 epochs训练策略）
    learning_rate = 1e-4  # 提高初始学习率（从5e-5到1e-4），加快训练速度
    weight_decay = 2e-3  # 增加权重衰减（从1e-3提升到2e-3）
    lr_decay = 0.9995  # 学习率衰减因子（每个epoch衰减，从0.9999调整为0.9995，更快的衰减）

    # 数据增强
    use_augmentation = True  # 启用数据增强

    # 数据加载
    num_workers = 0
    pin_memory = True

    # 日志和保存
    log_dir = './logs_adaptive'
    checkpoint_dir = './checkpoints_adaptive'
    save_interval = 5
    log_interval = 10

    # 验证
    val_interval = 5
