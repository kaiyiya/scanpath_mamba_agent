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

    # 训练参数（修复配置）
    batch_size = 16  # 保持batch size
    num_epochs = 50  # 保持50 epoch
    learning_rate = 1.2e-4  # 进一步降低学习率（从1.5e-4到1.2e-4），改善训练稳定性
    weight_decay = 2e-3  # 提高权重衰减（从1.5e-3到2e-3），减少过拟合
    lr_decay = 0.995  # 学习率衰减因子（保持）

    # 数据增强
    use_augmentation = True  # 启用数据增强

    # 数据加载
    num_workers = 0
    pin_memory = True

    # 日志和保存（快速验证配置）
    log_dir = './logs'  # 使用原logs目录
    checkpoint_dir = './checkpoints'  # 使用原checkpoints目录
    save_interval = 10  # 减少保存频率（从5到10）
    log_interval = 10

    # 验证（快速验证配置）
    val_interval = 5  # 每5个epoch验证一次

    # ==================== 新增配置项 ====================
    # 第一阶段配置
    use_wrap_around = True  # 启用360度wrap around
    use_simplified_loss = True  # 使用简化损失函数
    teacher_forcing_strategy = 'exponential'  # 'linear', 'exponential'

    # 第二阶段配置
    use_y_attention = True  # 启用Y方向注意力
    y_attention_bias_scale = 0.1  # Y方向偏置缩放
    use_simplified_feature_update = True  # 使用简化特征更新
    feature_update_sigma = 0.2  # 空间权重的sigma

    # 损失权重（简化后）
    loss_weights = {
        'reconstruction': 1.0,
        'kl': 0.005,
        'spatial_coverage': 0.5,  # 阶段1初始值
        'trajectory_smoothness': 1.5,
        'direction_consistency': 0.5,
        'boundary': 0.2
    }
