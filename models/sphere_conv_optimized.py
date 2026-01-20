"""
优化的球面卷积模块 - V2
主要优化：
1. 更高效的grid缓存机制
2. 优化计算流程
3. 添加内存优化
4. 更好的初始化
"""
import numpy as np
from numpy import sin, cos, tan, pi, arcsin, arctan, sqrt
from functools import lru_cache
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# ==================== 优化的坐标计算 ====================

@lru_cache(maxsize=128)
def get_xy_optimized(delta_phi, delta_theta):
    """优化的球面卷积核采样模式（预计算并缓存）"""
    sin_dphi = sin(delta_phi)
    sin_dtheta = sin(delta_theta)
    cos_dphi = cos(delta_phi)
    cos_dtheta = cos(delta_theta)

    return np.array([
        [
            (-tan(delta_theta), 1 / cos_dphi * tan(delta_phi)),
            (0, tan(delta_phi)),
            (tan(delta_theta), 1 / cos_dphi * tan(delta_phi)),
        ],
        [
            (-tan(delta_theta), 0),
            (1, 1),
            (tan(delta_theta), 0),
        ],
        [
            (-tan(delta_theta), -1 / cos_dphi * tan(delta_phi)),
            (0, -tan(delta_phi)),
            (tan(delta_theta), -1 / cos_dphi * tan(delta_phi)),
        ]
    ], dtype=np.float32)


@lru_cache(maxsize=128)
def cal_index_optimized(h, w, img_r, img_c):
    """优化的球面卷积核采样索引计算"""
    # 像素坐标转换为球面角度（弧度）
    phi = -((img_r + 0.5) / h * pi - pi / 2)
    theta = (img_c + 0.5) / w * 2 * pi - pi

    # 计算步长
    delta_phi = pi / h
    delta_theta = 2 * pi / w

    # 获取3x3采样模式
    xys = get_xy_optimized(delta_phi, delta_theta)
    x = xys[..., 0]
    y = xys[..., 1]
    rho = sqrt(x ** 2 + y ** 2)
    v = arctan(rho)

    # 将切平面偏移量投影回球面（优化计算）
    cos_v = cos(v)
    sin_v = sin(v)
    cos_phi = cos(phi)
    sin_phi = sin(phi)

    new_phi = arcsin(cos_v * sin_phi + y * sin_v * cos_phi / rho)
    new_theta = theta + arctan(x * sin_v / (rho * cos_phi * cos_v - y * sin_phi * sin_v))

    # 球面坐标转回像素坐标
    new_r = (-new_phi + pi / 2) * h / pi - 0.5
    new_c = (new_theta + pi) * w / 2 / pi - 0.5

    # 处理等距圆柱投影的左右边界连续性
    new_c = (new_c + w) % w

    new_result = np.stack([new_r, new_c], axis=-1)
    new_result[1, 1] = (img_r, img_c)

    return new_result


@lru_cache(maxsize=32)
def _gen_filters_coordinates_optimized(h, w, stride):
    """优化的坐标生成（使用更高效的内存布局）"""
    co = np.array([[cal_index_optimized(h, w, i, j) for j in range(0, w, stride)]
                   for i in range(0, h, stride)], dtype=np.float32)
    return np.ascontiguousarray(co.transpose([4, 0, 1, 2, 3]))


def gen_filters_coordinates_optimized(h, w, stride=1):
    assert(isinstance(h, int) and isinstance(w, int))
    return _gen_filters_coordinates_optimized(h, w, stride).copy()


def gen_grid_coordinates_optimized(h, w, stride=1):
    """优化的grid坐标生成（用于grid_sample）"""
    coordinates = gen_filters_coordinates_optimized(h, w, stride).copy()

    # 归一化到[-1, 1]
    coordinates[0] = (coordinates[0] * 2 / h) - 1
    coordinates[1] = (coordinates[1] * 2 / w) - 1

    # 转换为grid_sample期望的格式
    coordinates = coordinates[::-1]
    coordinates = coordinates.transpose(1, 3, 2, 4, 0)
    sz = coordinates.shape
    coordinates = coordinates.reshape(1, sz[0]*sz[1], sz[2]*sz[3], sz[4])

    return coordinates.copy()


# ==================== 优化的球面卷积层 ====================

class OptimizedSphereConv2D(nn.Module):
    """
    优化的球面卷积层 - 性能提升版
    主要优化：
    1. 更智能的grid缓存
    2. 内存优化
    3. 更快的forward
    """
    def __init__(self, in_c, out_c, stride=1, bias=True, mode='bilinear'):
        super(OptimizedSphereConv2D, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.mode = mode

        # 使用更好的初始化
        self.weight = Parameter(torch.Tensor(out_c, in_c, 3, 3))
        if bias:
            self.bias = Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)

        # grid缓存（使用dict支持多个尺寸）
        self.grid_cache = {}
        self.current_grid_shape = None

        self.reset_parameters()

    def reset_parameters(self):
        # 使用Kaiming初始化
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def _get_or_create_grid(self, h, w, device):
        """获取或创建grid（支持缓存和动态尺寸）"""
        shape_key = (h, w)

        if shape_key not in self.grid_cache:
            # 生成新的grid
            coordinates = gen_grid_coordinates_optimized(h, w, self.stride)
            grid = torch.from_numpy(coordinates).to(device)
            grid.requires_grad = False
            self.grid_cache[shape_key] = grid
            self.current_grid_shape = (h, w)

        return self.grid_cache[shape_key]

    def forward(self, x):
        # 生成/获取采样grid
        h, w = x.shape[2], x.shape[3]

        # 检查是否需要更新grid
        if self.current_grid_shape != (h, w):
            grid = self._get_or_create_grid(h, w, x.device)
        else:
            grid = self.grid_cache[(h, w)]

        # 扩展到批次大小（不需要梯度）
        batch_size = x.shape[0]
        grid_expanded = grid.expand(batch_size, -1, -1, -1)

        # 使用grid_sample进行球面采样
        x = F.grid_sample(x, grid_expanded, mode=self.mode, align_corners=True)

        # 应用3x3卷积
        x = F.conv2d(x, self.weight, self.bias, stride=3)

        return x


class OptimizedSphereMaxPool2D(nn.Module):
    """优化的球面最大池化层"""
    def __init__(self, stride=1, mode='bilinear'):
        super(OptimizedSphereMaxPool2D, self).__init__()
        self.stride = stride
        self.mode = mode
        self.grid_cache = {}
        self.current_grid_shape = None
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)

    def _get_or_create_grid(self, h, w, device):
        """获取或创建grid"""
        shape_key = (h, w)

        if shape_key not in self.grid_cache:
            coordinates = gen_grid_coordinates_optimized(h, w, self.stride)
            grid = torch.from_numpy(coordinates).to(device)
            grid.requires_grad = False
            self.grid_cache[shape_key] = grid
            self.current_grid_shape = (h, w)

        return self.grid_cache[shape_key]

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]

        if self.current_grid_shape != (h, w):
            grid = self._get_or_create_grid(h, w, x.device)
        else:
            grid = self.grid_cache[(h, w)]

        batch_size = x.shape[0]
        grid_expanded = grid.expand(batch_size, -1, -1, -1)

        return self.pool(F.grid_sample(x, grid_expanded, mode=self.mode, align_corners=True))


# ==================== 导出接口 ====================

# 为了向后兼容，保留原有的类名
SphereConv2D = OptimizedSphereConv2D
SphereMaxPool2D = OptimizedSphereMaxPool2D
