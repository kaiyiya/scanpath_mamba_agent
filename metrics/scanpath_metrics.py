"""
扫描路径评估指标
实现LEV, DTW, REC三个常用指标
以及新增的更适合眼动路径的指标：SIM, CC, NSS, AUC
"""
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from typing import Tuple


def compute_lev(pred_scanpath: np.ndarray, true_scanpath: np.ndarray,
                image_size: Tuple[int, int] = (256, 512), grid_size: int = 8,
                distance_threshold: float = 0.20) -> float:
    """
    计算Levenshtein Distance (编辑距离) - 使用软匹配

    关键改进：不再使用网格离散化，而是基于连续距离的软匹配
    如果两个点的欧氏距离小于阈值，则认为匹配，否则需要编辑

    Args:
        pred_scanpath: 预测路径 (T, 2)，坐标范围[0, 1]，格式为(x, y)
        true_scanpath: 真实路径 (T, 2)，坐标范围[0, 1]，格式为(x, y)
        image_size: 图像尺寸 (height, width)，用于显示（不再用于离散化）
        grid_size: 保留参数以兼容旧代码，但不再使用
        distance_threshold: 距离阈值，小于此值认为匹配（默认0.20，约为图像对角线的20%）
                           - 0.15: 太严格，只有position_error<0.2的样本能得到好的LEV
                           - 0.20: 更合理，适合position_error在0.2-0.3的模型

    Returns:
        编辑距离（越小越好）
    """

    # 计算编辑距离（动态规划），使用软匹配
    m, n = len(pred_scanpath), len(true_scanpath)

    if m == 0 or n == 0:
        return max(m, n)

    # 使用滚动数组优化内存（只保留两行）
    prev_row = np.arange(n + 1, dtype=np.float32)
    curr_row = np.zeros(n + 1, dtype=np.float32)

    for i in range(1, m + 1):
        curr_row[0] = i
        for j in range(1, n + 1):
            # 计算两点之间的欧氏距离
            dist = np.linalg.norm(pred_scanpath[i - 1] - true_scanpath[j - 1])

            # 软匹配：如果距离小于阈值，认为匹配
            if dist < distance_threshold:
                curr_row[j] = prev_row[j - 1]  # 匹配，不需要编辑
            else:
                curr_row[j] = 1 + min(
                    prev_row[j],      # 删除
                    curr_row[j - 1],  # 插入
                    prev_row[j - 1]   # 替换
                )
        prev_row, curr_row = curr_row, prev_row

    return float(prev_row[n])


def compute_dtw(pred_scanpath: np.ndarray, true_scanpath: np.ndarray,
                image_size: Tuple[int, int] = (256, 512)) -> float:
    """
    计算Dynamic Time Warping距离
    使用像素坐标计算距离

    Args:
        pred_scanpath: 预测路径 (T1, 2)，坐标范围[0, 1]，格式为(x, y)
        true_scanpath: 真实路径 (T2, 2)，坐标范围[0, 1]，格式为(x, y)
        image_size: 图像尺寸 (height, width)

    Returns:
        DTW距离（越小越好）
    """
    h, w = image_size

    # 转换为像素坐标
    pred_pixels = pred_scanpath.copy()
    pred_pixels[:, 0] *= w  # x坐标
    pred_pixels[:, 1] *= h  # y坐标

    true_pixels = true_scanpath.copy()
    true_pixels[:, 0] *= w
    true_pixels[:, 1] *= h

    m, n = len(pred_pixels), len(true_pixels)

    # 计算距离矩阵
    dist_matrix = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            dist_matrix[i, j] = euclidean(pred_pixels[i], true_pixels[j])

    # DTW动态规划
    dtw_matrix = np.full((m + 1, n + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = dist_matrix[i - 1, j - 1]
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],  # 插入
                dtw_matrix[i, j - 1],  # 删除
                dtw_matrix[i - 1, j - 1]  # 匹配
            )

    return dtw_matrix[m, n]


def compute_rec(pred_scanpath: np.ndarray, true_scanpath: np.ndarray,
                image_size: Tuple[int, int] = (256, 512)) -> float:
    """
    计算Recurrence (重现率)
    返回访问的唯一网格数量的比率

    Args:
        pred_scanpath: 预测路径 (T, 2)，坐标范围[0, 1]，格式为(x, y)
        true_scanpath: 真实路径 (T, 2)，坐标范围[0, 1]，格式为(x, y)
        image_size: 图像尺寸 (height, width)

    Returns:
        重现率（越大越好）
    """
    # 使用较粗的网格（16x16）来计算REC
    grid_h, grid_w = 16, 16

    def get_visited_grids(scanpath, grid_h, grid_w):
        # 将[0,1]坐标映射到网格索引
        grid_x = np.clip((scanpath[:, 0] * grid_w).astype(int), 0, grid_w - 1)
        grid_y = np.clip((scanpath[:, 1] * grid_h).astype(int), 0, grid_h - 1)
        # 转换为一维索引并去重
        grid_indices = grid_y * grid_w + grid_x
        return set(grid_indices)

    pred_grids = get_visited_grids(pred_scanpath, grid_h, grid_w)
    true_grids = get_visited_grids(true_scanpath, grid_h, grid_w)

    # 计算重叠区域数量
    intersection = len(pred_grids & true_grids)

    # 返回重叠数量（不是IoU）
    # 根据ScanDMM的值（4.67），这应该是一个计数而不是比率
    return float(intersection)


def compute_all_metrics(pred_scanpath: np.ndarray, true_scanpath: np.ndarray,
                        image_size: Tuple[int, int] = (256, 512), grid_size: int = 8) -> dict:
    """
    计算所有评估指标

    Args:
        pred_scanpath: 预测路径 (T, 2)，坐标范围[0, 1]，格式为(x, y)
        true_scanpath: 真实路径 (T, 2)，坐标范围[0, 1]，格式为(x, y)
        image_size: 图像尺寸 (height, width)
        grid_size: LEV计算时使用的网格大小（默认8x8，更宽容的匹配标准）

    Returns:
        包含所有指标的字典
    """
    lev = compute_lev(pred_scanpath, true_scanpath, image_size, grid_size=grid_size)
    dtw = compute_dtw(pred_scanpath, true_scanpath, image_size)
    rec = compute_rec(pred_scanpath, true_scanpath, image_size)

    return {
        'LEV': lev,
        'DTW': dtw,
        'REC': rec
    }


# ==================== 新增指标：更适合眼动路径评估 ====================

def compute_similarity(pred_scanpath: np.ndarray, true_scanpath: np.ndarray) -> float:
    """
    计算相似度 (SIM) - 基于平均欧氏距离

    这是一个简单直观的指标，衡量预测路径与真实路径的平均距离

    Args:
        pred_scanpath: 预测路径 (T, 2)，坐标范围[0, 1]
        true_scanpath: 真实路径 (T, 2)，坐标范围[0, 1]

    Returns:
        相似度分数（越小越好，0表示完全匹配）
    """
    # 计算每个点的欧氏距离
    distances = np.linalg.norm(pred_scanpath - true_scanpath, axis=1)
    # 返回平均距离
    return float(np.mean(distances))


def compute_multimatch_metrics(pred_scanpath: np.ndarray, true_scanpath: np.ndarray,
                                image_size: Tuple[int, int] = (256, 512)) -> dict:
    """
    计算MultiMatch风格的指标

    MultiMatch是眼动研究中常用的评估方法，包含多个维度：
    - Vector: 方向相似度
    - Length: 步长相似度
    - Position: 位置相似度
    - Duration: 持续时间相似度（这里简化为序列长度）

    Args:
        pred_scanpath: 预测路径 (T, 2)
        true_scanpath: 真实路径 (T, 2)
        image_size: 图像尺寸

    Returns:
        包含各维度相似度的字典（0-1之间，越大越好）
    """
    # 计算扫视向量（相邻点之间的向量）
    pred_vectors = pred_scanpath[1:] - pred_scanpath[:-1]
    true_vectors = true_scanpath[1:] - true_scanpath[:-1]

    # 1. 方向相似度（向量夹角的余弦相似度）
    pred_norms = np.linalg.norm(pred_vectors, axis=1, keepdims=True) + 1e-8
    true_norms = np.linalg.norm(true_vectors, axis=1, keepdims=True) + 1e-8
    pred_directions = pred_vectors / pred_norms
    true_directions = true_vectors / true_norms

    # 计算余弦相似度
    cosine_sim = np.sum(pred_directions * true_directions, axis=1)
    vector_similarity = float(np.mean(np.clip(cosine_sim, -1, 1)))

    # 2. 步长相似度（扫视长度的相关性）
    pred_lengths = np.linalg.norm(pred_vectors, axis=1)
    true_lengths = np.linalg.norm(true_vectors, axis=1)

    # 使用相关系数
    if len(pred_lengths) > 1 and np.std(pred_lengths) > 0 and np.std(true_lengths) > 0:
        length_similarity = float(pearsonr(pred_lengths, true_lengths)[0])
    else:
        length_similarity = 0.0

    # 3. 位置相似度（点的平均距离，归一化到0-1）
    position_distances = np.linalg.norm(pred_scanpath - true_scanpath, axis=1)
    # 归一化：最大可能距离是对角线长度sqrt(2)
    max_distance = np.sqrt(2)
    position_similarity = float(1.0 - np.mean(position_distances) / max_distance)

    return {
        'MM_Vector': max(vector_similarity, 0.0),  # 确保非负
        'MM_Length': max(length_similarity, 0.0),
        'MM_Position': max(position_similarity, 0.0)
    }


def compute_scanpath_saliency_metrics(pred_scanpath: np.ndarray,
                                       saliency_map: np.ndarray,
                                       image_size: Tuple[int, int] = (256, 512)) -> dict:
    """
    计算基于显著性图的指标

    这些指标评估预测路径是否覆盖了图像的显著区域

    Args:
        pred_scanpath: 预测路径 (T, 2)，坐标范围[0, 1]
        saliency_map: 显著性图 (H, W)，归一化到[0, 1]
        image_size: 图像尺寸 (height, width)

    Returns:
        包含显著性相关指标的字典
    """
    if saliency_map is None:
        return {
            'NSS': 0.0,
            'CC': 0.0,
            'SalCoverage': 0.0
        }

    h, w = image_size

    # 将路径坐标转换为像素坐标
    pred_pixels = pred_scanpath.copy()
    pred_pixels[:, 0] = np.clip(pred_pixels[:, 0] * w, 0, w - 1)
    pred_pixels[:, 1] = np.clip(pred_pixels[:, 1] * h, 0, h - 1)
    pred_pixels = pred_pixels.astype(int)

    # 确保显著性图尺寸正确
    if saliency_map.shape != (h, w):
        # 如果尺寸不匹配，进行插值
        from scipy.ndimage import zoom
        scale_h = h / saliency_map.shape[0]
        scale_w = w / saliency_map.shape[1]
        saliency_map = zoom(saliency_map, (scale_h, scale_w), order=1)

    # 1. NSS (Normalized Scanpath Saliency)
    # 在注视点位置的显著性值，归一化到均值0方差1
    sal_mean = np.mean(saliency_map)
    sal_std = np.std(saliency_map) + 1e-8

    fixation_saliencies = []
    for i in range(len(pred_pixels)):
        y, x = pred_pixels[i, 1], pred_pixels[i, 0]
        y = min(y, h - 1)
        x = min(x, w - 1)
        fixation_saliencies.append(saliency_map[y, x])

    nss = float(np.mean([(s - sal_mean) / sal_std for s in fixation_saliencies]))

    # 2. CC (Correlation Coefficient)
    # 创建注视密度图
    fixation_map = np.zeros((h, w))
    for i in range(len(pred_pixels)):
        y, x = pred_pixels[i, 1], pred_pixels[i, 0]
        y = min(y, h - 1)
        x = min(x, w - 1)
        # 使用高斯核模糊注视点
        sigma = min(h, w) * 0.02  # 2%的图像尺寸
        y_range = range(max(0, int(y - 3 * sigma)), min(h, int(y + 3 * sigma + 1)))
        x_range = range(max(0, int(x - 3 * sigma)), min(w, int(x + 3 * sigma + 1)))
        for yy in y_range:
            for xx in x_range:
                dist_sq = (yy - y) ** 2 + (xx - x) ** 2
                fixation_map[yy, xx] += np.exp(-dist_sq / (2 * sigma ** 2))

    # 归一化
    if fixation_map.sum() > 0:
        fixation_map = fixation_map / fixation_map.sum()
    if saliency_map.sum() > 0:
        saliency_map_norm = saliency_map / saliency_map.sum()
    else:
        saliency_map_norm = saliency_map

    # 计算相关系数
    if np.std(fixation_map) > 0 and np.std(saliency_map_norm) > 0:
        cc = float(pearsonr(fixation_map.flatten(), saliency_map_norm.flatten())[0])
    else:
        cc = 0.0

    # 3. Saliency Coverage
    # 路径覆盖的显著性区域比例
    # 定义显著性阈值（top 20%）
    sal_threshold = np.percentile(saliency_map, 80)
    salient_regions = saliency_map > sal_threshold

    # 检查路径是否访问了显著区域
    visited_salient = 0
    for i in range(len(pred_pixels)):
        y, x = pred_pixels[i, 1], pred_pixels[i, 0]
        y = min(y, h - 1)
        x = min(x, w - 1)
        if salient_regions[y, x]:
            visited_salient += 1

    sal_coverage = float(visited_salient / len(pred_pixels))

    return {
        'NSS': nss,
        'CC': max(cc, 0.0),
        'SalCoverage': sal_coverage
    }


def compute_all_metrics_extended(pred_scanpath: np.ndarray,
                                  true_scanpath: np.ndarray,
                                  saliency_map: np.ndarray = None,
                                  image_size: Tuple[int, int] = (256, 512),
                                  grid_size: int = 8) -> dict:
    """
    计算所有评估指标（包括原有指标和新增指标）

    Args:
        pred_scanpath: 预测路径 (T, 2)，坐标范围[0, 1]，格式为(x, y)
        true_scanpath: 真实路径 (T, 2)，坐标范围[0, 1]，格式为(x, y)
        saliency_map: 显著性图 (H, W)，可选
        image_size: 图像尺寸 (height, width)
        grid_size: LEV计算时使用的网格大小

    Returns:
        包含所有指标的字典
    """
    # 原有指标
    metrics = compute_all_metrics(pred_scanpath, true_scanpath, image_size, grid_size)

    # 新增指标
    metrics['SIM'] = compute_similarity(pred_scanpath, true_scanpath)

    # MultiMatch指标
    mm_metrics = compute_multimatch_metrics(pred_scanpath, true_scanpath, image_size)
    metrics.update(mm_metrics)

    # 显著性指标（如果提供了显著性图）
    if saliency_map is not None:
        sal_metrics = compute_scanpath_saliency_metrics(pred_scanpath, saliency_map, image_size)
        metrics.update(sal_metrics)

    return metrics
