"""
扫描路径评估指标 - 使用官方metrics.py的标准实现
整合了LEV, DTW, REC, ScanMatch, TDE, MultiMatch等指标
"""
import numpy as np
from typing import Tuple, Optional
import editdistance
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr


def scanpath_to_string(scanpath, height_width, Xbins, Ybins, Tbins):
    """
    将扫描路径转换为字符串表示（用于LEV和ScanMatch）

    Args:
        scanpath: (T, 2) numpy array，像素坐标
        height_width: ((0, height), (0, width)) 图像尺寸
        Xbins: X方向的bin数量
        Ybins: Y方向的bin数量
        Tbins: 时间bin（0表示不使用时间信息）

    Returns:
        string: 字符串表示
        num: 数值表示列表
    """
    if Tbins != 0:
        try:
            assert scanpath.shape[1] == 3
        except Exception as x:
            print("Temporal information doesn't exist.")

    height = height_width[0][1]  # 修复：应该是[0][1]而不是[0][0]
    width = height_width[1][1]
    height_step = height // Ybins if Ybins > 0 else 1
    width_step = width // Xbins if Xbins > 0 else 1
    string = ''
    num = list()

    for i in range(scanpath.shape[0]):
        fixation = scanpath[i].astype(np.int32)
        xbin = min(Xbins-1, fixation[0] // width_step)
        xbin = max(0, xbin)
        ybin = min(Ybins-1, ((height - fixation[1]) // height_step))
        ybin = max(0, ybin)
        corrs_x = chr(65 + xbin)
        corrs_y = chr(97 + ybin)
        T = 1
        if Tbins:
            T = fixation[2] // Tbins
        for t in range(T):
            string += (corrs_y + corrs_x)
            num += [ybin * Xbins + xbin]  # 修复：正确的索引计算

    return string, num


def compute_lev(pred_scanpath: np.ndarray, true_scanpath: np.ndarray,
                image_size: Tuple[int, int] = (256, 512),
                Xbins: int = 12, Ybins: int = 8) -> float:
    """
    计算Levenshtein Distance (编辑距离) - 官方实现

    使用网格离散化方法，将连续坐标映射到离散网格

    Args:
        pred_scanpath: 预测路径 (T, 2)，坐标范围[0, 1]，格式为(x, y)
        true_scanpath: 真实路径 (T, 2)，坐标范围[0, 1]，格式为(x, y)
        image_size: 图像尺寸 (height, width)
        Xbins: X方向的bin数量（默认12）
        Ybins: Y方向的bin数量（默认8）

    Returns:
        编辑距离（越小越好）
    """
    h, w = image_size
    height_width = ((0, h), (0, w))

    # 转换为像素坐标
    pred_pixels = pred_scanpath.copy()
    pred_pixels[:, 0] = np.clip(pred_pixels[:, 0] * w, 0, w - 1)
    pred_pixels[:, 1] = np.clip(pred_pixels[:, 1] * h, 0, h - 1)

    true_pixels = true_scanpath.copy()
    true_pixels[:, 0] = np.clip(true_pixels[:, 0] * w, 0, w - 1)
    true_pixels[:, 1] = np.clip(true_pixels[:, 1] * h, 0, h - 1)

    # 转换为字符串
    P, P_num = scanpath_to_string(pred_pixels, height_width, Xbins, Ybins, 0)
    Q, Q_num = scanpath_to_string(true_pixels, height_width, Xbins, Ybins, 0)

    return editdistance.eval(P, Q)


def compute_dtw(pred_scanpath: np.ndarray, true_scanpath: np.ndarray,
                image_size: Tuple[int, int] = (256, 512)) -> float:
    """
    计算Dynamic Time Warping距离 - 官方实现

    使用fastdtw库计算，速度更快

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
    pred_pixels[:, 0] = pred_pixels[:, 0] * w
    pred_pixels[:, 1] = pred_pixels[:, 1] * h

    true_pixels = true_scanpath.copy()
    true_pixels[:, 0] = true_pixels[:, 0] * w
    true_pixels[:, 1] = true_pixels[:, 1] * h

    # 使用fastdtw计算
    dist, _ = fastdtw(pred_pixels, true_pixels, dist=euclidean)

    return dist


def compute_rec(pred_scanpath: np.ndarray, true_scanpath: np.ndarray,
                threshold: float = 12.0, image_size: Tuple[int, int] = (256, 512)) -> float:
    """
    计算Recurrence (交叉重现率) - 官方实现

    基于论文: https://link.springer.com/content/pdf/10.3758%2Fs13428-014-0550-3.pdf

    Args:
        pred_scanpath: 预测路径 (T, 2)，坐标范围[0, 1]，格式为(x, y)
        true_scanpath: 真实路径 (T, 2)，坐标范围[0, 1]，格式为(x, y)
        threshold: 距离阈值（像素），默认12像素（2*6）
        image_size: 图像尺寸 (height, width)

    Returns:
        重现率百分比（0-100，越大越好）
    """
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
    def _C(P, Q, threshold):
        assert P.shape == Q.shape
        shape = P.shape[0]
        c = np.zeros((shape, shape))

        for i in range(shape):
            for j in range(shape):
                if euclidean(P[i], Q[j]) < threshold:
                    c[i, j] = 1
        return c

    c = _C(P, Q, threshold)
    R = np.triu(c, 1).sum()

    return 100 * (2 * R) / (min_len * (min_len - 1)) if min_len > 1 else 0.0


def compute_tde(pred_scanpath: np.ndarray, true_scanpath: np.ndarray,
                k: int = 2, distance_mode: str = 'Mean',
                image_size: Tuple[int, int] = (256, 512)) -> float:
    """
    计算Time-Delay Embedding距离

    参考: https://arxiv.org/abs/1802.02534
    Metric: Simulating Human Saccadic Scanpaths on Natural Images

    Args:
        pred_scanpath: 预测路径 (T1, 2)
        true_scanpath: 真实路径 (T2, 2)
        k: 时间嵌入向量维度
        distance_mode: 'Mean' 或 'Hausdorff'
        image_size: 图像尺寸

    Returns:
        TDE距离（越小越好）
    """
    h, w = image_size

    # 转换为像素坐标
    P = pred_scanpath.copy()
    P[:, 0] = P[:, 0] * w
    P[:, 1] = P[:, 1] * h

    Q = true_scanpath.copy()
    Q[:, 0] = Q[:, 0] * w
    Q[:, 1] = Q[:, 1] * h

    # 检查k是否合理
    if len(P) < k or len(Q) < k:
        return float('inf')

    # 创建时间嵌入向量
    P_vectors = []
    for i in np.arange(0, len(P) - k + 1):
        P_vectors.append(P[i:i + k])

    Q_vectors = []
    for i in np.arange(0, len(Q) - k + 1):
        Q_vectors.append(Q[i:i + k])

    # 计算最小距离
    distances = []
    for s_k_vec in Q_vectors:
        norms = []
        for h_k_vec in P_vectors:
            d = np.linalg.norm(s_k_vec - h_k_vec)
            norms.append(d)
        distances.append(min(norms) / k)

    # 根据distance_mode返回结果
    if distance_mode == 'Mean':
        return sum(distances) / len(distances)
    elif distance_mode == 'Hausdorff':
        return max(distances)
    else:
        return float('inf')


def create_substitution_matrix(Xbin: int, Ybin: int, threshold: float):
    """
    创建ScanMatch的替换矩阵

    Args:
        Xbin: X方向bin数量
        Ybin: Y方向bin数量
        threshold: 距离阈值

    Returns:
        替换矩阵
    """
    # 生成所有网格中心点的坐标
    coords = [(x, y) for y in range(Ybin) for x in range(Xbin)]
    coords = np.array(coords)

    # 计算欧几里得距离矩阵
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    # 转换成替换矩阵
    max_val = np.max(dist_matrix)
    sub_matrix = np.abs(dist_matrix - max_val) - (max_val - threshold)

    return sub_matrix


def scanmatch_nw_algo(intseq1, intseq2, scoring_matrix, gap):
    """
    Needleman-Wunsch算法用于ScanMatch，返回归一化得分

    Args:
        intseq1: 整数序列1
        intseq2: 整数序列2
        scoring_matrix: 得分矩阵
        gap: gap惩罚

    Returns:
        归一化得分 [0, 1]
    """
    m = len(intseq1)
    n = len(intseq2)

    # 边界检查：确保索引在scoring_matrix范围内
    matrix_size = scoring_matrix.shape[0]
    intseq1 = np.array(intseq1)
    intseq2 = np.array(intseq2)
    intseq1 = np.clip(intseq1, 0, matrix_size - 1)
    intseq2 = np.clip(intseq2, 0, matrix_size - 1)

    F = np.zeros((n + 1, m + 1))
    F[1:, 0] = gap * np.arange(1, n + 1)
    F[0, 1:] = gap * np.arange(1, m + 1)

    pointer = np.full((n + 1, m + 1), 4, dtype=np.uint8)
    pointer[:, 0] = 2
    pointer[0, 0] = 1

    currentFColumn = F[:, 0].copy()

    for outer in range(1, m + 1):
        scoredMatchColumn = scoring_matrix[intseq2, intseq1[outer - 1]]
        lastFColumn = currentFColumn.copy()
        currentFColumn = F[:, outer].copy()
        ptr = pointer[:, outer].copy()
        best = currentFColumn[0]

        for inner in range(1, n + 1):
            up = best + gap
            left = lastFColumn[inner] + gap
            diagonal = lastFColumn[inner - 1] + scoredMatchColumn[inner - 1]

            if up > left:
                best = up
                pos = 2
            else:
                best = left
                pos = 4

            if diagonal >= best:
                best = diagonal
                ptr[inner] = 1
            else:
                ptr[inner] = pos

            currentFColumn[inner] = best

        F[:, outer] = currentFColumn
        pointer[:, outer] = ptr

    score = F[n, m]

    # 归一化得分
    max_score = min(m, n) * np.max(np.diag(scoring_matrix))
    normalized_score = score / max_score if max_score != 0 else 0.0

    return normalized_score


def compute_scanmatch(pred_scanpath: np.ndarray, true_scanpath: np.ndarray,
                      image_size: Tuple[int, int] = (256, 512),
                      Xbins: int = 12, Ybins: int = 8, threshold: float = 3.5) -> float:
    """
    计算ScanMatch得分

    Args:
        pred_scanpath: 预测路径 (T, 2)，坐标范围[0, 1]
        true_scanpath: 真实路径 (T, 2)，坐标范围[0, 1]
        image_size: 图像尺寸
        Xbins: X方向bin数量
        Ybins: Y方向bin数量
        threshold: 替换矩阵阈值

    Returns:
        ScanMatch得分 [0, 1]（越大越好）
    """
    h, w = image_size
    height_width = ((0, h), (0, w))

    # 转换为像素坐标
    P = pred_scanpath.copy()
    P[:, 0] = np.clip(P[:, 0] * w, 0, w - 1)
    P[:, 1] = np.clip(P[:, 1] * h, 0, h - 1)

    Q = true_scanpath.copy()
    Q[:, 0] = np.clip(Q[:, 0] * w, 0, w - 1)
    Q[:, 1] = np.clip(Q[:, 1] * h, 0, h - 1)

    # 转换为字符串和数值
    P_str, P_num = scanpath_to_string(P, height_width, Xbins, Ybins, 0)
    Q_str, Q_num = scanpath_to_string(Q, height_width, Xbins, Ybins, 0)

    # 创建替换矩阵
    submatrix = create_substitution_matrix(Xbins, Ybins, threshold)

    # 计算ScanMatch得分
    return scanmatch_nw_algo(P_num, Q_num, submatrix, 0)


def compute_multimatch_metrics(pred_scanpath: np.ndarray, true_scanpath: np.ndarray,
                                image_size: Tuple[int, int] = (256, 512)) -> dict:
    """
    计算MultiMatch风格的指标

    MultiMatch是眼动研究中常用的评估方法，包含多个维度：
    - Vector: 方向相似度（余弦相似度）
    - Length: 步长相似度（Pearson相关系数）
    - Position: 位置相似度

    Args:
        pred_scanpath: 预测路径 (T, 2)，坐标范围[0, 1]
        true_scanpath: 真实路径 (T, 2)，坐标范围[0, 1]
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

    # 使用Pearson相关系数
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
        from scipy.ndimage import zoom
        scale_h = h / saliency_map.shape[0]
        scale_w = w / saliency_map.shape[1]
        saliency_map = zoom(saliency_map, (scale_h, scale_w), order=1)

    # 1. NSS (Normalized Scanpath Saliency)
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
    fixation_map = np.zeros((h, w))
    for i in range(len(pred_pixels)):
        y, x = pred_pixels[i, 1], pred_pixels[i, 0]
        y = min(y, h - 1)
        x = min(x, w - 1)
        sigma = min(h, w) * 0.02
        y_range = range(max(0, int(y - 3 * sigma)), min(h, int(y + 3 * sigma + 1)))
        x_range = range(max(0, int(x - 3 * sigma)), min(w, int(x + 3 * sigma + 1)))
        for yy in y_range:
            for xx in x_range:
                dist_sq = (yy - y) ** 2 + (xx - x) ** 2
                fixation_map[yy, xx] += np.exp(-dist_sq / (2 * sigma ** 2))

    if fixation_map.sum() > 0:
        fixation_map = fixation_map / fixation_map.sum()
    if saliency_map.sum() > 0:
        saliency_map_norm = saliency_map / saliency_map.sum()
    else:
        saliency_map_norm = saliency_map

    if np.std(fixation_map) > 0 and np.std(saliency_map_norm) > 0:
        cc = float(pearsonr(fixation_map.flatten(), saliency_map_norm.flatten())[0])
    else:
        cc = 0.0

    # 3. Saliency Coverage
    sal_threshold = np.percentile(saliency_map, 80)
    salient_regions = saliency_map > sal_threshold

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


def compute_all_metrics(pred_scanpath: np.ndarray, true_scanpath: np.ndarray,
                        image_size: Tuple[int, int] = (256, 512),
                        saliency_map: Optional[np.ndarray] = None) -> dict:
    """
    计算所有评估指标（使用官方实现）

    Args:
        pred_scanpath: 预测路径 (T, 2)，坐标范围[0, 1]，格式为(x, y)
        true_scanpath: 真实路径 (T, 2)，坐标范围[0, 1]，格式为(x, y)
        image_size: 图像尺寸 (height, width)
        saliency_map: 显著性图 (H, W)，可选

    Returns:
        包含所有指标的字典
    """
    metrics = {}

    # 基础指标（官方实现）
    metrics['LEV'] = compute_lev(pred_scanpath, true_scanpath, image_size)
    metrics['DTW'] = compute_dtw(pred_scanpath, true_scanpath, image_size)
    metrics['REC'] = compute_rec(pred_scanpath, true_scanpath, threshold=12.0, image_size=image_size)

    # ScanMatch
    metrics['ScanMatch'] = compute_scanmatch(pred_scanpath, true_scanpath, image_size)

    # TDE
    try:
        metrics['TDE'] = compute_tde(pred_scanpath, true_scanpath, k=2, image_size=image_size)
    except:
        metrics['TDE'] = float('inf')

    # MultiMatch指标
    mm_metrics = compute_multimatch_metrics(pred_scanpath, true_scanpath, image_size)
    metrics.update(mm_metrics)

    # 显著性指标（如果提供了显著性图）
    if saliency_map is not None:
        sal_metrics = compute_scanpath_saliency_metrics(pred_scanpath, saliency_map, image_size)
        metrics.update(sal_metrics)

    return metrics


# 保持向后兼容的别名
compute_all_metrics_extended = compute_all_metrics
