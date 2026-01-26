import editdistance
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from multimatch import docomparison
import re



def scanpath_to_string(scanpath, height_width, Xbins, Ybins, Tbins):
    if Tbins != 0:
        try:
            assert scanpath.shape[1] == 3
        except Exception as x:
            print("Temporal information doesn't exist.")
    height = height_width[0][0]
    width = height_width[1][1]
    height_step, width_step = height // Ybins, width // Xbins
    string = ''
    num = list()
    for i in range(scanpath.shape[0]):
        # fixation = scanpath[i]
        # fixation2 = np.multiply(fixation, height_width)
        # fixation = fixation2.astype(np.int32)
        fixation = scanpath[i].astype(np.int32)
        xbin = min(Xbins-1,fixation[0] // width_step)
        xbin = max(0,fixation[0] // width_step)
        ybin = min(Ybins-1, ((height - fixation[1]) // height_step))
        ybin = max(0, ((height - fixation[1]) // height_step))
        corrs_x = chr(65 + xbin)
        corrs_y = chr(97 + ybin)
        T = 1
        if Tbins:
            T = fixation[2] // Tbins
        for t in range(T):
            string += (corrs_y + corrs_x)
            num += [(ybin * (Xbins-1)) + xbin - 1 ]
    return string, num

def string_to_number(s: str, modulus: int):
    """
    Transforms a string sequence of double characters into single numbers
    based on the modulus used. The modulus is the length of the alphabet
    used for the second letter of a residue.

    Example:
        ScanMatch_DoubleStrToNum('aAaBbA', 26)  # returns [1, 2, 27]
        ScanMatch_DoubleStrToNum('aAaBbA', 10)  # returns [1, 2, 11]

    Parameters:
    - s: input string (must have even length)
    - modulus: integer between 1 and 26

    Returns:
    - list of integers
    """

    if modulus < 1 or modulus > 26:
        raise ValueError("The modulus has to be between 1 and 26!")

    s = s.upper()
    if len(s) % 2 != 0:
        raise ValueError("The number of characters in the input string needs to be even...")

    res = []
    for i in range(0, len(s), 2):
        # convert letters to numbers: 'A' -> 1, 'B' -> 2, ..., 'Z' -> 26
        num1 = ord(s[i]) - 64
        num2 = ord(s[i + 1]) - 64
        val = (num1 - 1) * modulus + (num2 - 1)   # +1 to match MATLAB indexing
        res.append(val)

    return res

def levenshtein_distance(P, Q, height_width, Xbins=12, Ybins=8, **kwargs):
    """
                Levenshtein distance
        """
    P, P_num = scanpath_to_string(P, height_width, Xbins, Ybins, 0)
    Q, Q_num = scanpath_to_string(Q, height_width, Xbins, Ybins, 0)

    return editdistance.eval(P, Q)


def DTW(P, Q, **kwargs):
    dist, _ = fastdtw(P, Q, dist=euclidean)
    return dist


def REC(P, Q, threshold, **kwargs):
    """
                Cross-recurrence
                https://link.springer.com/content/pdf/10.3758%2Fs13428-014-0550-3.pdf
        """

    def _C(P, Q, threshold):
        assert (P.shape == Q.shape)
        shape = P.shape[0]
        c = np.zeros((shape, shape))

        for i in range(shape):
            for j in range(shape):
                if euclidean(P[i], Q[j]) < threshold:
                    c[i, j] = 1
        return c

    P = np.array(P, dtype=np.float32)
    Q = np.array(Q, dtype=np.float32)
    min_len = P.shape[0] if (P.shape[0] < Q.shape[0]) else Q.shape[0]
    P = P[:min_len, :2]
    Q = Q[:min_len, :2]

    c = _C(P, Q, threshold)
    R = np.triu(c, 1).sum()
    return 100 * (2 * R) / (min_len * (min_len - 1))

def euclidean_distance(P, Q, **kwargs):
    if not isinstance(P, np.ndarray):
        P = np.array(P, dtype=np.float32)
    elif P.dtype != np.float32:
        P = P.astype(np.float32)

    if not isinstance(Q, np.ndarray):
        Q = np.array(Q, dtype=np.float32)
    elif Q.dtype != np.float32:
        Q = Q.astype(np.float32)

    if P.shape == Q.shape:
        return np.sqrt(np.sum((P - Q) ** 2))
    return False


def TDE(
        P,
        Q,

        # options
        k=2,  # time-embedding vector dimension
        distance_mode='Mean', **kwargs
):
    """
        code reference:
            https://github.com/dariozanca/FixaTons/
            https://arxiv.org/abs/1802.02534

        metric: Simulating Human Saccadic Scanpaths on Natural Images.
                 wei wang etal.
    """

    # P and Q can have different lenghts
    # They are list of fixations, that is couple of coordinates
    # k must be shorter than both lists lenghts

    # we check for k be smaller or equal then the lenghts of the two input scanpaths
    if len(P) < k or len(Q) < k:
        # print('ERROR: Too large value for the time-embedding vector dimension')
        return False

    # create time-embedding vectors for both scanpaths

    P_vectors = []
    for i in np.arange(0, len(P) - k + 1):
        P_vectors.append(P[i:i + k])

    Q_vectors = []
    for i in np.arange(0, len(Q) - k + 1):
        Q_vectors.append(Q[i:i + k])

    # in the following cicles, for each k-vector from the simulated scanpath
    # we look for the k-vector from humans, the one of minumum distance
    # and we save the value of such a distance, divided by k

    distances = []

    for s_k_vec in Q_vectors:

        # find human k-vec of minimum distance

        norms = []

        for h_k_vec in P_vectors:
            d = np.linalg.norm(euclidean_distance(s_k_vec, h_k_vec))
            norms.append(d)

        distances.append(min(norms) / k)

    # at this point, the list "distances" contains the value of
    # minumum distance for each simulated k-vec
    # according to the distance_mode, here we compute the similarity
    # between the two scanpaths.

    if distance_mode == 'Mean':
        return sum(distances) / len(distances)
    elif distance_mode == 'Hausdorff':
        return max(distances)
    else:
        print('ERROR: distance mode not defined.')
        return False
    
    
def DET(P, Q, threshold, **kwargs):
    """
                https://link.springer.com/content/pdf/10.3758%2Fs13428-014-0550-3.pdf
        """

    def _C(P, Q, threshold):
        assert (P.shape == Q.shape)
        shape = P.shape[0]
        c = np.zeros((shape, shape))

        for i in range(shape):
            for j in range(shape):
                if euclidean(Q[i], P[j]) < threshold:
                    c[i, j] = 1
        return c

    P = np.array(P, dtype=np.float32)
    Q = np.array(Q, dtype=np.float32)
    min_len = P.shape[0] if (P.shape[0] < Q.shape[0]) else Q.shape[0]
    P = P[:min_len, :2]
    Q = Q[:min_len, :2]

    c = _C(P, Q, threshold)
    R = np.triu(c, 1).sum()

    counter = 0
    for i in range(1, min_len):
        data = c.diagonal(i)
        data = ''.join([str(item) for item in data])
        counter += len(re.findall('1{2,}', data))

    return 100 * (counter / R)

def MAE(y_true, y_pred):
    """
    计算实际值与预测值之间的平均绝对误差（MAE）

    参数:
    y_true (list or array-like): 实际值
    y_pred (list or array-like): 预测值

    返回:
    float: 平均绝对误差
    """
    # 确保输入是数组格式
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 计算绝对误差
    abs_error = np.abs(y_true - y_pred)
    
    # 计算平均绝对误差
    mae = np.mean(abs_error)
    
    return mae

def compute_scanpath_mertics(pred_scanpath, gt_scanpath, height_width, gt_lengths):
    # assert pred_scanpath.shape == gt_scanpath.shape
    lev, dtw, rec, det = [], [], [], []
    num_samples = pred_scanpath.shape[1]
    num_gt_samples = gt_scanpath.shape[1]
    for user_i in range(num_samples):
        pred_scanpath_i = np.dot(pred_scanpath[:, user_i, :],height_width)
        # for user_j in range(1):  #
        for user_j in range(num_gt_samples):  #
            # length = min(lengths[user_j], 15)
            length = gt_lengths[user_j]
            gt_scanpath_j = np.dot(gt_scanpath[:length, user_j, :],height_width)
            lev.append(levenshtein_distance(
                pred_scanpath_i[:length], gt_scanpath_j, height_width))
            dtw.append(DTW(pred_scanpath_i[:length], gt_scanpath_j))
            rec.append(REC(pred_scanpath_i[:length], gt_scanpath_j, 2 * 6))
    LEV_res = np.nanmean(lev)
    DTW_res = np.nanmean(dtw)
    REC_res = np.nanmean(rec)
    return LEV_res, DTW_res, REC_res


def compute_scanpath_mertics_V2(pred_scanpath, gt_scanpath, height, width, lengths):
    # assert pred_scanpath.shape == gt_scanpath.shape
    lev, dtw, rec, det = [], [], [], []
    num_samples = gt_scanpath.shape[0]
    for user_i in range(num_samples):
        pred_scanpath_i = pred_scanpath[user_i, :, :]
        # for user_j in range(1):  #
        for user_j in range(num_samples):  #
            # length = min(lengths[user_j], 15)
            length = lengths[user_j]
            gt_scanpath_j = gt_scanpath[user_j, :length, :]
            lev.append(levenshtein_distance(
                pred_scanpath_i[:length], gt_scanpath_j, height, width))
            dtw.append(DTW(pred_scanpath_i[:length], gt_scanpath_j))
            rec.append(REC(pred_scanpath_i[:length], gt_scanpath_j, 2 * 6))
            det.append(DET(pred_scanpath_i[:length], gt_scanpath_j, 2 * 6))
    LEV_res = np.nanmean(lev)
    DTW_res = np.nanmean(dtw)
    REC_res = np.nanmean(rec)
    DET_res = np.nanmean(det)
    return LEV_res, DTW_res, REC_res, DET_res


def compute_scanpath_mertics_V3(pred_scanpath, gt_scanpath, height_width, gt_lengths):
    # assert pred_scanpath.shape == gt_scanpath.shape
    lev, dtw, rec = [], [], []
    num_gt_samples = gt_scanpath.shape[0]

    for user_i in range(num_gt_samples):
        pred_scanpath_i = np.dot(pred_scanpath[user_i, :, :],height_width)
        # for user_j in range(1):  #
        for user_j in range(num_gt_samples):  #
            # length = min(lengths[user_j], 15)
            length = min(gt_lengths[user_j],pred_scanpath.shape[1])
            gt_scanpath_j = np.dot(gt_scanpath[user_j, :length, :],height_width)
            lev.append(levenshtein_distance(
                pred_scanpath_i[:length], gt_scanpath_j, height_width))
            dtw.append(DTW(pred_scanpath_i[:length], gt_scanpath_j))
            rec.append(REC(pred_scanpath_i[:length], gt_scanpath_j, 2 * 6))
            

    LEV_res = np.nanmean(lev)
    DTW_res = np.nanmean(dtw)
    REC_res = np.nanmean(rec)


    return LEV_res, DTW_res, REC_res

def compute_scanpath_mertics_AOI(pred_scanpath, gt_scanpath, height_width, gt_lengths):
    # AOI的gt是list格式，且长度不定
    lev, dtw, rec= [], [], []
    num_gt_samples = len(gt_scanpath)

    for user_i in range(num_gt_samples):
        pred_scanpath_i = np.dot(pred_scanpath[user_i, :, :],height_width)
        # for user_j in range(1):  #
        for user_j in range(num_gt_samples):  #
            # length = min(lengths[user_j], 15)
            length = min(gt_lengths[user_j],pred_scanpath.shape[1])
            gt_scanpath_j = gt_scanpath[user_j].numpy()
            gt_scanpath_j = np.dot(gt_scanpath_j[:length, :],height_width)
            lev.append(levenshtein_distance(
                pred_scanpath_i[:length], gt_scanpath_j, height_width))
            dtw.append(DTW(pred_scanpath_i[:length], gt_scanpath_j))
            rec.append(REC(pred_scanpath_i[:length], gt_scanpath_j, 2 * 6))

    LEV_res = np.nanmean(lev)
    DTW_res = np.nanmean(dtw)
    REC_res = np.nanmean(rec)

    return LEV_res, DTW_res, REC_res

#add
def compute_scanpath_mertics_V4(pred_scanpath, gt_scanpath, height_width, gt_lengths):
    # assert pred_scanpath.shape == gt_scanpath.shape
    lev, dtw, tde, sam, mm, ss = [], [], [], [], [], []
    num_gt_samples = gt_scanpath.shape[0]

    for user_i in range(num_gt_samples):
        pred_scanpath_i = np.dot(pred_scanpath[user_i, :, :],height_width)
        # for user_j in range(1):  #
        for user_j in range(num_gt_samples):  #
            # length = min(lengths[user_j], 15)
            length = min(gt_lengths[user_j],pred_scanpath.shape[1])
            gt_scanpath_j = np.dot(gt_scanpath[user_j, :length, :],height_width)
            # lev.append(levenshtein_distance(
            #     pred_scanpath_i[:length], gt_scanpath_j, height_width))
            # dtw.append(DTW(pred_scanpath_i[:length], gt_scanpath_j))
            # rec.append(REC(pred_scanpath_i[:length], gt_scanpath_j, 2 * 6))
            sam.append(scanmatch(pred_scanpath_i[:length], gt_scanpath_j, height_width))
            # mm.append(multimatch(pred_scanpath_i[:length],gt_scanpath_j,[256,128]))
            tde.append(TDE(pred_scanpath_i[:length],gt_scanpath_j))
            # det.append(DET(pred_scanpath_i[:length], gt_scanpath_j, 120))
    # LEV_res = np.nanmean(lev)
    # DTW_res = np.nanmean(dtw)
    # REC_res = np.nanmean(rec)
    SAM_res = np.nanmean(sam)
    TDE_res = np.nanmean(tde)
    # MM_res = np.nanmean(mm)

    return SAM_res,TDE_res


#add
def compute_scanpath_mertics_AOI2(pred_scanpath, gt_scanpath, height_width, gt_lengths):
    # AOI的gt是list格式，且长度不定
    lev, dtw, tde, sam, mm, ss = [], [], [], [], [], []
    num_gt_samples = len(gt_scanpath)

    for user_i in range(num_gt_samples):
        pred_scanpath_i = np.dot(pred_scanpath[user_i, :, :],height_width)
        # for user_j in range(1):  #
        for user_j in range(num_gt_samples):  #
            # length = min(lengths[user_j], 15)
            length = min(gt_lengths[user_j],pred_scanpath.shape[1])
            gt_scanpath_j = gt_scanpath[user_j].numpy()
            gt_scanpath_j = np.dot(gt_scanpath_j[:length, :],height_width)
            sam.append(scanmatch(pred_scanpath_i[:length], gt_scanpath_j, height_width))
            tde.append(TDE(pred_scanpath_i[:length],gt_scanpath_j))
    SAM_res = np.nanmean(sam)
    TDE_res = np.nanmean(tde)

    return SAM_res,TDE_res

def compute_scanpath_mertics_AOI3(pred_scanpath, gt_scanpath, height_width, gt_lengths):
    # AOI的gt是list格式，且长度不定
    lev, dtw, tde, sam, mm, ss = [], [], [], [], [], []
    num_gt_samples = len(gt_scanpath)

    for user_i in range(num_gt_samples):
        pred_scanpath_i = np.dot(pred_scanpath[user_i].numpy(),height_width)
        # for user_j in range(1):  #
        for user_j in range(num_gt_samples):  #
            # length = min(lengths[user_j], 15)
            length = min(gt_lengths[user_j],gt_lengths[user_i])
            gt_scanpath_j = gt_scanpath[user_j].numpy()
            gt_scanpath_j = np.dot(gt_scanpath_j[:length, :],height_width)
            sam.append(scanmatch(pred_scanpath_i[:length], gt_scanpath_j, height_width))
            tde.append(TDE(pred_scanpath_i[:length],gt_scanpath_j))
    SAM_res = np.nanmean(sam)
    TDE_res = np.nanmean(tde)

    return SAM_res,TDE_res

#add
def scanmatch(P, Q, height_width, Xbins=12, Ybins=8):
    P, P_num = scanpath_to_string(P, height_width, Xbins, Ybins, 0)
    Q, Q_num = scanpath_to_string(Q, height_width, Xbins, Ybins, 0)
    submatrix = create_substitution_matrix(Xbins,Ybins,3.5)

    return scanmatch_nw_algo(P_num, Q_num, submatrix, 0)

#add
def scanmatch_nw_algo(intseq1, intseq2, scoring_matrix, gap):
    """
    Needleman-Wunsch algorithm for ScanMatch with normalized score output.
    
    Returns:
    - score: raw alignment score
    - normalized_score: score normalized to [0, 1]
    - path: list of (j, i) index pairs for backtracking path
    - step: total number of steps in path
    - F: dynamic programming matrix
    """
    m = len(intseq1)
    n = len(intseq2)
    
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

    # Traceback
    i, j = n, m
    path = []
    step = 0

    while i > 0 or j > 0:
        direction = pointer[i, j]
        if direction == 1:
            i -= 1
            j -= 1
            path.append((j, i))
        elif direction == 2:
            i -= 1
            path.append((j, i))
        elif direction == 4 or direction == 6:
            j -= 1
            path.append((j, i))
        else:
            j -= 1
            i -= 1
            path.append((j, i))
        step += 1

    path = path[::-1]
    score = F[n, m]

    # ==== 归一化得分 ====
    max_score = min(m, n) * np.max(np.diag(scoring_matrix))
    normalized_score = score / max_score if max_score != 0 else 0.0

    # return score, normalized_score, path, step, F
    return normalized_score

#add
def create_substitution_matrix(Xbin, Ybin, threshold):
    # 1. 生成所有网格中心点的坐标（按顺序编号）
    coords = [(x, y) for y in range(Ybin) for x in range(Xbin)]
    coords = np.array(coords)

    # 2. 计算欧几里得距离矩阵
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    # 3. 转换成 Substitution Matrix
    max_val = np.max(dist_matrix)
    sub_matrix = np.abs(dist_matrix - max_val) - (max_val - threshold)
    
    return sub_matrix




#########
def multimatch(s1, s2, im_size):
    s1x = s1[:, 0]
    s1y = s1[:, 1]
    l1 = len(s1x)
    if l1 < 3:
        scanpath1 = np.ones((3, 3), dtype=np.float32)
        scanpath1[:l1, 0] = s1x
        scanpath1[:l1, 1] = s1y
    else:
        scanpath1 = np.ones((l1, 3), dtype=np.float32)
        scanpath1[:, 0] = s1x
        scanpath1[:, 1] = s1y
    s2x = s2[:, 0]
    s2y = s2[:, 0]
    l2 = len(s2x)
    if l2 < 3:
        scanpath2 = np.ones((3, 3), dtype=np.float32)
        scanpath2[:l2, 0] = s2x
        scanpath2[:l2, 1] = s2y
    else:
        scanpath2 = np.ones((l2, 3), dtype=np.float32)
        scanpath2[:, 0] = s2x
        scanpath2[:, 1] = s2y
    mm = docomparison(scanpath1, scanpath2, sz=im_size)
    return mm[0]


def compute_mm(human_trajs, model_trajs, im_w, im_h, tasks=None):
    """
    compute scanpath similarity using multimatch
    """
    all_mm_scores = []
    for traj in model_trajs:
        img_name = traj['name']
        task = traj['task']
        gt_trajs = list(
            filter(lambda x: x['name'] == img_name and x['task'] == task,
                   human_trajs))
        all_mm_scores.append((task,
                              np.mean([
                                  multimatch(traj, gt_traj, (im_w, im_h))[:4]
                                  for gt_traj in gt_trajs
                              ],
                                      axis=0)))

    if tasks is not None:
        mm_tasks = {}
        for task in tasks:
            mm = np.array([x[1] for x in all_mm_scores if x[0] == task])
            mm_tasks[task] = np.mean(mm, axis=0)
        return mm_tasks
    else:
        return np.mean([x[1] for x in all_mm_scores], axis=0)


def scanpath2clusters(meanshift, scanpath):
    string = []
    # xs = scanpath['X']
    # ys = scanpath['Y']
    # for i in range(len(xs)):
    #     symbol = meanshift.predict([[xs[i], ys[i]]])[0]
    #     string.append(symbol)
    # return string
    string = []

    for i in range(scanpath.shape[0]):
        symbol = meanshift.predict([[scanpath[i][1], scanpath[i][0]]])[0]
        string.append(symbol)
    return string


def zero_one_similarity(a, b):
    if a == b:
        return 1.0
    else:
        return 0.0


def nw_matching(pred_string, gt_string, gap=0.0):
    # NW string matching with zero_one_similarity
    F = np.zeros((len(pred_string) + 1, len(gt_string) + 1), dtype=np.float32)
    for i in range(1 + len(pred_string)):
        F[i, 0] = gap * i
    for j in range(1 + len(gt_string)):
        F[0, j] = gap * j
    for i in range(1, 1 + len(pred_string)):
        for j in range(1, 1 + len(gt_string)):
            a = pred_string[i - 1]
            b = gt_string[j - 1]
            match = F[i - 1, j - 1] + zero_one_similarity(a, b)
            delete = F[i - 1, j] + gap
            insert = F[i, j - 1] + gap
            F[i, j] = np.max([match, delete, insert])
    score = F[len(pred_string), len(gt_string)]
    return score / max(len(pred_string), len(gt_string))


# compute sequence score
def compute_SS(preds, image_name, clusters, truncate, reduce='mean'):
    results = []
    for scanpath in preds:
        key = image_name
        ms = clusters[key]
        strings = ms['strings']
        cluster = ms['cluster']

        pred = scanpath2clusters(cluster, scanpath)
        scores = []
        for gt in strings:
            if len(gt) > 0:
                pred = pred[:truncate] if len(pred) > truncate else pred
                gt = gt[:truncate] if len(gt) > truncate else gt
                score = nw_matching(pred, gt)
                scores.append(score)
        result = {}
        # result['condition'] = scanpath['condition']
        # result['task'] = scanpath['task']
        # result['name'] = scanpath['name']
        if reduce == 'mean':
            result['score'] = np.array(scores).mean()
        elif reduce == 'max':
            result['score'] = max(scores)
        else:
            raise NotImplementedError
        results.append(result)
    return results

def get_seq_score(preds, image_name, clusters, max_step, tasks=None):
    results = compute_SS(preds, image_name, clusters, truncate=max_step)
    if tasks is None:
        return np.mean([r['score'] for r in results])
    else:
        scores = []
        for task in tasks:
            scores.append(
                np.mean([r['score'] for r in results if r['task'] == task]))
        return dict(zip(tasks, scores))