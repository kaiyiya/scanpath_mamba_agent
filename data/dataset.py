"""
数据加载器
"""
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class Salient360ScanpathDataset(Dataset):
    def __init__(self, data_path, split='train', seq_len=30, augment=True):
        print(f"Loading {split} data from {data_path}...")
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        raw_data = data_dict[split]
        self.seq_len = seq_len
        self.augment = augment and (split == 'train')  # 仅在训练时增强

        # 方案3：展开数据集，每条路径作为独立样本
        # 这样解决了"同一图像每次对应不同GT"的问题
        print(f"Expanding dataset: each scanpath becomes an independent sample...")
        self.samples = []
        for key in raw_data.keys():
            sample = raw_data[key]
            scanpaths = sample['scanpaths_2d']

            # 处理 object 类型
            if scanpaths.dtype == np.object_:
                scanpaths_list = []
                for i in range(len(scanpaths)):
                    sp = np.array(scanpaths[i], dtype=np.float32)
                    if sp.shape[1] == 3:
                        sp = sp[:, :2]
                    scanpaths_list.append(sp)
                scanpaths = np.array(scanpaths_list, dtype=np.float32)

            # 确保是 2D 坐标
            if scanpaths.shape[2] == 3:
                scanpaths = scanpaths[:, :, :2]

            # 为每条路径创建一个独立样本
            for scanpath_idx in range(len(scanpaths)):
                self.samples.append({
                    'key': key,
                    'image': sample['image'],
                    'saliency_map': sample.get('saliency_map') or sample.get('salmap'),
                    'scanpath': scanpaths[scanpath_idx].copy(),
                    'scanpath_idx': scanpath_idx
                })

        print(f"Loaded {len(self.samples)} samples from {len(raw_data)} images (augmentation: {self.augment})")
        print(f"  Average {len(self.samples)/len(raw_data):.1f} scanpaths per image")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        key = sample['key']

        # 加载图像
        img = sample['image']
        if isinstance(img, torch.Tensor):
            image = img.float()
        else:
            image = torch.from_numpy(img).float() if isinstance(img, np.ndarray) else torch.tensor(img, dtype=torch.float32)

        # 加载显著性图（如果存在）
        saliency_map = None
        sal = sample['saliency_map']
        if sal is not None:
            if isinstance(sal, torch.Tensor):
                saliency_map = sal.float()
            else:
                saliency_map = torch.from_numpy(sal).float() if isinstance(sal, np.ndarray) else torch.tensor(sal, dtype=torch.float32)
            # 确保是(1, H, W)格式
            if saliency_map.ndim == 2:
                saliency_map = saliency_map.unsqueeze(0)

        # 使用预先分配的固定路径（不再随机选择）
        scanpath = sample['scanpath'].copy()

        # 只取前2维（经纬度），忽略第3维（时间戳等）
        if scanpath.shape[1] > 2:
            scanpath = scanpath[:, :2]

        # 归一化坐标到[0, 1]范围（如果坐标不在[0,1]范围内）
        if scanpath.max() > 1.0 or scanpath.min() < 0.0:
            # 判断是否是经纬度坐标（经度范围大致在-180到180，纬度在-90到90）
            if scanpath[:, 0].min() < -100 or scanpath[:, 0].max() > 100:
                # 经纬度坐标：经度[-180,180] -> [0,1], 纬度[-90,90] -> [0,1]
                scanpath[:, 0] = (scanpath[:, 0] + 180.0) / 360.0  # 经度归一化
                scanpath[:, 1] = (scanpath[:, 1] + 90.0) / 180.0   # 纬度归一化
            else:
                # 其他坐标系统：使用min-max归一化
                scanpath_min = scanpath.min(axis=0, keepdims=True)
                scanpath_max = scanpath.max(axis=0, keepdims=True)
                scanpath_range = scanpath_max - scanpath_min
                scanpath_range[scanpath_range < 1e-8] = 1.0  # 避免除零
                scanpath = (scanpath - scanpath_min) / scanpath_range

        scanpath = torch.from_numpy(scanpath).float()

        # 数据增强（仅训练时）
        if self.augment:
            if saliency_map is not None:
                image, scanpath, saliency_map = self._apply_augmentation_with_saliency(image, scanpath, saliency_map)
            else:
                image, scanpath = self._apply_augmentation(image, scanpath)

        if scanpath.shape[0] > self.seq_len:
            scanpath = scanpath[:self.seq_len]
        elif scanpath.shape[0] < self.seq_len:
            pad_length = self.seq_len - scanpath.shape[0]
            last_point = scanpath[-1:]
            scanpath = torch.cat([scanpath, last_point.repeat(pad_length, 1)], dim=0)

        # 确保在[0,1]范围内（安全边界）
        scanpath = torch.clamp(scanpath, 0.0, 1.0)

        # 返回数据
        result = {'image': image, 'scanpath': scanpath, 'key': key}
        if saliency_map is not None:
            result['saliency_map'] = saliency_map
        return result

    def _apply_augmentation_with_saliency(self, image, scanpath, saliency_map):
        """
        应用数据增强（包含显著性图）
        对于360度全景图，支持：
        1. 水平翻转（左右镜像）
        2. 水平循环移位（利用360度连续性）
        """
        # 50%概率水平翻转
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])  # 水平翻转图像 (C, H, W)
            saliency_map = torch.flip(saliency_map, dims=[2])  # 水平翻转显著性图
            scanpath[:, 0] = 1.0 - scanpath[:, 0]  # X坐标镜像

        # 50%概率水平循环移位（利用360度连续性）
        if torch.rand(1).item() < 0.5:
            shift_ratio = torch.rand(1).item()  # 随机移位比例[0, 1]

            # 图像循环移位
            _, _, W = image.shape
            shift_pixels = int(W * shift_ratio)
            image = torch.roll(image, shifts=shift_pixels, dims=2)
            saliency_map = torch.roll(saliency_map, shifts=shift_pixels, dims=2)

            # 扫描路径X坐标相应移位
            scanpath[:, 0] = (scanpath[:, 0] + shift_ratio) % 1.0

        return image, scanpath, saliency_map

    def _apply_augmentation(self, image, scanpath):
        """
        应用数据增强
        对于360度全景图，支持：
        1. 水平翻转（左右镜像）
        2. 水平循环移位（利用360度连续性）
        """
        # 50%概率水平翻转
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])  # 水平翻转图像 (C, H, W)
            scanpath[:, 0] = 1.0 - scanpath[:, 0]  # X坐标镜像

        # 50%概率水平循环移位（利用360度连续性）
        if torch.rand(1).item() < 0.5:
            shift_ratio = torch.rand(1).item()  # 随机移位比例[0, 1]

            # 图像循环移位
            _, _, W = image.shape
            shift_pixels = int(W * shift_ratio)
            image = torch.roll(image, shifts=shift_pixels, dims=2)

            # 扫描路径X坐标相应移位
            scanpath[:, 0] = (scanpath[:, 0] + shift_ratio) % 1.0

        return image, scanpath

def create_dataloaders(config):
    train_dataset = Salient360ScanpathDataset(
        config.processed_data_path,
        'train',
        config.seq_len,
        augment=config.use_augmentation
    )
    test_dataset = Salient360ScanpathDataset(
        config.processed_data_path,
        'test',
        config.seq_len,
        augment=False  # 测试时不增强
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=config.pin_memory, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory, drop_last=False)

    return train_loader, test_loader
