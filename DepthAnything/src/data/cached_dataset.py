#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: cached_dataset.py

"""
带缓存的数据集模块（支持增量追加）

使用方式：
1. 首次训练：自动构建缓存
2. 新增样本后：设置 force_rebuild_cache=True
   - 已存在的样本（在index.json中）直接跳过
   - 新增样本追加到已有缓存后面
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Dict, Tuple

from .dataset import collect_valid_samples
from .augmentation import DEMDataAugmentation


class CachedDEMDataset(Dataset):
    """
    带缓存的DEM数据集（分块版本，支持增量追加，支持DAM Encoder缓存）
    
    索引文件结构（index.json）：
    {
        "version": 1,
        "chunk_size": 10,
        "cache_dam_encoder": true,
        "samples": [
            {"filename": "tile_001", "chunk_idx": 0, "local_idx": 0},
            ...
        ]
    }
    """
    
    INDEX_FILE = "index.json"

    def __init__(
        self,
        samples: List[Dict],
        cache_dir: str = "./data_cache",
        target_size: int = 1022,
        chunk_size: int = 10,
        normalize: bool = True,
        augmentation: bool = True,
        force_rebuild_cache: bool = False,
        cache_dam_encoder: bool = False,
        dam_model=None,
        dam_device='cuda',
        dam_batch_size: int = 1,
    ):
        self.target_size = target_size
        self.chunk_size = chunk_size
        self.normalize = normalize
        self.augmentation = augmentation
        self.cache_dir = cache_dir
        self.cache_dam_encoder = cache_dam_encoder
        self.dam_model = dam_model
        self.dam_device = dam_device
        self.dam_batch_size = dam_batch_size
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if augmentation:
            self.aug = DEMDataAugmentation()
        else:
            self.aug = None

        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]

        # LRU缓存
        self._chunk_cache = {}
        self._chunk_cache_keys = []
        self._chunk_cache_size = 2

        # 加载索引
        self.index = self._load_index()
        
        # 检查配置是否变化
        config_changed = False
        index_chunk_size = self.index.get('chunk_size', chunk_size)
        if index_chunk_size != chunk_size:
            print(f"警告: chunk_size 从 {index_chunk_size} 变为 {chunk_size}，需要重建缓存")
            config_changed = True
        
        index_cache_dam = self.index.get('cache_dam_encoder', False)
        if index_cache_dam != cache_dam_encoder:
            print(f"警告: cache_dam_encoder 从 {index_cache_dam} 变为 {cache_dam_encoder}，需要重建缓存")
            config_changed = True
        
        if config_changed and len(self.index['samples']) > 0:
            force_rebuild_cache = True
        
        if force_rebuild_cache:
            # 强制重建：跳过已存在的，追加新增的
            self._build_cache_incremental(samples)
        elif len(self.index['samples']) == 0:
            # 首次构建
            print(f"首次构建缓存，分块大小: {self.chunk_size}，DAM Encoder缓存: {self.cache_dam_encoder}")
            self._build_cache_full(samples)
        else:
            # 缓存已存在，直接使用（不检查是否有新增样本）
            print(f"缓存已存在，共 {len(self.index['samples'])} 个样本")
        
        # 根据索引重建samples列表（保持顺序）
        self.samples = self._rebuild_samples_list(samples)

    def _get_index_path(self) -> str:
        return os.path.join(self.cache_dir, self.INDEX_FILE)

    def _load_index(self) -> Dict:
        index_path = self._get_index_path()
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                return json.load(f)
        return {
            "version": 1,
            "chunk_size": self.chunk_size,
            "cache_dam_encoder": self.cache_dam_encoder,
            "dam_batch_size": getattr(self, 'dam_batch_size', 1),
            "samples": []
        }

    def _save_index(self):
        self.index['chunk_size'] = self.chunk_size
        self.index['cache_dam_encoder'] = self.cache_dam_encoder
        self.index['dam_batch_size'] = getattr(self, 'dam_batch_size', 1)
        with open(self._get_index_path(), 'w') as f:
            json.dump(self.index, f, indent=2)

    def _get_chunk_path(self, chunk_idx: int, data_type: str) -> str:
        return os.path.join(self.cache_dir, f"chunk_{chunk_idx:06d}_{data_type}.npy")

    def _get_chunk_meta_path(self, chunk_idx: int) -> str:
        return os.path.join(self.cache_dir, f"chunk_{chunk_idx:06d}_meta.npy")

    def _get_dam_encoder_chunk_path(self, chunk_idx: int) -> str:
        return os.path.join(self.cache_dir, f"chunk_{chunk_idx:06d}_dam_encoder.npz")

    def _get_existing_filenames(self) -> set:
        return {s['filename'] for s in self.index['samples']}

    def _build_cache_incremental(self, all_samples: List[Dict]):
        """增量重建（支持batch推理）"""
        existing = self._get_existing_filenames()
        new_samples = [s for s in all_samples if s['filename'] not in existing]

        if len(new_samples) == 0:
            print("没有新增样本，无需更新缓存")
            return

        print(f"重建缓存：已有 {len(existing)} 个，新增 {len(new_samples)} 个")

        # 找到最后一个chunk
        if len(self.index['samples']) > 0:
            last_chunk_idx = max(s['chunk_idx'] for s in self.index['samples'])
            last_meta_path = self._get_chunk_meta_path(last_chunk_idx)
            if os.path.exists(last_meta_path):
                last_meta = np.load(last_meta_path, allow_pickle=True).item()
                last_count = last_meta.get('valid_count', 0)
            else:
                last_count = 0
        else:
            last_chunk_idx = 0
            last_count = 0

        # 加载最后一个chunk的数据
        if last_count > 0 and last_count < self.chunk_size:
            chunk_cop, chunk_google, chunk_usgs, chunk_stats = self._load_chunk_data(last_chunk_idx)
            current_chunk_idx = last_chunk_idx
        else:
            chunk_cop, chunk_google, chunk_usgs, chunk_stats = [], [], [], []
            current_chunk_idx = last_chunk_idx + 1 if len(self.index['samples']) > 0 else 0

        # 处理新增样本
        valid_count = 0
        invalid_count = 0
        chunk_dam_features = []

        # 如果缓存DAM Encoder
        if self.cache_dam_encoder:
            if self.dam_model is None:
                raise ValueError("cache_dam_encoder=True 时需要提供 dam_model 参数")
            self.dam_model.eval()
            self.dam_model.to(self.dam_device)
            print(f"预计算新增样本的DAM Encoder特征（batch size: {self.dam_batch_size}）...")

        # 使用batch处理
        sample_buffer = []  # 缓存待处理的样本
        google_buffer = []  # 缓存待处理的Google数据

        for sample in tqdm(new_samples, desc="Adding new samples"):
            result = self._process_sample(sample)
            if result is None:
                invalid_count += 1
                continue

            copernicus_norm, google_data, usgs_norm, stats = result

            # 添加到当前chunk
            chunk_cop.append(copernicus_norm)
            chunk_google.append(google_data)
            if usgs_norm is not None:
                chunk_usgs.append(usgs_norm)
            chunk_stats.append(stats)

            # 缓存DAM Encoder特征
            if self.cache_dam_encoder:
                sample_buffer.append({
                    'filename': sample['filename'],
                    'chunk_idx': current_chunk_idx,
                    'local_idx': len(chunk_cop) - 1
                })
                google_buffer.append(google_data)

                # 当buffer满时，批量处理
                if len(google_buffer) >= self.dam_batch_size:
                    dam_features_list = self._compute_dam_encoder_features_batch(google_buffer)
                    chunk_dam_features.extend(dam_features_list)

                    # 更新索引
                    for s in sample_buffer:
                        self.index['samples'].append(s)

                    sample_buffer = []
                    google_buffer = []
            else:
                # 更新索引
                self.index['samples'].append({
                    "filename": sample['filename'],
                    "chunk_idx": current_chunk_idx,
                    "local_idx": len(chunk_cop) - 1
                })

            valid_count += 1

            # chunk满了，保存并新建
            if len(chunk_cop) >= self.chunk_size:
                # 处理剩余的buffer
                if self.cache_dam_encoder and len(google_buffer) > 0:
                    dam_features_list = self._compute_dam_encoder_features_batch(google_buffer)
                    chunk_dam_features.extend(dam_features_list)
                    for s in sample_buffer:
                        self.index['samples'].append(s)
                    sample_buffer = []
                    google_buffer = []

                self._save_chunk(current_chunk_idx, chunk_cop, chunk_google, chunk_usgs, chunk_stats)
                if self.cache_dam_encoder:
                    self._save_dam_encoder_chunk(current_chunk_idx, chunk_dam_features)
                    chunk_dam_features = []
                current_chunk_idx += 1
                chunk_cop, chunk_google, chunk_usgs, chunk_stats = [], [], [], []

        # 保存最后一个未满的chunk
        if len(chunk_cop) > 0:
            # 处理剩余的buffer
            if self.cache_dam_encoder and len(google_buffer) > 0:
                dam_features_list = self._compute_dam_encoder_features_batch(google_buffer)
                chunk_dam_features.extend(dam_features_list)
                for s in sample_buffer:
                    self.index['samples'].append(s)

            self._save_chunk(current_chunk_idx, chunk_cop, chunk_google, chunk_usgs, chunk_stats)
            if self.cache_dam_encoder:
                self._save_dam_encoder_chunk(current_chunk_idx, chunk_dam_features)

        self._save_index()
        print(f"增量完成：有效 {valid_count}，丢弃 {invalid_count}，总计 {len(self.index['samples'])}")

    def _build_cache_full(self, samples: List[Dict]):
        """完整重建（支持batch推理）"""
        # 清空旧数据
        for f in os.listdir(self.cache_dir):
            if f.startswith('chunk_') or f == self.INDEX_FILE:
                os.remove(os.path.join(self.cache_dir, f))

        self.index = {
            "version": 1,
            "chunk_size": self.chunk_size,
            "cache_dam_encoder": self.cache_dam_encoder,
            "dam_batch_size": self.dam_batch_size,
            "samples": []
        }

        chunk_cop, chunk_google, chunk_usgs, chunk_stats = [], [], [], []
        chunk_dam_features = []
        chunk_idx = 0
        valid_count = 0
        invalid_count = 0

        # 如果缓存DAM Encoder
        if self.cache_dam_encoder:
            if self.dam_model is None:
                raise ValueError("cache_dam_encoder=True 时需要提供 dam_model 参数")
            self.dam_model.eval()
            self.dam_model.to(self.dam_device)
            print(f"预计算DAM Encoder特征（batch size: {self.dam_batch_size}）...")

        # 使用batch处理
        sample_buffer = []
        google_buffer = []

        for sample in tqdm(samples, desc="Building cache"):

            try:

                result = self._process_sample(sample)
                if result is None:
                    invalid_count += 1
                    continue

                copernicus_norm, google_data, usgs_norm, stats = result

                chunk_cop.append(copernicus_norm)
                chunk_google.append(google_data)
                if usgs_norm is not None:
                    chunk_usgs.append(usgs_norm)
                chunk_stats.append(stats)

                # 缓存DAM Encoder特征
                if self.cache_dam_encoder:
                    sample_buffer.append({
                        'filename': sample['filename'],
                        'chunk_idx': chunk_idx,
                        'local_idx': len(chunk_cop) - 1
                    })
                    google_buffer.append(google_data)

                    # 当buffer满时，批量处理
                    if len(google_buffer) >= self.dam_batch_size:
                        dam_features_list = self._compute_dam_encoder_features_batch(google_buffer)
                        chunk_dam_features.extend(dam_features_list)

                        # 更新索引
                        for s in sample_buffer:
                            self.index['samples'].append(s)

                        sample_buffer = []
                        google_buffer = []
                else:
                    # 更新索引
                    self.index['samples'].append({
                        "filename": sample['filename'],
                        "chunk_idx": chunk_idx,
                        "local_idx": len(chunk_cop) - 1
                    })

                valid_count += 1

                # chunk满了，保存
                if len(chunk_cop) >= self.chunk_size:
                    # 处理剩余的buffer
                    if self.cache_dam_encoder and len(google_buffer) > 0:
                        dam_features_list = self._compute_dam_encoder_features_batch(google_buffer)
                        chunk_dam_features.extend(dam_features_list)
                        for s in sample_buffer:
                            self.index['samples'].append(s)
                        sample_buffer = []
                        google_buffer = []

                    self._save_chunk(chunk_idx, chunk_cop, chunk_google, chunk_usgs, chunk_stats)
                    if self.cache_dam_encoder:
                        self._save_dam_encoder_chunk(chunk_idx, chunk_dam_features)
                        chunk_dam_features = []
                    chunk_idx += 1
                    chunk_cop, chunk_google, chunk_usgs, chunk_stats = [], [], [], []

            except Exception as e:
                print(e)
            finally:
                self._save_index()

        # 保存最后一个
        if len(chunk_cop) > 0:
            # 处理剩余的buffer
            if self.cache_dam_encoder and len(google_buffer) > 0:
                dam_features_list = self._compute_dam_encoder_features_batch(google_buffer)
                chunk_dam_features.extend(dam_features_list)
                for s in sample_buffer:
                    self.index['samples'].append(s)

            self._save_chunk(chunk_idx, chunk_cop, chunk_google, chunk_usgs, chunk_stats)
            if self.cache_dam_encoder:
                self._save_dam_encoder_chunk(chunk_idx, chunk_dam_features)

        self._save_index()
        print(f"构建完成：有效 {valid_count}，丢弃 {invalid_count}，总计 {len(self.index['samples'])}")
    
    def _compute_dam_encoder_features_batch(self, google_data_list: List[np.ndarray]) -> List[Dict]:
        """
        批量计算DAM Encoder输出特征（优化版本）

        Args:
            google_data_list: Google影像列表 [np.ndarray, ...]

        Returns:
            List[Dict]: 每个样本的特征字典列表
        """
        import torch

        batch_size = len(google_data_list)

        with torch.no_grad():
            # 批量转换为tensor
            batch_tensor = torch.stack([
                torch.from_numpy(g).float() for g in google_data_list
            ], dim=0).to(self.dam_device)

            # 批量获取DAM Encoder的输出特征
            if hasattr(self.dam_model, 'get_encoder_features'):
                # 注意：get_encoder_features可能不支持真正的batch处理
                # 需要检查dam_model的实现
                features_list = []

                # 尝试批量处理
                try:
                    features, patch_h, patch_w = self.dam_model.get_encoder_features(batch_tensor)
                    # 如果成功，解析batch结果
                    for i in range(batch_size):
                        features_np = {}
                        for j, feat_tuple in enumerate(features):
                            patch_tokens, cls_token = feat_tuple
                            # 提取第i个样本的特征
                            features_np[f'layer_{j}_patch'] = patch_tokens[i].cpu().numpy()
                            features_np[f'layer_{j}_cls'] = cls_token[i].cpu().numpy()
                        features_np['patch_h'] = np.array(patch_h)
                        features_np['patch_w'] = np.array(patch_w)
                        features_list.append(features_np)
                except Exception as e:
                    # 如果批量处理失败，回退到逐个处理
                    print(f"批量处理失败，回退到逐个处理: {e}")
                    for i in range(batch_size):
                        single_tensor = batch_tensor[i:i+1]
                        features, patch_h, patch_w = self.dam_model.get_encoder_features(single_tensor)
                        features_np = {}
                        for j, feat_tuple in enumerate(features):
                            patch_tokens, cls_token = feat_tuple
                            features_np[f'layer_{j}_patch'] = patch_tokens.cpu().numpy()
                            features_np[f'layer_{j}_cls'] = cls_token.cpu().numpy()
                        features_np['patch_h'] = np.array(patch_h)
                        features_np['patch_w'] = np.array(patch_w)
                        features_list.append(features_np)

                return features_list
            else:
                raise NotImplementedError("DAM模型需要提供 get_encoder_features 方法")
    
    def _save_dam_encoder_chunk(self, chunk_idx: int, dam_features_list: List[Dict]):
        """保存DAM Encoder特征chunk"""
        # 合并列表中的特征字典
        merged_features = {}
        for key in dam_features_list[0].keys():
            merged_features[key] = np.stack([f[key] for f in dam_features_list])
        
        np.savez(self._get_dam_encoder_chunk_path(chunk_idx), **merged_features)

    def _process_sample(self, sample: Dict) -> Tuple:
        """处理单个样本，返回归一化后的数据，如果无效返回None"""
        filename = sample['filename']
        
        # 检查文件
        if not os.path.exists(sample['copernicus_path']) or not os.path.exists(sample['google_path']):
            return None
        
        has_ground_truth = sample['usgs_path'] is not None and os.path.exists(sample['usgs_path'])
        if sample['usgs_path'] is not None and not has_ground_truth:
            return None
        
        # 读取
        copernicus_data = self._read_tif(sample['copernicus_path'])
        google_data = self._read_tif(sample['google_path'])
        
        # 检查Google
        if np.all(np.abs(google_data) < 1e-6):
            print(f"[丢弃] {filename}: Google全为NoData")
            return None
        
        # 归一化
        if has_ground_truth:
            usgs_data = self._read_tif(sample['usgs_path'])
            copernicus_norm, usgs_norm, cop_stats, usgs_stats = self._normalize_dem(copernicus_data, usgs_data)
            
            if copernicus_norm is None:
                print(f"[丢弃] {filename}: Copernicus全为NoData")
                return None
            if usgs_norm is None:
                print(f"[丢弃] {filename}: USGS全为NoData")
                return None
        else:
            copernicus_norm, _, cop_stats, _ = self._normalize_dem(copernicus_data, None)
            if copernicus_norm is None:
                print(f"[丢弃] {filename}: Copernicus全为NoData")
                return None
            usgs_norm = None
            usgs_stats = None
        
        # Google归一化
        if self.normalize:
            if google_data.max() > 1:
                google_data = google_data / 255.0
            for i in range(google_data.shape[0]):
                google_data[i] = (google_data[i] - self.image_mean[i]) / self.image_std[i]
        
        # 统计量
        cop_mean, cop_std = cop_stats
        usgs_mean, usgs_std = usgs_stats if usgs_stats else (0.0, 1.0)
        stats = [cop_mean, cop_std, usgs_mean, usgs_std]
        
        return copernicus_norm, google_data, usgs_norm, stats

    def _load_chunk_data(self, chunk_idx: int):
        """加载已有chunk数据（用于续写）"""
        cop = np.load(self._get_chunk_path(chunk_idx, 'copernicus'))
        google = np.load(self._get_chunk_path(chunk_idx, 'google'))
        meta = np.load(self._get_chunk_meta_path(chunk_idx), allow_pickle=True).item()
        
        usgs_list = []
        usgs_path = self._get_chunk_path(chunk_idx, 'usgs')
        if os.path.exists(usgs_path):
            usgs = np.load(usgs_path)
            usgs_list = list(usgs)
        
        return list(cop), list(google), usgs_list, meta['stats'].tolist()

    def _save_chunk(self, chunk_idx: int, cop_list, google_list, usgs_list, stats_list):
        """保存chunk"""
        np.save(self._get_chunk_path(chunk_idx, 'copernicus'), np.stack(cop_list))
        np.save(self._get_chunk_path(chunk_idx, 'google'), np.stack(google_list))
        if len(usgs_list) > 0:
            np.save(self._get_chunk_path(chunk_idx, 'usgs'), np.stack(usgs_list))
        
        meta = {
            'stats': np.array(stats_list, dtype=np.float32),
            'valid_count': len(cop_list)
        }
        np.save(self._get_chunk_meta_path(chunk_idx), meta)

    def _rebuild_samples_list(self, all_samples: List[Dict]) -> List[Dict]:
        """根据索引重建samples列表"""
        filename_map = {s['filename']: s for s in all_samples}
        result = []
        for idx_info in self.index['samples']:
            if idx_info['filename'] in filename_map:
                result.append(filename_map[idx_info['filename']])
        return result

    def _read_tif(self, filepath: str) -> np.ndarray:
        import rasterio
        with rasterio.open(filepath) as src:
            data = src.read()
            if data.ndim == 2:
                data = data[np.newaxis, ...]
        if data.shape[1] == 1024 and data.shape[2] == 1024:
            data = data[:, 1:-1, 1:-1]
        return data.astype(np.float32)

    def _zscore_normalize(self, data: np.ndarray) -> Tuple:
        nodata_value = -100
        data = data.copy()
        data[data <= nodata_value] = np.nan
        
        if np.all(np.isnan(data)):
            return None, 0.0, 1.0
        
        valid_data = data[~np.isnan(data)]
        if np.all(np.abs(valid_data) < 1e-6):
            return None, 0.0, 1.0
        
        try:
            lower = np.nanpercentile(data, 1.0)
            upper = np.nanpercentile(data, 99.0)
        except:
            lower, upper = np.nanmin(data), np.nanmax(data)
        
        data_clipped = np.clip(data, lower, upper)
        mean = np.nanmean(data_clipped)
        std = np.nanstd(data_clipped)
        
        if std < 1e-6 or not np.isfinite(std) or not np.isfinite(mean):
            std = 1.0
            mean = 0.0
        
        normalized = (data - mean) / std
        normalized = np.nan_to_num(normalized, nan=0.0)
        
        return normalized.astype(np.float32), float(mean), float(std)

    def _normalize_dem(self, cop_data: np.ndarray, usgs_data: np.ndarray = None) -> Tuple:
        cop_norm, cop_mean, cop_std = self._zscore_normalize(cop_data)
        if usgs_data is not None:
            usgs_norm, usgs_mean, usgs_std = self._zscore_normalize(usgs_data)
            return cop_norm, usgs_norm, (cop_mean, cop_std), (usgs_mean, usgs_std)
        return cop_norm, None, (cop_mean, cop_std), None

    def _load_chunk_with_cache(self, chunk_idx: int, data_type: str):
        cache_key = f"{chunk_idx}_{data_type}"
        if cache_key in self._chunk_cache:
            return self._chunk_cache[cache_key]
        
        data = np.load(self._get_chunk_path(chunk_idx, data_type))
        
        self._chunk_cache[cache_key] = data
        self._chunk_cache_keys.append(cache_key)
        
        while len(self._chunk_cache_keys) > self._chunk_cache_size:
            old_key = self._chunk_cache_keys.pop(0)
            del self._chunk_cache[old_key]
        
        return data

    def __len__(self) -> int:
        return len(self.samples)

    def _load_dam_encoder_features(self, chunk_idx: int, local_idx: int) -> Dict:
        """加载DAM Encoder特征"""
        dam_path = self._get_dam_encoder_chunk_path(chunk_idx)
        if not os.path.exists(dam_path):
            raise FileNotFoundError(f"DAM Encoder特征文件不存在: {dam_path}")
        
        # 使用LRU缓存
        cache_key = f"{chunk_idx}_dam_encoder"
        if cache_key in self._chunk_cache:
            dam_data = self._chunk_cache[cache_key]
        else:
            dam_data = np.load(dam_path)
            self._chunk_cache[cache_key] = dam_data
            self._chunk_cache_keys.append(cache_key)
            
            while len(self._chunk_cache_keys) > self._chunk_cache_size:
                old_key = self._chunk_cache_keys.pop(0)
                del self._chunk_cache[old_key]
        
        # 提取指定索引的特征
        features = {}
        for key in dam_data.files:
            features[key] = dam_data[key][local_idx]
        
        return features

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        idx_info = self.index['samples'][idx]
        chunk_idx, local_idx = idx_info['chunk_idx'], idx_info['local_idx']
        
        cop_chunk = self._load_chunk_with_cache(chunk_idx, 'copernicus')
        google_chunk = self._load_chunk_with_cache(chunk_idx, 'google')
        meta = np.load(self._get_chunk_meta_path(chunk_idx), allow_pickle=True).item()
        
        cop_data = cop_chunk[local_idx]
        google_data = google_chunk[local_idx]
        stats = meta['stats'][local_idx]
        
        usgs_path = self._get_chunk_path(chunk_idx, 'usgs')
        if os.path.exists(usgs_path):
            usgs_chunk = self._load_chunk_with_cache(chunk_idx, 'usgs')
            usgs_data = usgs_chunk[local_idx]
            has_gt = True
        else:
            usgs_data = np.zeros_like(cop_data)
            has_gt = False

        cop_t = torch.from_numpy(cop_data)
        google_t = torch.from_numpy(google_data)
        usgs_t = torch.from_numpy(usgs_data)

        if self.aug is not None and has_gt:
            cop_t, google_t, usgs_t = self.aug(cop_t, google_t, usgs_t)

        result = {
            'copernicus': cop_t,
            'google': google_t,
            'usgs': usgs_t,
            'group': self.samples[idx]['group'],
            'filename': self.samples[idx]['filename'],
            'cop_mean': float(stats[0]),
            'cop_std': float(stats[1]),
            'usgs_mean': float(stats[2]),
            'usgs_std': float(stats[3]),
            'has_ground_truth': has_gt
        }
        
        # 如果缓存了DAM Encoder特征，添加到结果
        if self.cache_dam_encoder:
            dam_features = self._load_dam_encoder_features(chunk_idx, local_idx)

            # 提取patch_h和patch_w
            patch_h = int(dam_features['patch_h'])
            patch_w = int(dam_features['patch_w'])

            # 重组特征列表: [(patch_tokens, cls_token), ...]
            features = []
            layer_idx = 0
            while f'layer_{layer_idx}_patch' in dam_features:
                patch_tokens = torch.from_numpy(dam_features[f'layer_{layer_idx}_patch'])
                cls_token = torch.from_numpy(dam_features[f'layer_{layer_idx}_cls'])
                features.append((patch_tokens, cls_token))
                layer_idx += 1

            result['dam_encoder_features'] = features
            result['dam_encoder_patch_h'] = patch_h
            result['dam_encoder_patch_w'] = patch_w
        
        return result


def _collate_fn_filter_none(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)


def create_dataloaders_with_cache(
    base_dir: str,
    batch_size: int = 4,
    num_workers: int = 4,
    seed: int = 42,
    cache_dir: str = './data_cache',
    target_size: int = 1022,
    chunk_size: int = 10,
    force_rebuild_cache: bool = False,
    cache_dam_encoder: bool = False,
    dam_model=None,
    dam_device='cuda',
    dam_batch_size=1,
) -> Tuple[DataLoader, DataLoader]:
    """
    创建带缓存的数据加载器
    
    Args:
        cache_dam_encoder: 是否缓存DAM Encoder输出（当不微调DAM时使用）
        dam_model: DAM模型实例（用于预计算Encoder输出）
        dam_device: DAM模型运行的设备
    """
    train_samples, test_samples = collect_valid_samples(base_dir, seed=seed)
    
    train_dataset = CachedDEMDataset(
        train_samples, cache_dir=cache_dir, target_size=target_size,
        chunk_size=chunk_size, normalize=True, augmentation=True, 
        force_rebuild_cache=force_rebuild_cache,
        cache_dam_encoder=cache_dam_encoder,
        dam_model=dam_model,
        dam_device=dam_device,
        dam_batch_size=dam_batch_size
    )
    
    test_dataset = CachedDEMDataset(
        test_samples, cache_dir=cache_dir + '_test', target_size=target_size,
        chunk_size=chunk_size, normalize=True, augmentation=False, 
        force_rebuild_cache=force_rebuild_cache,
        cache_dam_encoder=cache_dam_encoder,
        dam_model=dam_model,
        dam_device=dam_device,
        dam_batch_size=dam_batch_size
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0, collate_fn=_collate_fn_filter_none
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        persistent_workers=num_workers > 0, collate_fn=_collate_fn_filter_none
    )
    
    return train_loader, test_loader
