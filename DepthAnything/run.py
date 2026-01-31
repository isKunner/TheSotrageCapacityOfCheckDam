import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import sys

current_file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_file_dir)
sys.path.insert(0, parent_dir)

from depth_anything_v2.dpt import DepthAnythingV2
from DEMAndRemoteSensingUtils import crop_source_to_reference, batch_modify_tifs_vectorized
from LocalPath import Loess_Plateau_Copernicus, ROOT_DIR
sys.path.insert(0, str(ROOT_DIR))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img-path', default=r'D:\研究文件\ResearchData\USA\GoogleRemoteSensing\GeoDAR_v11_dams_of_USA_group1\33.tif', type=str)
    parser.add_argument('--input-size', type=int, default=1024)
    parser.add_argument('--outdir', type=str, default='./remote_img_results/vitl/USA/GeoDAR_v11_dams_of_USA_group1')
    # parser.add_argument('--reference-dem', type=str, default=r'D:\研究文件\ResearchData\USA\USGSDEM\GeoDAR_v11_dams_of_USA_group1\33.tif')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    DEVICE = 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(fr'C:\Users\Kevin\Desktop\TheSotrageCapacityOfCheckDam\DepthAnything\checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    # 设置合适的批次大小
    batch_size = args.batch_size  # 根据显存调整

    # 批量处理
    for i in range(0, len(filenames), batch_size):

        batch_filenames = filenames[i:i+batch_size]

        # 加载批次图像
        batch_images = []
        for filename in batch_filenames:
            raw_image = cv2.imread(filename)
            batch_images.append(raw_image)

        # 批量推理
        # 获取批次深度图 (假设返回的列表中每个 ndarray 形状相同)
        batch_depths_list = depth_anything.infer_batch(batch_images, args.input_size)

        # 将列表转换为 NumPy 数组 (B, H, W)
        batch_depths = np.stack(batch_depths_list, axis=0)  # Shape: (B, H, W)

        # --- 向量化归一化 ---
        # 计算沿空间维度 (H, W) 的最小值和最大值
        mins = np.min(batch_depths, axis=(1, 2), keepdims=True)  # Shape: (B, 1, 1)
        maxs = np.max(batch_depths, axis=(1, 2), keepdims=True)  # Shape: (B, 1, 1)

        # 计算范围，防止除零
        ranges = maxs - mins  # Shape: (B, 1, 1)

        # 向量化归一化，处理除零情况
        normalized_batch_depths_array = np.where(
            ranges != 0,
            (batch_depths - mins) / ranges,
            np.zeros_like(batch_depths)
        )  # Shape: (B, H, W)

        # 批量裁剪
        pair_path = crop_source_to_reference(Loess_Plateau_Copernicus, batch_filenames, args.outdir)

        tif_paths = [tif_path for _, tif_path in pair_path]

        output_paths = [os.path.join(args.outdir, os.path.basename(tif_path)) for tif_path in tif_paths]

        batch_modify_tifs_vectorized(tif_paths=tif_paths, input_matrices=normalized_batch_depths_array, output_paths=output_paths)


        if args.grayscale:
            depths_scaled_uint8 = (normalized_batch_depths_array * 255.0).astype(np.uint8)  # Shape: (B, H, W)
            depths_batch = np.repeat(depths_scaled_uint8[..., np.newaxis], 3, axis=-1)  # Shape: (B, H, W, 3)
            if args.pred_only:
                for i, (filename, depth_img) in enumerate(zip(batch_filenames, depths_batch)):
                    output_filename = os.path.join(args.outdir,
                                                   os.path.splitext(os.path.basename(filename))[0] + '.png')
                    cv2.imwrite(output_filename, depth_img)
            else:
                for i, (filename, orig_img, depth_img) in enumerate(zip(batch_filenames, batch_images, depths_batch)):
                    # Create separator for *this* specific image pair
                    split_region = np.full((orig_img.shape[0], 50, 3), 255, dtype=np.uint8)  # White region (H, 50, 3)
                    # Horizontally concatenate: original | separator | depth
                    combined_result = cv2.hconcat([orig_img, split_region, depth_img])
                    output_filename = os.path.join(args.outdir,
                                                   os.path.splitext(os.path.basename(filename))[0] + '.png')
                    cv2.imwrite(output_filename, combined_result)
        else:
            # Apply colormap to each depth map in the batch
            # Note: cmap expects [0, 1] input, so we use normalized_batch_depths_array
            if args.pred_only:
                for i, (filename, d_norm) in enumerate(zip(batch_filenames, normalized_batch_depths_array)):
                    colored_d = cmap(d_norm)  # Shape: (H, W, 4) RGBA
                    depth_img = (colored_d[:, :, :3] * 255)[:, :, ::-1].astype(
                        np.uint8)  # Take RGB, convert to BGR, uint8
                    output_filename = os.path.join(args.outdir,
                                                   os.path.splitext(os.path.basename(filename))[0] + '.png')
                    cv2.imwrite(output_filename, depth_img)
            else:
                for i, (filename, orig_img, d_norm) in enumerate(
                        zip(batch_filenames, batch_images, normalized_batch_depths_array)):
                    colored_d = cmap(d_norm)  # Shape: (H, W, 4) RGBA
                    depth_img = (colored_d[:, :, :3] * 255)[:, :, ::-1].astype(
                        np.uint8)  # Take RGB, convert to BGR, uint8
                    # Create separator for *this* specific image pair
                    split_region = np.full((orig_img.shape[0], 50, 3), 255, dtype=np.uint8)  # White region (H, 50, 3)
                    # Horizontally concatenate: original | separator | depth
                    combined_result = cv2.hconcat([orig_img, split_region, depth_img])
                    output_filename = os.path.join(args.outdir,
                                                   os.path.splitext(os.path.basename(filename))[0] + '.png')
                    cv2.imwrite(output_filename, combined_result)

