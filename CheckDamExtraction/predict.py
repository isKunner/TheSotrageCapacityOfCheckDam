#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName: predict
# @Time    : 2025/12/6 11:02
# @Author  : Kevin
# @Describe: Prediction module for check dam segmentation model
import glob

import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import argparse
import json
import os

from Model import SegmentationModel

class CheckDamPredictor:
    def __init__(self, model_name, checkpoint_path, class_names, device=None):
        """
        Initialize the predictor with trained model.

        Args:
            checkpoint_path (str): Path to trained model checkpoint
            class_names (list): List of class names
            device (str): Device to run prediction on
        """
        self.class_names = class_names
        if "background" not in class_names:
            class_names.insert(0, "background")
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load trained model
        self.model = SegmentationModel.load_from_checkpoint(checkpoint_path, model_name=model_name, num_classes=len(class_names))
        # try:
        #     self.model = FCN.load_from_checkpoint(checkpoint_path, model_name='FCN', num_classes=len(class_names))
        # except:
        #     self.model = FCN(num_classes=len(class_names))
        #     checkpoint = torch.load(checkpoint_path, map_location=self.device)
        #     if 'state_dict' in checkpoint:
        #         state_dict = checkpoint['state_dict']
        #         # 过滤掉不匹配的键（如 criterion.alpha）
        #         filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('criterion')}
        #         self.model.load_state_dict(filtered_state_dict, strict=False)
        #     else:
        #         # 直接加载checkpoint
        #         filtered_checkpoint = {k: v for k, v in checkpoint.items() if not k.startswith('criterion')}
        #         self.model.load_state_dict(filtered_checkpoint, strict=False)

        self.model.to(self.device)
        self.model.eval()

        # Create color map for consistent coloring
        self.custom_colors = [
            (0, 0, 0),  # 0: #000000 - Black (保留黑色作为背景或特定类别)
            (230, 25, 75),  # 1: #E6194B - Red
            (60, 180, 75),  # 2: #3CB44B - Green
            (255, 225, 25),  # 3: #FFE119 - Yellow
            (0, 130, 200),  # 4: #0082C8 - Blue
            (0, 128, 128),   # 5: #008080 - Teal
            (145, 30, 180),  # 6: #911EB4 - Purple
            (70, 240, 240),  # 7: #46F0F0 - Cyan
            (245, 130, 48),  # 8: #F58231 - Orange
            # 可以根据需要继续添加更多对比鲜明的颜色
        ]

    def predict_single_image(self, image_path, output_path=None, show=False):
        """
        Predict segmentation mask for a single image.

        Args:
            image_path (str): Path to input image
            output_path (str): Path to save prediction result
            show (bool): Whether to display result

        Returns:
            np.ndarray: Predicted segmentation mask
        """
        # Load and preprocess image (following the same process as CheckDamSegmentationDataSet.__getitem__)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)

        # Apply the same transformation as in training
        image_tensor = transforms.ToTensor()(image_pil)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)  # Add batch dimension

        # Run prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # Save result if needed
        if output_path:
            self._save_prediction_visualization(image, prediction, output_path)

        # Show result if needed
        if show:
            self._show_prediction(image, prediction)

        return prediction

    # 在 CheckDamPredictor 类中添加批量预测方法
    def predict_multiple_images(self, image_paths, output_dir=None, show=False):
        """
        Predict segmentation masks for multiple images.

        Args:
            image_paths (list): List of image paths
            output_dir (str): Directory to save prediction results
            show (bool): Whether to display results

        Returns:
            list: List of prediction results
        """
        if output_dir is None:
            output_dir = "predictions"

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        predictions = []

        for image_path in image_paths:
            # Extract filename without extension
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)

            # Create output path with same filename
            output_path = os.path.join(output_dir, f"{name}_prediction{ext}")

            # Perform prediction
            prediction = self.predict_single_image(
                image_path=image_path,
                output_path=output_path,
                show=False
            )

            predictions.append(prediction)

            print(f"Prediction completed for {filename}")

            # Show result if requested
            if show:
                self._show_prediction(cv2.imread(image_path), prediction)

        return predictions

    def predict_with_json_annotation(self, image_path, json_path, output_path=None, show=False):
        """
        Predict segmentation mask for an image with its JSON annotation (for consistency testing).

        Args:
            image_path (str): Path to input image
            json_path (str): Path to JSON annotation file
            output_path (str): Path to save prediction result
            show (bool): Whether to display result

        Returns:
            tuple: (original_mask, predicted_mask)
        """
        # Process like CheckDamSegmentationDataSet.__getitem__
        with open(json_path, 'r') as f:
            label_data = json.load(f)

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Create ground truth mask like in CheckDamSegmentationDataSet.__getitem__
        mask = np.zeros((h, w), dtype=np.uint8)
        shapes = label_data.get("shapes", [])
        for shape in shapes:
            label = shape.get("label", "")
            points = shape.get("points", [])
            if not points or len(points) < 3:
                continue
            if label in self.class_to_idx:
                class_id = self.class_to_idx[label]
                pts = np.array(points, np.int32)
                cv2.fillPoly(mask, [pts], color=class_id)

        # Predict
        image_pil = Image.fromarray(image)
        image_tensor = transforms.ToTensor()(image_pil)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        # Save result if needed
        if output_path:
            self._save_comparison_visualization(image, mask, prediction, output_path)

        # Show result if needed
        if show:
            self._show_comparison(image, mask, prediction)

        return mask, prediction

    # 在 CheckDamPredictor 类中添加批量预测带 JSON 的方法
    def predict_multiple_images_with_json(self, image_paths, json_paths, output_dir=None, show=False):
        """
        Predict segmentation masks for multiple images with their corresponding JSON annotations.

        Args:
            image_paths (list): List of image paths
            json_paths (list): List of JSON annotation paths
            output_dir (str): Directory to save prediction results
            show (bool): Whether to display results

        Returns:
            list: List of prediction results
        """
        if output_dir is None:
            output_dir = "predictions"

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        predictions = []

        # 验证图像和JSON文件数量匹配
        if len(image_paths) != len(json_paths):
            raise ValueError(f"Number of images ({len(image_paths)}) must match number of JSON files ({len(json_paths)})")

        for image_path, json_path in zip(image_paths, json_paths):
            # Extract filename without extension
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)

            # Create output path with same filename
            output_path = os.path.join(output_dir, f"{name}_prediction{ext}")

            # Perform prediction with JSON annotation
            ground_truth, prediction = self.predict_with_json_annotation(
                image_path=image_path,
                json_path=json_path,
                output_path=output_path,
                show=False
            )

            predictions.append(prediction)

            print(f"Prediction completed for {filename}")

            # Show result if requested
            if show:
                self._show_comparison(cv2.imread(image_path), ground_truth, prediction)

        return predictions

    def _get_color_for_class(self, class_id):
        """
        Get RGB color for a specific class ID.

        Args:
            class_id (int): Class ID

        Returns:
            tuple: (R, G, B) color values (0-255)
        """
        if class_id >= len(self.custom_colors):
            # Fallback to default colors if class_id exceeds custom palette size
            colors = plt.cm.get_cmap('tab10', len(self.class_names))
            color = colors(class_id)
            return tuple((np.array(color[:3]) * 255).astype(np.uint8))
        return self.custom_colors[class_id]

    def _create_legend_image(self, item_height=30, padding=5, target_width=None):
        """
        Create a legend image showing class names with framed color indicators,
        laid out in rows and justified to fit the target_width.

        Args:
            item_height (int): Height allocated for each class item.
            padding (int): Padding around elements within an item and between items/rows.
                           Padding between items is also used for justification spacing.
            target_width (int, optional): Desired total width of the legend image.
                                          If None, items are placed compactly.

        Returns:
            np.ndarray: Legend image (conceptually RGB, drawn with BGR internally).
        """
        num_classes = len(self.class_names)
        if num_classes == 0:
            # Return a small white image if no classes
            return np.ones((item_height + 2 * padding, 200, 3), dtype=np.uint8) * 255

        # --- Pre-calculate maximum text width needed ---
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        font_thickness = 1
        max_text_width = 0
        sample_text_sizes = {} # Cache sizes for reuse

        for i, class_name in enumerate(self.class_names):
            text = f"{i}: {class_name}"
            if text not in sample_text_sizes: # Cache text size
                text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                sample_text_sizes[text] = text_size
            text_width = sample_text_sizes[text][0]
            if text_width > max_text_width:
                max_text_width = text_width

        # --- Define item components dimensions ---
        frame_size = max(1, item_height - 2 * padding)  # Ensure frame has positive size
        frame_and_text_width = padding + frame_size + padding + max_text_width + padding # Frame + padding + text + padding

        # --- Determine layout based on target_width ---
        items_per_row = num_classes # Default: all items in one row if no target width
        if target_width is not None and target_width > 0:
            # Calculate minimum space needed per item
            min_item_space = frame_and_text_width
            # Calculate how many items can fit based on minimum space
            # Need space for items and paddings between them and outer paddings
            # Total space needed = items * min_item_space + (items + 1) * padding
            # Solve for items: items * (min_item_space + padding) + padding <= target_width
            # items <= (target_width - padding) / (min_item_space + padding)
            if (min_item_space + padding) > 0:
                 max_possible_items_float = (target_width - padding) / (min_item_space + padding)
                 items_per_row = max(1, min(num_classes, int(max_possible_items_float)))
            else:
                 items_per_row = 1 # Fallback if calculation fails

        items_per_row = max(1, items_per_row)
        num_rows = int(np.ceil(num_classes / items_per_row))

        # --- Calculate final legend dimensions ---
        # Height is straightforward
        legend_height = num_rows * item_height + (num_rows + 1) * padding

        # Width depends on whether we are justifying or not
        if target_width is not None and target_width > 0 and items_per_row > 1:
            legend_width = target_width
        else:
            # Compact width if no target or only one item per row
            legend_width = items_per_row * frame_and_text_width + (items_per_row + 1) * padding


        # --- Create the legend image ---
        legend_bgr = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255

        # --- Draw items row by row ---
        for i, class_name in enumerate(self.class_names):
            color_rgb = self._get_color_for_class(i)
            color_bgr = color_rgb[::-1] # Convert RGB to BGR for OpenCV

            row = i // items_per_row
            col = i % items_per_row

            # --- Calculate base item position ---
            # Items in the row will be distributed to justify across legend_width
            items_in_this_row = min(items_per_row, num_classes - row * items_per_row)

            if target_width is not None and target_width > 0 and items_in_this_row > 1:
                # --- Justified Layout ---
                # Total space occupied by items themselves in this row
                total_item_content_width = items_in_this_row * frame_and_text_width
                # Total padding space available for distribution
                total_available_padding_space = legend_width - total_item_content_width
                if total_available_padding_space > 0:
                    # Number of gaps between/around items (left of first, between items, right of last)
                    num_gaps = items_in_this_row + 1
                    # Base padding per gap
                    base_gap_padding = total_available_padding_space // num_gaps
                    # Extra pixels to distribute starting from the left
                    extra_pixels = total_available_padding_space % num_gaps

                    # Calculate cumulative x-position for the item
                    # Sum of paddings before this item
                    cumul_padding = (col + 1) * base_gap_padding
                    # Add extra pixels for gaps up to this item's left gap
                    cumul_padding += min(col + 1, extra_pixels)

                    item_left_x = cumul_padding + col * frame_and_text_width
                else:
                   # Fallback if no extra space (shouldn't happen often with pre-calculation)
                   item_left_x = col * frame_and_text_width + (col + 1) * padding
            else:
                # --- Compact Layout (no justification or single item row) ---
                item_left_x = col * frame_and_text_width + (col + 1) * padding


            item_top_y = row * item_height + (row + 1) * padding

            # --- Draw frame ---
            frame_top_left = (item_left_x + padding, item_top_y + padding)
            frame_bottom_right = (frame_top_left[0] + frame_size, frame_top_left[1] + frame_size)
            cv2.rectangle(
                legend_bgr,
                frame_top_left,
                frame_bottom_right,
                color_bgr,
                thickness=2,
                lineType=cv2.LINE_AA
            )

            # --- Draw text ---
            text = f"{i}: {class_name}"
            # Position text to the right of the frame
            text_x = frame_bottom_right[0] + padding
            text_y = item_top_y + item_height // 2 + 5 # Approximate vertical centering

            cv2.putText(
                legend_bgr,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (0, 0, 0), # Black text
                font_thickness,
                cv2.LINE_AA
            )

        return legend_bgr

    def _save_prediction_visualization(self, original_image, prediction, output_path):
        """
        Save visualization of prediction result with legend.
        All internal processing uses RGB format.

        Args:
            original_image (np.ndarray): Original image (RGB)
            prediction (np.ndarray): Prediction mask
            output_path (str): Path to save visualization
        """

        # Ensure original_image is RGB (it should be based on your loading)
        assert original_image.ndim == 3 and original_image.shape[2] == 3, "Input image must be RGB"

        # Create an empty mask for drawing edges (in RGB)
        edge_mask = np.zeros((*prediction.shape, 3), dtype=np.uint8)

        # Iterate through classes (skip background)
        for class_id in range(1, len(self.class_names)):
            mask = (prediction == class_id).astype(np.uint8)

            if not np.any(mask):  # Skip if no pixels belong to this class
                continue

            # Find contours or use morphological operations for edges
            # Using morphological gradient often gives cleaner lines than dilate-erode diff
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # Morphological Gradient: difference between dilation and erosion
            edges = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)
            # edges is a binary mask where edge pixels are 1 (255 if kept as uint8 after operation)

            # Ensure edges is binary (0 or 1)
            edges = (edges > 0).astype(np.uint8)

            if np.any(edges):
                # Get the RGB color for this class
                color_rgb = self._get_color_for_class(class_id)  # Returns (R, G, B) tuple

                # Apply the color to the edge_mask where edges are present
                # This broadcasts the color tuple across the selected pixels
                edge_mask[edges == 1] = color_rgb
                # Alternative using loop (slower):
                # for c in range(3):
                #     edge_mask[:, :, c] = np.where(edges == 1, color_rgb[c], edge_mask[:, :, c])

        # Overlay the edge mask onto the original image
        # Work with copies in RGB
        overlay_rgb = original_image.copy()

        # Option 1: Direct replacement (sharp edges)
        # non_black_mask = np.any(edge_mask != [0, 0, 0], axis=-1)
        # overlay_rgb[non_black_mask] = edge_mask[non_black_mask]

        # Option 2: Blending (makes edges slightly transparent)
        alpha = 0.6  # Transparency factor for edges. Adjust as needed.
        # Create a mask for where to blend (where edges exist)
        edge_locations = np.any(edge_mask != [0, 0, 0], axis=-1)
        # Blend only at edge locations
        overlay_rgb[edge_locations] = (
                alpha * edge_mask[edge_locations] +
                (1 - alpha) * overlay_rgb[edge_locations]
        ).astype(np.uint8)

        # Create the legend (already in conceptual RGB, drawn with BGR internally)
        legend_rgb = self._create_legend_image()

        # Combine overlay and legend horizontally
        # Handle potential height mismatch
        combined_height = overlay_rgb.shape[0] + legend_rgb.shape[0]
        combined_width = max(overlay_rgb.shape[1], legend_rgb.shape[1])
        # Create a white canvas in RGB
        combined_rgb = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255

        # Place the overlay image
        combined_rgb[:overlay_rgb.shape[0], :overlay_rgb.shape[1]] = overlay_rgb
        # Place the legend image
        combined_rgb[overlay_rgb.shape[0]:, :legend_rgb.shape[1]] = legend_rgb

        # Convert final combined RGB image to BGR for saving with OpenCV
        combined_bgr = cv2.cvtColor(combined_rgb, cv2.COLOR_RGB2BGR)

        # Save the result
        success = cv2.imwrite(output_path, combined_bgr)
        if not success:
            print(f"Warning: Failed to save image to {output_path}")

    def _save_comparison_visualization(self, original_image, ground_truth, prediction, output_path):
        """
        Save comparison visualization between ground truth and prediction with legend.

        Args:
            original_image (np.ndarray): Original image (RGB)
            ground_truth (np.ndarray): Ground truth mask
            prediction (np.ndarray): Prediction mask
            output_path (str): Path to save visualization
        """

        # --- Helper function to create edge mask ---
        def create_edge_mask(mask_data):
            edge_msk = np.zeros((*mask_data.shape, 3), dtype=np.uint8)
            for cls_id in range(1, len(self.class_names)):  # Skip background
                msk = (mask_data == cls_id).astype(np.uint8)
                if np.any(msk):
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    eroded = cv2.erode(msk, kernel, iterations=1)
                    dilated = cv2.dilate(msk, kernel, iterations=1)
                    edges = dilated - eroded
                    color = self._get_color_for_class(cls_id)  # Returns (R, G, B) tuple
                    # Apply color to edge mask (convert RGB tuple to BGR for OpenCV)
                    bgr_color = color[::-1]  # (B, G, R)
                    for c in range(3):
                        edge_msk[:, :, c] = np.where(edges > 0, bgr_color[c], edge_msk[:, :, c])
                    # Or: edge_msk[edges > 0] = bgr_color
            return edge_msk

        # --- Generate edge masks ---
        edge_mask_gt = create_edge_mask(ground_truth)
        edge_mask_pred = create_edge_mask(prediction)

        # --- Prepare original image copies (convert to BGR once) ---
        original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)

        # --- Overlay edges onto original image copies ---
        def apply_edges(img_bgr, edge_msk):
            img_copy = img_bgr.copy()
            non_black = np.any(edge_msk != [0, 0, 0], axis=-1)
            img_copy[non_black] = edge_msk[non_black]
            return img_copy

        gt_overlay_bgr = apply_edges(original_bgr, edge_mask_gt)
        pred_overlay_bgr = apply_edges(original_bgr, edge_mask_pred)

        # --- Combine overlays side-by-side ---
        # Calculate the combined width here
        combined_comparison_width = gt_overlay_bgr.shape[1] + pred_overlay_bgr.shape[1]

        try:
            # Concatenate along the width (axis=1)
            comparison = np.concatenate((gt_overlay_bgr, pred_overlay_bgr), axis=1)
        except ValueError as e:
            print(f"Error concatenating images for comparison: {e}")
            return  # Or handle the error appropriately

        # --- Add labels ---
        height, width = comparison.shape[:2]
        gt_width = gt_overlay_bgr.shape[1]  # Width of the Ground Truth part
        pred_text_x = gt_width + 10  # Offset from the start of the Prediction part
        cv2.putText(comparison, 'Ground Truth', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(comparison, 'Prediction', (pred_text_x, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # --- Create legend matching the combined width ---
        # **** KEY CHANGE HERE ****
        # Pass the calculated combined_comparison_width to _create_legend_image
        legend = self._create_legend_image(target_width=combined_comparison_width)
        # **** END KEY CHANGE ****

        # --- Combine comparison and legend ---
        # Ensure the legend width matches the comparison width before vertical stacking
        # (Handled by passing target_width to _create_legend_image)
        combined_height = comparison.shape[0] + legend.shape[0]
        # Widths should now match closely due to target_width specification
        combined_width = max(comparison.shape[1], legend.shape[1])

        # Create a white background canvas for the final image
        combined = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255

        # Place the comparison image on the top part of the canvas
        combined[:comparison.shape[0], :comparison.shape[1]] = comparison
        # Place the (now correctly sized) legend image on the bottom part
        combined[comparison.shape[0]:, :legend.shape[1]] = legend

        # --- Save result ---
        # `comparison` and `combined` are already in BGR format, suitable for cv2.imwrite
        success = cv2.imwrite(output_path, combined)
        if not success:
            print(f"Warning: Failed to save comparison visualization to {output_path}")

    def _show_prediction(self, original_image, prediction):
        """
        Display prediction result with legend.

        Args:
            original_image (np.ndarray): Original image
            prediction (np.ndarray): Prediction mask
        """
        # Colorize prediction
        colored_mask = np.zeros((*prediction.shape, 3), dtype=np.uint8)
        for class_id in range(len(self.class_names)):
            colored_mask[prediction == class_id] = self._get_color_for_class(class_id)

        # Display
        fig = plt.figure(figsize=(15, 5))

        # Original image
        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(original_image)
        ax1.set_title('Original Image')
        ax1.axis('off')

        # Prediction mask
        ax2 = plt.subplot(1, 3, 2)
        ax2.imshow(prediction, cmap='tab10')
        ax2.set_title('Prediction Mask')
        ax2.axis('off')

        # Overlay
        overlay = cv2.addWeighted(original_image, 0.5, colored_mask, 0.5, 0)
        ax3 = plt.subplot(1, 3, 3)
        ax3.imshow(overlay)
        ax3.set_title('Overlay')
        ax3.axis('off')

        # Add legend
        legend_elements = []
        for i, class_name in enumerate(self.class_names):
            color = self._get_color_for_class(i)
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=np.array(color)/255))

        fig.legend(legend_elements, self.class_names, loc='lower center', ncol=len(self.class_names))

        plt.tight_layout()
        plt.show()

    def _show_comparison(self, original_image, ground_truth, prediction):
        """
        Display comparison between ground truth and prediction with legend.

        Args:
            original_image (np.ndarray): Original image
            ground_truth (np.ndarray): Ground truth mask
            prediction (np.ndarray): Prediction mask
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(ground_truth, cmap='tab10')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')

        axes[2].imshow(prediction, cmap='tab10')
        axes[2].set_title('Prediction')
        axes[2].axis('off')

        # Add legend
        legend_elements = []
        for i, class_name in enumerate(self.class_names):
            color = plt.cm.get_cmap('tab10')(i)
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color))

        fig.legend(legend_elements, self.class_names, loc='lower center', ncol=len(self.class_names))

        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Check Dam Segmentation Prediction')
    parser.add_argument('--model', type=str, default='UNet', choices=['FCN', 'UNet', 'CheckDamNet'])
    parser.add_argument('--checkpoint', type=str,
                        default=r"C:\Users\Kevin\Desktop\TheSotrageCapacityOfCheckDam\CheckDamExtraction\lightning_logs\version_1\checkpoints\epoch=99-step=12500.ckpt",
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str,
                        default=r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SpatialInfoExtraction\Google\243.tif",
                        help='Path to input image')
    parser.add_argument('--json', type=str,
                        default=r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SpatialInfoExtraction\Label\243.json",
                        help='Path to JSON annotation file (optional)')
    parser.add_argument('--output', type=str, default='prediction_result.png',
                        help='Path to save prediction result')
    parser.add_argument('--labels_file', type=str,
                        default=r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SpatialInfoExtraction\labels.txt",
                        help='Path to labels file')
    parser.add_argument('--show', action='store_true',
                        help='Show prediction result')
    parser.add_argument('--batch', action='store_true',
                        help='Process multiple images in batch mode')
    parser.add_argument('--image_dir', type=str,
                        default=r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SpatialInfoExtraction\Google",
                        help='Directory containing images for batch processing')
    parser.add_argument('--json_dir', type=str,
                        default=r"C:\Users\Kevin\Documents\PythonProject\CheckDam\Datasets\SpatialInfoExtraction\Label",
                        help='Directory containing JSON annotations for batch processing')

    args = parser.parse_args()

    # Load class names
    with open(args.labels_file) as f:
        class_names = [line.strip() for line in f.readlines()]

    # Initialize predictor
    predictor = CheckDamPredictor(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        class_names=class_names
    )

    if args.batch:
        # Get all image files from directory
        ids = [51423, 1974, 2285, 1205, 243, 3118, 3184, 1721, 1824, 1945, 2161, 2805, 58, 117, 1624, 1632, 2866, 352, 51725, 1378, 878, 1459, 1485, 1602, 3140, 41341, 6925, 7546]

        image_paths = [os.path.join(args.image_dir, f"{_id}.tif") for _id in ids]
        json_paths = [os.path.join(args.json_dir, f"{_id}.json") for _id in ids]

        # Process multiple images with JSON annotations
        predictions = predictor.predict_multiple_images_with_json(
            image_paths=image_paths,
            json_paths=json_paths,
            output_dir="predictions",
            show=args.show
        )
        print(f"Batch prediction completed. Results saved to 'predictions' folder.")
    else:
        # Run prediction
        if args.json:
            # Single image prediction
            ground_truth, prediction = predictor.predict_with_json_annotation(
                image_path=args.image,
                json_path=args.json,
                output_path=args.output,
                show=args.show
            )
            print(f"Comparison completed. Result saved to {args.output}")
        else:
            # Simple prediction
            prediction = predictor.predict_single_image(
                image_path=args.image,
                output_path=args.output,
                show=args.show
            )
            print(f"Prediction completed. Result saved to {args.output}")

if __name__ == "__main__":
    main()
