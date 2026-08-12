import os
import io
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from tools.mask_viewer import combine_masks_to_rgb


class ImageLogger(Callback):
    def __init__(self, batch_frequency=2000, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, image_style='normal', data_split='val', 
                 log_images_kwargs=None, log_folder=None,
                 frame_camera_size=(448, 252), frame_jpeg_target_kb=200):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.save_dir = log_folder  # unified log folder passed from training script
        self.save_counter = 0
        self.frame_camera_size = tuple(frame_camera_size)
        self.frame_jpeg_target_bytes = int(frame_jpeg_target_kb * 1024)
        self._font_cache = {}
        # self.val_sample_input = None    # to save sample during validation

        assert image_style in ['normal', 'video_frame'], \
            f"Invalid image_style: {image_style}. Choose from ['normal', 'video_frame']."
        self.image_style = image_style
        if self.image_style == 'video_frame':
            assert data_split in ['train','val','test','mini_train','mini_val'], \
            f"Invalid data_split: {data_split}"
            if 'mini' in data_split:
                ds_version = 'v1.0-mini'
            else:
                ds_version = 'v1.0-trainval'
            # Load nuScenes dataset
            from nuscenes.nuscenes import NuScenes
            self.nusc = NuScenes(version=ds_version, dataroot='./data/nuscenes', verbose=False)

    @rank_zero_only
    def log_local(self, split, images, global_step, current_epoch, batch_idx, render=True):
        root = os.path.join(self.save_dir, "image_log_" + split)
        processed_images = []
        spacing = 8

        for k in images:
            if isinstance(images[k], np.ndarray):
                images[k] = torch.from_numpy(images[k])

            masks = images[k]  # tensor of shape [B, C, H, W] or [C, H, W]
            if masks.ndim == 3:
                masks = masks.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]

            B, C, H, W = masks.shape
            row_images = []

            for c in range(C):
                for b in range(B):
                    bin_mask = (masks[b, c] > 0).cpu().numpy().astype(np.uint8)
                    rgb_mask = np.stack([bin_mask] * 3, axis=2) * 255  # [H, W, 3]
                    row_images.append(Image.fromarray(rgb_mask))

            if render:
                for b in range(B):
                    rgb_combined = combine_masks_to_rgb(masks[b].cpu().numpy())  # [H, W, 3]
                    rgb_combined = (rgb_combined * 255).astype(np.uint8)
                    row_images.append(Image.fromarray(rgb_combined))

            # combine row_image and rgb_image horizontally
            total_width = sum(im.width for im in row_images) + spacing * (len(row_images) - 1)
            row_height = row_images[0].height
            row_image = Image.new("RGB", (total_width, row_height), (255, 255, 255))

            x_offset = 0
            for idx, im in enumerate(row_images):
                row_image.paste(im, (x_offset, 0))
                x_offset += im.width
                if idx < len(row_images) - 1:
                    x_offset += spacing  # add spacing between images

            processed_images.append(row_image)

        if len(processed_images) == 0:
            return

        # Add vertical spacing between rows
        images_with_spacing = []
        for img in processed_images:
            images_with_spacing.append(img)
            blank_image = Image.new("RGB", (img.width, spacing), (255, 255, 255))
            images_with_spacing.append(blank_image)
        images_with_spacing = images_with_spacing[:-1]  # Remove last blank

        # Stack vertically
        total_height = sum(img.height for img in images_with_spacing)
        max_width = max(img.width for img in images_with_spacing)
        stacked_image = Image.new("RGB", (max_width, total_height), (255, 255, 255))

        y_offset = 0
        for img in images_with_spacing:
            stacked_image.paste(img, (0, y_offset))
            y_offset += img.height

        # Save image
        filename = "combined_gs-{:06}_e-{:06}_b-{:06}.png".format(global_step, current_epoch, batch_idx)
        path = os.path.join(root, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        stacked_image.save(path)

    @staticmethod
    def _tokens_from_batch(sample_token, count):
        if sample_token is None:
            return [None] * count
        if isinstance(sample_token, str):
            tokens = [sample_token]
        elif isinstance(sample_token, np.ndarray):
            tokens = [str(token) for token in sample_token.reshape(-1).tolist()]
        elif isinstance(sample_token, torch.Tensor):
            tokens = [str(token) for token in sample_token.detach().cpu().reshape(-1).tolist()]
        elif isinstance(sample_token, (list, tuple)):
            tokens = [str(token) for token in sample_token]
        else:
            tokens = [str(sample_token)]
        if len(tokens) < count:
            tokens.extend([None] * (count - len(tokens)))
        return tokens[:count]

    def _token_filename(self, token, extension='png'):
        self.save_counter += 1
        extension = extension.lstrip('.')
        if token:
            return f"{self.save_counter:05d}_{token}.{extension}"
        return f"{self.save_counter:05d}.{extension}"

    @staticmethod
    def _resample_filter(name):
        if hasattr(Image, 'Resampling'):
            return getattr(Image.Resampling, name)
        return getattr(Image, name)

    def _load_font(self, size):
        size = int(size)
        if size not in self._font_cache:
            try:
                self._font_cache[size] = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=size)
            except OSError:
                self._font_cache[size] = ImageFont.load_default()
        return self._font_cache[size]

    def _open_resized_camera_image(self, img_path):
        with Image.open(img_path) as img:
            img.draft("RGB", self.frame_camera_size)
            img = img.convert("RGB")
            if img.size != self.frame_camera_size:
                img = img.resize(self.frame_camera_size, self._resample_filter('LANCZOS'))
            return img

    def _save_frame_image(self, image, path):
        image = image.convert("RGB")
        target_bytes = self.frame_jpeg_target_bytes
        min_quality = 35
        quality_steps = (72, 66, 60, 54, 48, 42, min_quality)
        best_payload = None

        for _ in range(5):
            for quality in quality_steps:
                buffer = io.BytesIO()
                image.save(buffer, format='JPEG', quality=quality,
                           optimize=True, progressive=True, subsampling=2)
                payload = buffer.getvalue()
                best_payload = payload
                if len(payload) <= target_bytes or quality == min_quality:
                    break

            if best_payload is None or len(best_payload) <= target_bytes:
                break

            new_size = (max(1, int(image.width * 0.9)),
                        max(1, int(image.height * 0.9)))
            image = image.resize(new_size, self._resample_filter('LANCZOS'))

        with open(path, 'wb') as handle:
            handle.write(best_payload)

    @staticmethod
    def _mask_to_rgb_image(mask):
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        rgb_img = combine_masks_to_rgb(mask)
        rgb_img = (rgb_img * 255).astype(np.uint8)
        return Image.fromarray(rgb_img)

    @staticmethod
    def _stack_vertical(images, spacing=8):
        if not images:
            return None
        width = max(image.width for image in images)
        height = sum(image.height for image in images) + spacing * (len(images) - 1)
        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        y_offset = 0
        for image in images:
            canvas.paste(image, (0, y_offset))
            y_offset += image.height + spacing
        return canvas

    @staticmethod
    def _stack_horizontal(images, spacing=8):
        if not images:
            return None
        width = sum(image.width for image in images) + spacing * (len(images) - 1)
        height = max(image.height for image in images)
        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        x_offset = 0
        for image in images:
            canvas.paste(image, (x_offset, 0))
            x_offset += image.width + spacing
        return canvas

    @staticmethod
    def _as_batched_tensor(masks):
        if isinstance(masks, np.ndarray):
            masks = torch.from_numpy(masks)
        if masks.ndim == 3:
            masks = masks.unsqueeze(0)
        elif masks.ndim != 4:
            raise ValueError(f"Unexpected mask shape: {masks.shape}")
        return masks

    def _cldm_sample_triplets(self, images):
        ordered_keys = ["recon_ref", "samples", "gt"]
        batched_images = {
            key: self._as_batched_tensor(images[key])
            for key in ordered_keys
            if key in images
        }
        if not batched_images:
            return []

        batch_size = min(masks.shape[0] for masks in batched_images.values())
        sample_triplets = []
        for sample_index in range(batch_size):
            panels = [
                self._mask_to_rgb_image(batched_images[key][sample_index])
                for key in ordered_keys
                if key in batched_images
            ]
            stacked_image = self._stack_vertical(panels)
            if stacked_image is not None:
                sample_triplets.append(stacked_image)
        return sample_triplets

    def log_cldm_triplet_rgb(self, split, images, global_step, current_epoch, batch_idx, sample_token):
        root = os.path.join(self.save_dir, "image_log_" + split)
        os.makedirs(root, exist_ok=True)

        sample_triplets = self._cldm_sample_triplets(images)
        if not sample_triplets:
            return

        if split == "predict":
            tokens = self._tokens_from_batch(sample_token, len(sample_triplets))
            for sample_index, stacked_image in enumerate(sample_triplets):
                filename = self._token_filename(tokens[sample_index])
                stacked_image.save(os.path.join(root, filename))
            return

        grid_image = self._stack_horizontal(sample_triplets)
        if grid_image is None:
            return
        filename = "combined_rgb_gs-{:06}_e-{:06}_b-{:06}.png".format(
            global_step, current_epoch, batch_idx)
        grid_image.save(os.path.join(root, filename))


    def log_local_rgb(self, split, images, global_step, current_epoch, batch_idx, sample_token=None):
        """
        Log images as RGB masks, combining multiple masks into a single image.
        """
        root = os.path.join(self.save_dir, "image_log_" + split)
        os.makedirs(root, exist_ok=True)

        spacing = 8
        processed_images = []

        for k in images:
            masks = images[k]
            if isinstance(masks, np.ndarray):
                masks = torch.from_numpy(masks)

            # Ensure shape is [B, 4, H, W]
            if masks.ndim == 3:
                masks = masks.unsqueeze(0)  # [4, H, W] -> [1, 4, H, W]
            elif masks.ndim != 4:
                raise ValueError(f"Unexpected mask shape: {masks.shape}")

            for i in range(masks.shape[0]):
                mask4 = masks[i].cpu().numpy()  # [4, H, W]
                rgb_img = combine_masks_to_rgb(mask4)  # [H, W, 3], float32 in [0,1]
                rgb_img = (rgb_img * 255).astype(np.uint8)
                processed_images.append(rgb_img)

        if len(processed_images) == 0:
            return

        # Add blank gap between images
        images_with_spacing = []
        for img in processed_images:
            images_with_spacing.append(img)
            blank_gap = np.ones((img.shape[0], spacing, 3), dtype=np.uint8) * 255
            images_with_spacing.append(blank_gap)
        images_with_spacing = images_with_spacing[:-1]  # Remove last blank

        stacked_image = np.concatenate(images_with_spacing, axis=1)  # stack horizontally

        tokens = self._tokens_from_batch(sample_token, 1)
        if split == "predict" and tokens[0]:
            filename = self._token_filename(tokens[0])
        else:
            filename = "combined_rgb_gs-{:06}_e-{:06}_b-{:06}.png".format(global_step, current_epoch, batch_idx)
        path = os.path.join(root, filename)
        Image.fromarray(stacked_image).save(path)


    def log_frame(self, split, images, global_step, current_epoch, batch_idx, sample_token):
        """
        Log superimposed mask with camera images.
        """
        root = os.path.join(self.save_dir, "image_log_" + split)
        os.makedirs(root, exist_ok=True)

        if 'samples' not in images:
            return

        sample_masks = self._as_batched_tensor(images['samples'])
        pred_masks = None
        for pred_key in ('recon_ref', 'ref'):
            if pred_key in images:
                pred_masks = self._as_batched_tensor(images[pred_key])
                break
        gt_masks = self._as_batched_tensor(images['gt']) if 'gt' in images else None

        tokens = self._tokens_from_batch(sample_token, sample_masks.shape[0])
        for i in range(sample_masks.shape[0]):
            rgb_mask = self._mask_to_rgb_image(sample_masks[i])
            rgb_pred = None
            if pred_masks is not None and i < pred_masks.shape[0]:
                rgb_pred = self._mask_to_rgb_image(pred_masks[i])
            rgb_gt = None
            if gt_masks is not None and i < gt_masks.shape[0]:
                rgb_gt = self._mask_to_rgb_image(gt_masks[i])

            token = tokens[i]
            if token is None:
                continue
            sample = self.nusc.get('sample', token)
            cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
                         'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']
            img_dict = {}
            for cam in cam_names:
                sd_token = sample['data'][cam]
                sd_record = self.nusc.get('sample_data', sd_token)
                img_path = os.path.join(self.nusc.dataroot, sd_record['filename'])
                img = self._open_resized_camera_image(img_path)
                img_dict[cam] = img

            combined_img = self.stack_camera_imgs_with_bev(img_dict, rgb_mask, rgb_pred, rgb_gt)
            if split == "predict":
                filename = self._token_filename(token, extension='jpg')
            else:
                filename = "frame_gs-{:06}_e-{:06}_b-{:06}_i-{:02}.jpg".format(
                    global_step, current_epoch, batch_idx, i)
            self._save_frame_image(combined_img, os.path.join(root, filename))

    def stack_camera_imgs_with_bev(self, img_dict, rgb_mask, pred_mask=None, gt_mask=None, gap=8):
        W, H = img_dict['CAM_FRONT'].size
        gap_color = (255, 255, 255)
        row_width = W * 3 + gap * 2
        camera_label_font_size = max(12, min(28, W // 18))
        bev_caption_font_size = max(14, min(24, W // 18))
        legend_font_size = max(18, min(30, W // 14))
        camera_label_font = self._load_font(camera_label_font_size)
        legend_font = self._load_font(legend_font_size)

        # --- Step 1: define camera label height ---
        camera_label_height = max(camera_label_font_size + 10, H // 7)

        # --- Step 2: create top and bottom camera label areas ---
        top_label_area = Image.new('RGB', (row_width, camera_label_height), gap_color)
        bottom_label_area = Image.new('RGB', (row_width, camera_label_height), gap_color)

        # --- Step 3: stitch camera images into two rows ---
        row1 = Image.new('RGB', (row_width, H), gap_color)
        row1.paste(img_dict['CAM_FRONT_LEFT'], (0, 0))
        row1.paste(img_dict['CAM_FRONT'], (W + gap, 0))
        row1.paste(img_dict['CAM_FRONT_RIGHT'], (2 * W + 2 * gap, 0))

        row2 = Image.new('RGB', (row_width, H), gap_color)
        row2.paste(img_dict['CAM_BACK_LEFT'], (0, 0))
        row2.paste(img_dict['CAM_BACK'], (W + gap, 0))
        row2.paste(img_dict['CAM_BACK_RIGHT'], (2 * W + 2 * gap, 0))

        # --- Step 4: stack all camera parts: top label + rows + bottom label ---
        cam_panel_height = camera_label_height + H + gap + H + camera_label_height
        cam_panel = Image.new('RGB', (row_width, cam_panel_height), gap_color)
        cam_panel.paste(top_label_area, (0, 0))
        cam_panel.paste(row1, (0, camera_label_height))
        cam_panel.paste(row2, (0, camera_label_height + H + gap))
        cam_panel.paste(bottom_label_area, (0, cam_panel_height - camera_label_height))

        # --- Step 5: compute BEV layout: label + maps + legend ---
        if isinstance(rgb_mask, np.ndarray):
            rgb_mask = Image.fromarray(rgb_mask.astype('uint8'))
        if isinstance(pred_mask, np.ndarray):
            pred_mask = Image.fromarray(pred_mask.astype('uint8'))
        if isinstance(gt_mask, np.ndarray):
            gt_mask = Image.fromarray(gt_mask.astype('uint8'))

        bev_sources = []
        if pred_mask is not None:
            bev_sources.append(("Original Segmentation", pred_mask))
        bev_sources.append(("CG-BEV Refined Segmentation", rgb_mask))
        if gt_mask is not None:
            bev_sources.append(("Ground Truth", gt_mask))

        # Align the BEV map area exactly with the 3x2 camera image area.
        raw_img_height = cam_panel.height - 2 * camera_label_height  # 2H + gap
        image_top_y = camera_label_height
        mask_target_height = raw_img_height

        resized_bev_maps = []
        for title, bev_image in bev_sources:
            mask_orig_w, mask_orig_h = bev_image.size
            scale = mask_target_height / mask_orig_h
            mask_new_w = int(mask_orig_w * scale)
            resized = bev_image.resize((mask_new_w, mask_target_height), self._resample_filter('BILINEAR'))
            resized_bev_maps.append((title, resized))
        bev_maps_width = sum(image.width for _, image in resized_bev_maps) + gap * (len(resized_bev_maps) - 1)

        colors = [
            (100, 100, 100),      # drivable_area
            (255, 178, 102),      # ped_crossing
            (220, 220, 100),      # walkway
            (255, 50, 50),        # stop_line
            (160, 120, 60),       # carpark_area
            (240, 240, 240),      # lane_divider
        ]
        labels = [
            "Drivable Area", "Ped Crossing", "Walkway",
            "Stop Line", "Carpark Area", "Lane Divider"
        ]
        box_w = max(40, int(W * 0.12))
        box_h = max(28, int(H * 0.11))
        legend_padding_x = max(14, int(W * 0.035))
        legend_padding_y = max(14, int(H * 0.055))
        legend_text_bbox = legend_font.getbbox("Drivable Area")
        legend_text_height = legend_text_bbox[3] - legend_text_bbox[1]
        legend_columns = 3 if bev_maps_width >= 900 else 2 if bev_maps_width >= 650 else 1
        legend_rows = (len(labels) + legend_columns - 1) // legend_columns
        legend_row_gap = max(22, int(H * 0.09))
        legend_row_height = max(box_h, legend_text_height) + legend_row_gap
        legend_height = max(
            128,
            int(H * 0.55),
            legend_padding_y * 2 + legend_rows * legend_row_height - legend_row_gap,
        )

        # --- Step 6: compute final canvas size ---
        bev_panel_height = image_top_y + mask_target_height + legend_height
        final_height = max(cam_panel.height, bev_panel_height)
        final_width = cam_panel.width + gap + bev_maps_width

        # --- Step 7: create canvas and paste camera + BEV components ---
        canvas = Image.new('RGB', (final_width, final_height), gap_color)
        canvas.paste(cam_panel, (0, 0))
        draw = ImageDraw.Draw(canvas)

        def draw_centered_text(text, x, y, width, height, font):
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = x + (width - text_width) // 2 - bbox[0]
            text_y = y + (height - text_height) // 2 - bbox[1]
            draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)

        def fit_font_to_width(text, preferred_size, max_width, min_size=14):
            size = preferred_size
            text_padding_x = max(6, int(W * 0.02))
            while size > min_size:
                font = self._load_font(size)
                bbox = draw.textbbox((0, 0), text, font=font)
                if bbox[2] - bbox[0] <= max_width - 2 * text_padding_x:
                    return font
                size -= 2
            return self._load_font(min_size)

        bev_x = cam_panel.width + gap
        for title, bev_image in resized_bev_maps:
            title_font = fit_font_to_width(title, bev_caption_font_size, bev_image.width)
            draw_centered_text(title, bev_x, 0, bev_image.width, image_top_y, title_font)
            canvas.paste(bev_image, (bev_x, image_top_y))
            bev_x += bev_image.width + gap

        # --- Step 8: draw camera labels outside the images ---
        top_labels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']
        bottom_labels = ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        for i, label in enumerate(top_labels):
            x = i * (W + gap)
            draw_centered_text(label, x, 0, W, camera_label_height, camera_label_font)

        for i, label in enumerate(bottom_labels):
            x = i * (W + gap)
            y = cam_panel_height - camera_label_height
            draw_centered_text(label, x, y, W, camera_label_height, camera_label_font)

        # --- Step 9: draw semantic legend below BEV ---
        legend_start_x = cam_panel.width + gap
        legend_start_y = image_top_y + mask_target_height + legend_padding_y
        col_w = bev_maps_width // legend_columns

        for i, (color, label) in enumerate(zip(colors, labels)):
            row = i // legend_columns
            col = i % legend_columns
            x = legend_start_x + col * col_w + legend_padding_x
            y = legend_start_y + row * legend_row_height
            draw.rectangle([x, y, x + box_w, y + box_h], fill=color, outline=(0, 0, 0))
            text_bbox = draw.textbbox((0, 0), label, font=legend_font)
            text_height = text_bbox[3] - text_bbox[1]
            text_y = y + (box_h - text_height) // 2 - text_bbox[1]
            draw.text((x + box_w + legend_padding_x, text_y), label, fill=(0, 0, 0), font=legend_font)

        return canvas


    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx  # if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)
            
            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            ## two log styles for ldm and cldm
            sample_token = batch.get('sample_token', None)
            if self.image_style == 'video_frame':
                # Convert each image to a video frame
                self.log_frame(split, images,
                               pl_module.global_step, pl_module.current_epoch, batch_idx,
                               sample_token)
                    
            elif hasattr(pl_module, "control_model"):   # for cldm
                self.log_cldm_triplet_rgb(split, images,
                                          pl_module.global_step, pl_module.current_epoch, batch_idx,
                                          sample_token)
            else:   # for ldm
                self.log_local_rgb(split, images,
                                   pl_module.global_step, pl_module.current_epoch, batch_idx,
                                   sample_token=sample_token)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        return check_idx % self.batch_freq == 0

    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #     if not self.disabled:
    #         self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="val")

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="predict")