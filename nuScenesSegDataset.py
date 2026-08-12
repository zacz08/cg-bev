import json
import cv2
import re
import torch
import numpy as np
from torch.utils.data import Dataset

class nuScenesSegDataset(Dataset):
    """BEV segmentation dataset with BEV-legal augmentation.

    Augmentation policy (~4x effective dataset):
        Geometric symmetries that keep the BEV scene physically plausible.
    When ``augment=True``, the dataset is deterministically expanded to
    four concatenated blocks: original, x-flip, y-flip, and 180-degree
    rotation. Each item therefore receives exactly one state, never a
    stacked composition.
          - **x-flip** (left/right mirror) — legal: BEV scenes are
            statistically symmetric about the ego-forward axis.
          - **y-flip** (front/back mirror) — adds the only systematic shift
            (rear-facing layouts) but is essentially free vs. the cost of
            having 4x less data; included per appendix C.
      - **180-degree rotation** — composition of x- and y-flip; equivalent
            to swapping ego heading. Same caveat as y-flip.
    """

    def __init__(self, data_split, resolution=512, augment=False, return_feature=False, model='lss'):
        self.augment = augment
        self.model = model
        self.resolution = resolution
        self.data_split = data_split
        self.return_feature = return_feature
        self.aug_ops = ('original', 'xflip', 'yflip', 'rot180')

        assert data_split in ['train','val','test','mini_train','mini_val'], \
            f"Invalid data_split: {self.data_split}"
        assert model in ['lss', 'stp3', 'bevformer', 'bevfusion', 'vggt'], \
            f"Invalid model: {model}. Supported models are 'lss', 'stp3', 'bevformer', 'bevfusion', and 'vggt'."

        # CLDM training requires aligned (bev_feat, pred_map, bev_map_gt) triplets.
        # Augmenting only `bev_map_gt` would break this alignment. Force-off here
        # so the user gets an explicit signal rather than silently bad metrics.
        if self.return_feature and self.augment:
            print("[nuScenesSegDataset] WARNING: augment=True is unsafe when "
                  "return_feature=True (CLDM); forcing augment=False.")
            self.augment = False
        json_file = "./data/nuscenes/prompt_"+ model + "_" +self.data_split + ".json"
        self.data = []
        with open(json_file, 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def bev_legal_augment(self, img, op):
        """Apply one deterministic BEV augmentation state.

        Args:
            img: numpy array of shape (H, W, C).
            op: one of ``original``, ``xflip``, ``yflip``, ``rot180``.

        Returns:
            Augmented numpy array of the same shape.
        """
        if op == 'original':
            return img
        if op == 'xflip':
            return np.ascontiguousarray(img[:, ::-1, :])
        if op == 'yflip':
            return np.ascontiguousarray(img[::-1, :, :])
        if op == 'rot180':
            return np.ascontiguousarray(img[::-1, ::-1, :])
        raise ValueError(f"Unknown BEV augmentation op: {op}")

    
    def get_sample_token_by_filename(self, filename):
        """
        Extracts the sample token from the filename.
        The filename format is expected to be '00000_bev_gt_da1c10a971e84fd1b80c623521fc186f.npy'.
        """
        match = re.search(r'([a-f0-9]{32})', filename)
        assert match is not None, f"Filename {filename} does not contain a valid token."
        token = match.group(1)
        return token

    def __len__(self):
        if self.augment:
            return len(self.data) * len(self.aug_ops)
        return len(self.data)

    def __getitem__(self, idx):
        aug_op = 'original'
        if self.augment:
            base_len = len(self.data)
            aug_op = self.aug_ops[idx // base_len]
            idx = idx % base_len

        item = self.data[idx]

        file_pred_map = item['pred_map']
        file_bev_map_gt = item['bev_map_gt']
        sample_token = self.get_sample_token_by_filename(file_bev_map_gt)

        pred_map = np.load(f'./data/nuscenes/bev_pred_{self.model}/{self.data_split}/{file_pred_map}')
        pred_map = np.transpose(pred_map, (1, 2, 0)).astype(np.float32)
        if self.resolution == 192:
            pred_map = cv2.resize(pred_map, (self.resolution, self.resolution))
        
        bev_map_gt = np.load(f'./data/nuscenes/bev_seg_gt_mask_{self.resolution}/{self.data_split}/{file_bev_map_gt}')
        bev_map_gt = np.transpose(bev_map_gt, (1, 2, 0)).astype(np.float32)

        if self.augment:
            # Only for VAE / unconditional LDM training
            # Keep `augment=False` for CLDM.
            bev_map_gt = self.bev_legal_augment(bev_map_gt, aug_op)

        if self.return_feature :
            file_bev_feat = item['bev_feat']
            # bev_feat = torch.load(f'./data/nuscenes/bev_feat_raw_stp3/{self.data_split}/{file_bev_feat}', weights_only=True)
            bev_feat = torch.load(f'./data/nuscenes/bev_feat_{self.model}/{self.data_split}/{file_bev_feat}', weights_only=True)
            bev_feat = bev_feat.squeeze(0).permute(1, 2, 0).float()     # [1, 256, 200, 200] -> [200, 200, 256], follow CHW->HWC

            # remove the predict frame from bev_feat (for ST-P3 only)
            # bev_feat = bev_feat.squeeze(0).view(4, 64, 200, 200)[:3]    # [1, 256, 200, 200] -> [256, 200, 200] -> [4, 64, 200, 200] -> [3, 64, 200, 200]
            # bev_feat = bev_feat.view(3 * 64, 200, 200).permute(1, 2, 0)    # [192, 200, 200] -> [200, 200, 192]

        # Normalize mask to [-1, 1].
        pred_map = (pred_map * 2 - 1.0).astype(np.float32, copy=False)
        bev_map_gt = (bev_map_gt * 2 - 1.0).astype(np.float32, copy=False)
        # bev_map_gt[:, :, 1:5] = -1       # set the ignore class to -1 for 2 layers version of STP3

        if self.return_feature:
            return dict(bev_map_gt=bev_map_gt,          # bev seg mat GT (h, w, c)
                        pred_map=pred_map,              # as diffusion input ([h, w, bev_map_gt]) during training
                        bev_feat=bev_feat,              # bev feature, as controlnet input ([200, 200, channel])
                        sample_token=sample_token)      # nuScenes sample token
        return dict(bev_map_gt=bev_map_gt,
                    sample_token=sample_token)

