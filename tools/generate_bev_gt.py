import argparse
from dataset_processing import NuScenesBevSegMap


def parse_args():
    parser = argparse.ArgumentParser(description="Generate BEV segmentation ground truth for nuScenes dataset.")
    parser.add_argument("--ds-version", type=str, required=True, 
                        help="NuScenes dataset version.", 
                        choices=["v1.0-trainval", "v1.0-mini"])
    parser.add_argument("--data-split", type=str, required=True, choices=["train", "val", "test", "mini_train", "mini_val"],
                        help="Data split to process (train/val/test/mini_train/mini_val).")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Resolution of the output ground truth mask. Default is 512.")
    return parser.parse_args()

def main():
    """
    The splits are as follows:
    - train/val/test: The standard splits of the nuScenes dataset (700/150/150 scenes).
    - mini_train/mini_val: Train and val splits of the mini subset used for visualization and debugging (8/2 scenes).
    """

    args = parse_args()
    map_processer = NuScenesBevSegMap(
        ds_root='./data/nuscenes',
        ds_version=args.ds_version,
        data_split=args.data_split
        )

    out_folder_path = f'./data/nuscenes/bev_seg_gt_mask_{args.resolution}/{args.data_split}/'

    map_processer.generate_bev_gt(out_folder_path, format='mask', resolution=args.resolution)


if __name__ == '__main__':
	main()