import cv2
import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes import NuScenesExplorer, NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
from pyquaternion import Quaternion
from typing import List, Tuple
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


# ---------------------------------------------------------------------------
# Shapely 2.x compatibility shim for nuscenes-devkit.
# In Shapely >= 2, MultiLineString / MultiPolygon are no longer iterable;
# callers must use the `.geoms` accessor. nuscenes-devkit's
# `NuScenesMapExplorer.mask_for_lines` still does `for line in lines:` on a
# MultiLineString, which raises `TypeError: 'MultiLineString' object is not
# iterable` whenever a sample's `lane_divider` / `road_divider` geometry
# happens to be multi-part (intermittent, depends on the patch contents).
# Patch the staticmethod once at import time.
def _mask_for_lines_shapely2(lines, mask):
    if lines.geom_type == 'MultiLineString':
        line_iter = lines.geoms
    else:
        line_iter = [lines]
    for line in line_iter:
        coords = np.asarray(list(line.coords), np.int32).reshape((-1, 2))
        cv2.polylines(mask, [coords], False, 1, 2)
    return mask


NuScenesMapExplorer.mask_for_lines = staticmethod(_mask_for_lines_shapely2)

'''
1. Generate rendered BEV map
2. Draw all annotated bounding boxes (filled) for each four BEV map
3. Crop BEV map to a preset size based on ego vehicle coordinate
'''


class NuScenesBevSegMap():

    def __init__(self, ds_root, ds_version, data_split):
        self.dataroot = ds_root
        self.data_split = data_split
        assert data_split in ['train', 'val', 'test', 'mini_train', 'mini_val'], 'Invalid dataset split!'
        self.nusc = NewNuScenes(version=ds_version, dataroot=ds_root, verbose=True)
        self.nusc_can = NuScenesCanBus(dataroot=self.dataroot)
        self.scene2map = get_scene2map(self.nusc)
        self.map_layer_list = [
            'drivable_area',
            # 'road_segment',   # Note: this layer is totally same as drivable_area
            # 'road_block',     # Note: enable road_block layer will extremely increase time cost
            # 'lane',           # consider to comment it if enable lane_divider
            'ped_crossing',
            'walkway',
            'stop_line',        # an area where the ego vehicle must stop
            'carpark_area',
            # 'road_divider',
            'lane_divider'
        ]

    def get_scenes(self):
        # filter by scene split
        scenes_list = create_splits_scenes()[self.data_split][:]

        # blacklist = [419] + self.nusc_can.can_blacklist  # # remove scenes that don't have vehicle CAN bus data
        # blacklist = ['scene-' + str(scene_no).zfill(4) for scene_no in blacklist]

        # for scene_no in blacklist:
        #     if scene_no in scenes_list:
        #         scenes_list.remove(scene_no)

        scenes = [scene for scene in self.nusc.scene if scene['name'] in scenes_list]

        return scenes


    def generate_bev_gt(self, out_folder, format='image', resolution=512):
        '''
        Generate BEV segmentation ground truth for each keyframe
        :param split: The splits are as follows:
                    - train/val/test: The standard splits of the nuScenes dataset (700/150/150 scenes).
                    - mini_train/mini_val: Train and val splits of the mini subset used for visualization and debugging (8/2 scenes).
        :param out_folder: folder path to save the generated BEV segmentation ground truth
        :param format: format of the output image, 'image' or 'mask'
        '''

        assert format in ['image', 'mask'], 'Invalid format!'

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # split dataset
        splited_scenes = self.get_scenes()

        scene_num = len(splited_scenes)
        sample_num = 0
        ann_num = 0

        # for scene in self.nusc.scene:
        for scene in tqdm.tqdm(splited_scenes):
            self.sample = None
            mapname = self.scene2map[scene['name']]
            self.nusc.explorer.update_map(mapname)     # update map based on scene
            while True:
                if self.sample is None:
                    # get the first keyframe (sample) of each scene
                    self.sample = self.nusc.get('sample', scene['first_sample_token'])

                # calculate the number of keyframes and the number of annotations contained in each keyframe
                sample_num += 1
                ann_num += len(self.sample['anns'])

                ## Render BEV segmentation ground truth
                BEV_data = self.nusc.get('sample_data', self.sample['data']['LIDAR_TOP'])
                if out_folder is not None:
                    img_name = f"{sample_num:05d}_bev_gt_{self.sample['token']}.jpg"
                    img_name = os.path.join(out_folder, img_name)
                else:
                    img_name = None

                if format == 'image':
                    self.nusc.explorer.render_sample_data(BEV_data['token'],
                                                          with_anns=False,
                                                          layer_names=self.map_layer_list,
                                                          out_path=img_name,
                                                          verbose=False,
                                                          resolution=resolution)
                elif format == 'mask':
                    self.nusc.explorer.get_bev_mask_by_sample(BEV_data['token'],
                                                        with_anns=False,
                                                        layer_names=self.map_layer_list,
                                                        out_path=img_name,
                                                        verbose=False,
                                                        resolution=resolution)

                ## debug
                # get ego_pose
                # tra, rot = self.get_ego_pose('CAM_FRONT')

                if self.sample['next'] != '':
                    # "next": <str> -- Foreign key. Sample that follows this in time. Empty if end of scene.
                    self.sample = self.nusc.get('sample', self.sample['next'])
                else:
                    break

        print('====== Genearation Finish ====== ')
        print('Scene Num: %d\nSample Num: %d\nAnnotation Num: %d' % (scene_num, sample_num, ann_num))


    def get_ego_pose(self, sensor_type):
        '''
        Return ego vehicle location and orientation according to current sample and sensor
        :param sensor_type (str): Name of the sensor.
            Available sensor:
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK_RIGHT'
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_FRONT_LEFT'
        '''
        self.nusc.ego_pose[0] # TBC
        sensor_data = self.nusc.get('sample_data', self.sample['data'][sensor_type])
        ego_pose = self.nusc.get('ego_pose', sensor_data['ego_pose_token'])
        rotation = ego_pose['rotation']
        translation = ego_pose['translation']
        return translation, rotation

    def get_dataroot_path(self):
        return self.dataroot


class NewNuScenesExplorer(NuScenesExplorer):

    def __init__(self,
                 nusc: NuScenes,
                 data_path):
        super().__init__(nusc)
        self.dataroot = data_path
        self.map = None
        self.map_cache = {}  # Cache for map instances

        self.color_map_layer = {
            'drivable_area': [31, 119, 180],
            # 'lane': [255, 127, 14],
            'lane': [23, 190, 207],
            'ped_crossing': [44, 160, 44],
            'walkway': [148, 103, 189],
            'stop_line': [140, 86, 75],
            'carpark_area': [127, 127, 127],
            'road_divider': [188, 189, 34],
            # 'lane_divider': [23, 190, 207]
            'lane_divider': [255, 127, 14]
        }
        self.color_map_ann = {
            'vehicle': [255, 255, 0],
            'pedestrian': [214, 39, 40]
        }

    def get_map_instance(self, map_name):
        if map_name not in self.map_cache:
            self.map_cache[map_name] = NuScenesMap(dataroot=self.dataroot,
                                                   map_name=map_name)
        return self.map_cache[map_name]

    def update_map(self, map_name):
        self.map = self.get_map_instance(map_name)

    def get_bev_mask_by_sample(self,
                               sample_data_token: str,
                               with_anns: bool = False,
                               bev_range: float = 100,
                               out_path: str = None,
                               layer_names: List[str] = None,
                               verbose: bool = True,
                               resolution: int = 512) -> None:

        sd_record = self.nusc.get('sample_data', sample_data_token)

        pose = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
        patch_box = [pose['translation'][0], pose['translation'][1], bev_range, bev_range]
        ypr_rad = Quaternion(pose['rotation']).yaw_pitch_roll
        patch_angle = np.degrees(ypr_rad[0])
        canvas_size = (resolution, resolution)

        bev_height, bev_width = patch_box[2], patch_box[3]
        map_mask = self.map.explorer.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)

        for i in range(map_mask.shape[0]):
            map_mask[i] = np.flipud(map_mask[i])
            map_mask[i] = np.rot90(map_mask[i], k=1)

        # Show boxes.
        if with_anns:
            # Get boxes in lidar frame.
            _, boxes, _ = self.nusc.get_sample_data(sample_data_token, box_vis_level=BoxVisibility.ANY,
                                                use_flat_vehicle_coordinates=True)

            masks = {}
            for key in self.color_map_ann.keys():
                masks[key] = np.zeros(canvas_size, dtype=np.uint8)

            for box in boxes:
                for key in self.color_map_ann.keys():
                    if key in box.name:
                        masks[key] = get_bbox_in_bev_mask(box.corners(), masks[key], view=np.eye(4))
                        break

            for key in masks.keys():
                masks[key] = np.rot90(masks[key], k=1)
                masks[key] = np.expand_dims(masks[key], axis=0)   # (1024,1024) -> (1,1024,1024)
                map_mask = np.concatenate((map_mask, masks[key]), axis=0)

        if verbose:
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))

            axs[0, 0].imshow(map_mask[0], cmap='gray')
            axs[0, 1].imshow(map_mask[1], cmap='gray')

            if with_anns:
                axs[1, 0].imshow(map_mask[2], cmap='gray')
                axs[1, 1].imshow(map_mask[3], cmap='gray')

            # Turn off all axes
            for ax_row in axs:
                for ax in ax_row:
                    ax.axis('off')

            plt.tight_layout()
            plt.show()
            # plt.close(fig)

        # resized_mask = np.zeros((map_mask.shape[0], resolution, resolution), dtype=np.uint8)
        # for i in range(map_mask.shape[0]):
        #     resized_mask[i] = cv2.resize(map_mask[i], (resolution, resolution), interpolation=cv2.INTER_NEAREST)
        np.save(out_path.replace('.jpg', '.npy'), map_mask)


    def render_sample_data(self,
                           sample_data_token: str,
                           with_anns: bool = False,
                           box_vis_level: BoxVisibility = BoxVisibility.ANY,
                           bev_range: float = 100,
                           out_path: str = None,
                           layer_names: List[str] = None,
                           verbose: bool = True,
                           resolution: int = 512) -> None:
        '''
        This function is modified from nuscenes.render_sample_data().
        The function is to render a 2D BEV map with annotations and non_geometric_layers
        :param sample_data_token: Sample_data token.
        :param with_anns: Whether to draw box annotations.
        :param bev_range: The distance from ego center to BEV edge (in meters).
        :param layer_names: Semantic layers to be rendered in BEV map. Render all layers if = None
        :param output: Return rendered segmentation map (as image).
        '''
        sd_record = self.nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']

        if sensor_modality in ['lidar', 'radar']:
            # Render map
            pose = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
            patch_box = [pose['translation'][0], pose['translation'][1], bev_range, bev_range]
            ypr_rad = Quaternion(pose['rotation']).yaw_pitch_roll
            patch_angle = np.degrees(ypr_rad[0])
            canvas_size = (1024, 1024)
            fig, ax = self.render_map_mask(
                patch_box,
                patch_angle,
                layer_names,
                canvas_size,
                figsize = (10, 10),
                legend=False)

            # # Show ego vehicle center.
            # ax.plot(0, 0, 'x', color='red')

            # Show ego vehicle bounding box and direction
            # vehicle size: 3.45 * 1.68 (in meter)
            vehicle_w = 1.68
            vehicle_h = 3.45
            x = [-vehicle_h / 2, -vehicle_h / 2, vehicle_h / 2, vehicle_h / 2, -vehicle_h / 2]
            y = [-vehicle_w / 2, vehicle_w / 2, vehicle_w / 2, -vehicle_w / 2, -vehicle_w / 2]
            # ax.plot(x, y, 'w-', linewidth=2)
            ax.fill(x, y, 'white', edgecolor=None)


            # Show boxes.
            if with_anns:
                # Get boxes in lidar frame.
                _, boxes, _ = self.nusc.get_sample_data(sample_data_token, box_vis_level=box_vis_level,
                                                    use_flat_vehicle_coordinates=True)
                for box in boxes:
                    for key in self.color_map_ann.keys():
                        if key in box.name:
                            c = np.array(self.color_map_ann[key]) / 255.0
                            ## nuscence style bbox and color
                            # c = np.array(self.get_color(box.name)) / 255.0
                            # box.render(ax, view=np.eye(4), colors=(c, c, c))
                            render_bbox(box.corners(), ax, view=np.eye(4), color=c)
                            break

            # Limit visible range.
            ax.set_xlim(-bev_range//2, bev_range//2)
            ax.set_ylim(-bev_range//2, bev_range//2)
        elif sensor_modality == 'camera':
            # Load boxes and image.
            data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_data_token,
                                                                           box_vis_level=box_vis_level)
            data = Image.open(data_path)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 16))

            # Show image.
            ax.imshow(data)

            # Show boxes.
            if with_anns:
                for box in boxes:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))

            # Limit visible range.
            ax.set_xlim(0, data.size[0])
            ax.set_ylim(data.size[1], 0)

        else:
            raise ValueError("Error: Unknown sensor modality!")

        # ax.axis('off')
        # ax.set_title('{} {labels_type}'.format(
        #     sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
        # ax.set_aspect('equal')

        if out_path is not None:
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)

            image = cv2.imread(out_path)
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image = cv2.resize(rotated_image, (resolution, resolution))
            cv2.imwrite(out_path, image)

        if verbose:
            plt.show()

        plt.close(fig)

    def render_map_mask(self,
                        patch_box: Tuple[float, float, float, float],
                        patch_angle: float,
                        layer_names: List[str],
                        canvas_size: Tuple[int, int],
                        figsize: Tuple[int, int],
                        legend: bool = True):
        """
        Render map mask of the patch specified by patch_box and patch_angle.
        :param patch_box: Patch box defined as [x_center, y_center, height, width].
        :param patch_angle: Patch orientation in degrees.
        :param layer_names: A list of layer names to be extracted.
        :param canvas_size: Size of the output mask (h, w).
        :param figsize: Size of the figure.
        :return: The matplotlib figure and a list of axes of the rendered layers.
        """
        # if layer_names is None:
        #     layer_names = self.map_api.non_geometric_layers

        bev_height, bev_width = patch_box[2], patch_box[3]
        map_mask = self.map.explorer.get_map_mask(patch_box, patch_angle, layer_names, canvas_size)

        # If no canvas_size is specified, retrieve the default from the output of get_map_mask.
        if canvas_size is None:
            canvas_size = map_mask.shape[1:]

        # Initialize a combined mask with three channels (for RGB)
        combined_mask = np.zeros((canvas_size[0], canvas_size[1], 3), dtype=np.uint8)
        # Define colors for each layer
        # colors = plt.cm.get_cmap('tab10', len(layer_names))

        for i, layer in enumerate(map_mask):
            # color = colors(i)[:3]  # Get the RGB values of the color
            # color = np.array(color) * 255  # Scale to 0-255
            color = self.color_map_layer[layer_names[i]]
            layer_mask = layer > 0  # Create a binary mask for the current layer

            for j in range(3):  # Apply the color to the combined mask
                combined_mask[..., j] = np.where(layer_mask, color[j], combined_mask[..., j])

        # MAGIC, DO NOT DELETE
        combined_mask = np.flipud(combined_mask)
        # combined_mask = np.rot90(combined_mask, k=1)

        # Plot the combined mask
        plt.ioff()
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(combined_mask,
                  extent=[-bev_width // 2,
                          bev_width // 2,
                          -bev_height // 2,
                          bev_height // 2])
        if legend:
            ax.set_title('Combined Map Mask')
            # legend_elements = [Patch(facecolor=colors(i)[:3], edgecolor='r', label=layer_names[i]) for i in range(len(layer_names))]
            legend_elements = [Patch(facecolor=self.color_map_layer[layer_names[i]],
                                     edgecolor='r',
                                     label=layer_names[i]) for i in range(len(layer_names))]
            legend = ax.legend(handles=legend_elements, loc='upper right')

            # set legend name text to white
            for text in legend.get_texts():
                text.set_color("white")
        else:
            ax.axis('off')

        return fig, ax


class NewNuScenes(NuScenes):
    '''
    This class include NuScenesMap and NuScenesMapExplorer
    '''
    def __init__(self,
                 version: str = 'v1.0-mini',
                 dataroot: str = 'data/nuscenes',
                 verbose: bool = True,
                 map_resolution: float = 0.1):
        super().__init__(version, dataroot, verbose, map_resolution)
        self.dataroot = dataroot
        self.explorer = NewNuScenesExplorer(self,
                                            data_path=self.dataroot)


def get_scene2map(nusc):
    scene2map = {}
    for scene in nusc.scene:
        log = nusc.get('log', scene['log_token'])
        scene2map[scene['name']] = log['location']
    return scene2map

def crop_image(image: np.array,
                       x_px: int,
                       y_px: int,
                       axes_limit_px: int) -> np.array:
            x_min = int(x_px - axes_limit_px)
            x_max = int(x_px + axes_limit_px)
            y_min = int(y_px - axes_limit_px)
            y_max = int(y_px + axes_limit_px)

            cropped_image = image[y_min:y_max, x_min:x_max]

            return cropped_image

def render_bbox(corners,
               axis: Axes,
               view: np.ndarray = np.eye(3),
               normalize: bool = False,
               color: Tuple = ('b', 'r', 'k')) -> None:

        """
        Renders the box in BEV (Bird's Eye View) in the provided Matplotlib axis, with filled surface.
        :param axis: Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection if needed (for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinates.
        :param color: Color to fill the bounding box.
        """
        # Project the 3D corners to 2D (get the first 4 corners for the base in BEV)
        corners = view_points(corners, view, normalize=normalize)[:2, :]  # Get x, y coordinates (2D)

        # Extract the bottom four corners (which form the base of the box in BEV)
        bev_corners = np.array([corners.T[0], corners.T[1], corners.T[5], corners.T[4], corners.T[0]])

        # Fill the rectangle in the BEV with the specified color
        axis.fill(bev_corners[:, 0], bev_corners[:, 1], color=color, edgecolor=None)
        # axis.plot(bev_corners[:, 0], bev_corners[:, 1], color='k', linewidth=2)


def get_bbox_in_bev_mask(corners,
                               mask: np.ndarray,
                               bev_range: float = 100,
                               view: np.ndarray = np.eye(3),
                               normalize: bool = False) -> np.ndarray:
    """
    Render a 3D bounding box onto a BEV (Bird's Eye View) binary mask using OpenCV.
    Efficient alternative to matplotlib rendering.

    :param corners: shape=(3, 8), the 8 corners of a 3D bounding box.
    :param mask: shape=(H, W), binary BEV mask to draw on.
    :param bev_range: Physical range of the BEV area (in meters), assumed square. Default is 100.
    :param view: Transformation matrix, default is identity.
    :param normalize: Whether to normalize the projected coordinates.
    :return: Updated binary mask with the box rendered in it.
    """
    corners = view_points(corners, view, normalize=normalize)[:2, :]  # Project to 2D (x, y)

    # Select the 4 bottom corners of the box to define the BEV footprint
    pts = np.array([
        [corners[0, 0], corners[1, 0]],
        [corners[0, 1], corners[1, 1]],
        [corners[0, 5], corners[1, 5]],
        [corners[0, 4], corners[1, 4]]
    ], dtype=np.float32)

    H, W = mask.shape
    scale = W / bev_range  # Pixels per meter (e.g., 512 / 100 = 5.12)

    # Convert world coordinates to image pixel coordinates:
    # - X to the right → positive X
    # - Y to the top → negative Y in image space
    pts[:, 0] = pts[:, 0] * scale + (W // 2)
    pts[:, 1] = -(pts[:, 1] * scale) + (H // 2)

    pts = np.round(pts).astype(np.int32)
    cv2.fillPoly(mask, [pts], 1)

    return mask
