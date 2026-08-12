import os
import torch
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def combine_masks_to_rgb(mask):
    h, w = mask.shape[1:]
    background_color = np.array([30, 30, 30], dtype=np.uint8)
    rgb_img = np.ones((h, w, 3), dtype=np.uint8) * background_color.reshape(1, 1, 3)

    colors = [
    (100, 100, 100),      # drivable_area - asphalt gray
    (255, 178, 102),      # ped_crossing - orange
    (220, 220, 100),      # walkway - light yellow
    (255, 50, 50),        # stop_line - red
    (160, 120, 60),       # carpark_area - brown
    (240, 240, 240),      # lane_divider - white
]

    for i in range(mask.shape[0]):
        mask_i = (mask[i] > 0)  # binary mask: shape [H, W]
        color = np.array(colors[i], dtype=np.uint8).reshape(1, 1, 3)  # shape [1, 1, 3]
        rgb_img[mask_i] = color  # overwrite only where mask==1

    rgb_img = rgb_img.astype(np.float32) / 255.0  # normalize for matplotlib
    return rgb_img


class MaskViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mask + RGB Viewer")

        self.file_list = []
        self.current_index = 0

        # Button frame
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(side=tk.TOP, fill=tk.X)

        self.btn_load = tk.Button(self.button_frame, text="Select Folder", command=self.load_folder)
        self.btn_load.pack(side=tk.LEFT, padx=10)

        self.label = tk.Label(self.button_frame, text="")
        self.label.pack(side=tk.LEFT, padx=10)

        self.entry = tk.Entry(self.button_frame, width=30)
        self.entry.pack(side=tk.LEFT, padx=10)

        self.btn_jump = tk.Button(self.button_frame, text="Go", command=self.jump_to_file)
        self.btn_jump.pack(side=tk.LEFT, padx=10)

        self.btn_prev = tk.Button(self.button_frame, text="Previous", command=self.show_prev)
        self.btn_prev.pack(side=tk.LEFT, padx=10)

        self.btn_next = tk.Button(self.button_frame, text="Next", command=self.show_next)
        self.btn_next.pack(side=tk.LEFT, padx=10)

        self.btn_save = tk.Button(self.button_frame, text="Save Image", command=self.save_image)
        self.btn_save.pack(side=tk.LEFT, padx=10)

        # Canvas frame
        self.canvas_frame = tk.Frame(root)
        self.canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.current_rgb_img = None  # Store current RGB image

    def jump_to_file(self):
        name = self.entry.get().strip()
        if not name:
            return

        # Exact filename match
        matches = [i for i, path in enumerate(self.file_list) if os.path.basename(path) == name]
        
        if matches:
            self.current_index = matches[0]
            self.show_file()
        else:
            # Optional: show message if not found
            self.label.config(text=f"File not found: {name}")

    def load_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.file_list = sorted([os.path.join(folder_path, f)
                                     for f in os.listdir(folder_path)
                                     if f.endswith('.npy')])
            self.current_index = 0
            self.show_file()

    def show_file(self):
        if not self.file_list:
            return
        npy_path = self.file_list[self.current_index]
        self.label.config(text=os.path.basename(npy_path))

        masks = np.load(npy_path)
        num_layers = masks.shape[0]

        rgb_img = combine_masks_to_rgb(masks)
        self.current_rgb_img = rgb_img  # Save current RGB image

        # # downsample to 200x200 with max pooling
        # masks_tensor = torch.from_numpy(masks).unsqueeze(0).float()  # [1, 4, H, W]
        # masks_200 = torch.nn.functional.adaptive_max_pool2d(masks_tensor, output_size=(200, 200))
        # masks_200 = masks_200.squeeze(0).numpy()  # [4, 200, 200]

        # rgb_img_200 = combine_masks_to_rgb(masks_200)

        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        fig, axes = plt.subplots(1, num_layers + 1, figsize=(8 * (num_layers + 1), 16))

        titles = ['drivable_area',
                  'ped_crossing',
                  'walkway',
                  'stop_line',        # an area where the ego vehicle must stop
                  'carpark_area',
                  'lane_divider']

        # first row: original 512×512 images
        for i in range(num_layers):
            axes[i].imshow(masks[i], cmap='gray')
            axes[i].set_title(f'{titles[i]} (512×512)')
            axes[i].axis('off')
        axes[num_layers].imshow(rgb_img)
        axes[num_layers].set_title('Combined RGB (512×512)')
        axes[num_layers].axis('off')

        # # second row: downsampled 200×200 images
        # for i in range(4):
        #     axes[1, i].imshow(masks_200[i], cmap='gray')
        #     axes[1, i].set_title(f'{titles[i]} (200×200)')
        #     axes[1, i].axis('off')
        # axes[1, 4].imshow(rgb_img_200)
        # axes[1, 4].set_title('Combined RGB (200×200)')
        # axes[1, 4].axis('off')

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def show_prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_file()

    def show_next(self):
        if self.current_index < len(self.file_list) - 1:
            self.current_index += 1
            self.show_file()

    def save_image(self):
        if self.current_rgb_img is None:
            return

        npy_path = self.file_list[self.current_index]
        base_name = os.path.basename(npy_path).replace('.npy', '_rgb.png')
        save_path = os.path.join(os.getcwd(), base_name)

        img_uint8 = (self.current_rgb_img * 255).astype(np.uint8)
        from PIL import Image
        Image.fromarray(img_uint8).save(save_path)
        self.label.config(text=f"Saved to: {base_name}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MaskViewerApp(root)
    root.mainloop()
