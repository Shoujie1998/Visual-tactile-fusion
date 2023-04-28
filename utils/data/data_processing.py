import glob
import os
import cv2
from utils.dataset_processing import grasp, image
from .grasp_data import GraspDatasetBase
import numpy as np


class TransDataset(GraspDatasetBase):
    def __init__(self, file_path, mode, ds_rotate=0, **kwargs):
        super(TransDataset, self).__init__(**kwargs)

        self.grasp_files = glob.glob(os.path.join(file_path, mode, r'anno_mask', '*.npy'))
        print('location:', self.grasp_files)
        self.grasp_files.sort()
        self.length = len(self.grasp_files)


        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]

        self.rgb_files = [f.replace('.npy', '.png').replace('anno_mask', 'rgb') for f in self.grasp_files]
        self.center = [f.replace('anno_mask', 'center') for f in self.grasp_files]

    def _get_crop_attrs(self, idx):

        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx], self.center[idx])  # 从标签获取抓取的矩形框
        center = gtbbs.center
        center = np.array([center[1], center[0]])
        left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx], self.center[idx])
        center, left, top = self._get_crop_attrs(idx)
        gtbbs.rotate(rot, center)
        gtbbs.crop((top, left),
                   (min(480, top + self.output_size), min(640, left + self.output_size)))
        gtbbs.offset((-top, -left))
        gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left),
                       (min(480, top + self.output_size), min(640, left + self.output_size)))  # 进行裁剪操作，剪成224的大小
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])

        center, left, top = self._get_crop_attrs(idx)
        rgb_img.rotate(rot, center)

        rgb_img.crop((top, left),
                     (min(480, top + self.output_size), min(640, left + self.output_size)))

        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img
