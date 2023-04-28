import random
import numpy as np
import torch
import torch.utils.data
import numpy.matlib


class GraspDatasetBase(torch.utils.data.Dataset):
    def __init__(self, output_size=224, random_rotate=False, random_zoom=False, input_only=False):
        self.output_size = output_size
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.input_only = input_only
        self.grasp_files = []

    def gauss_transformation(self, pos_img, width_img, center):
        center_x, center_y = center[0][0], center[0][1]
        IMAGE_HEIGHT = pos_img.shape[0]
        IMAGE_WIDTH = pos_img.shape[1]
        R = 0.5 * width_img.max()
        mask_x = np.matlib.repmat(center_x, IMAGE_HEIGHT, IMAGE_WIDTH)
        mask_y = np.matlib.repmat(center_y, IMAGE_HEIGHT, IMAGE_WIDTH)

        x1 = np.arange(IMAGE_WIDTH)
        x_map = np.matlib.repmat(x1, IMAGE_HEIGHT, 1)

        y1 = np.arange(IMAGE_HEIGHT)
        y_map = np.matlib.repmat(y1, IMAGE_WIDTH, 1)
        y_map = np.transpose(y_map)

        Gauss_map = np.sqrt((x_map - mask_x) ** 2 + (y_map - mask_y) ** 2)

        Gauss_map = np.exp(-0.5 * Gauss_map / R)
        guess_img = pos_img * Gauss_map
        return guess_img

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def get_rgb(self, idx, rot=0, zoom=1.0):
        raise NotImplementedError()

    def __getitem__(self, idx):
        if self.random_rotate:
            rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
            rot = random.choice(rotations)
        else:
            rot = 0.0
        if self.random_zoom:
            zoom_factor = np.random.uniform(0.6, 1.0)
        else:
            zoom_factor = 1.0

        rgb_img = self.get_rgb(idx, rot, zoom_factor)
        bbs = self.get_gtbb(idx, rot, zoom_factor)
        pos_img, width_img = bbs.draw((self.output_size, self.output_size))
        width_img = width_img / zoom_factor
        pos_img = self.gauss_transformation(pos_img, width_img, bbs.center)
        radius_img = width_img / (self.output_size / 2)

        x = self.numpy_to_torch(rgb_img)

        pos = self.numpy_to_torch(pos_img)
        radius = self.numpy_to_torch(radius_img)

        return x, (pos, radius), idx, rot, zoom_factor

    def __len__(self):
        return len(self.grasp_files)
