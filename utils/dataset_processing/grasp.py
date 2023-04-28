import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import disk
from skimage.feature import peak_local_max
from skimage.transform import rotate, resize


def _gr_text_to_no(l, offset=(0, 0)):

    x, y = l.split()
    return [int((int(round(float(y))) - offset[0]) / 2), int((int(round(float(x))) - offset[1]) / 2)]


class GraspRectangles:
    def __init__(self, grs=None):
        if grs:
            self.grs = grs
        else:
            self.grs = []

    def __getitem__(self, item):
        return self.grs[item]

    def __iter__(self):
        return self.grs.__iter__()

    def __getattr__(self, attr):

        if hasattr(GraspRectangle, attr) and callable(getattr(GraspRectangle, attr)):
            return lambda *args, **kwargs: list(map(lambda gr: getattr(gr, attr)(*args, **kwargs), self.grs))
        else:
            raise AttributeError("Couldn't find function %s in BoundingBoxes or BoundingBox" % attr)

    @classmethod
    def load_from_array(cls, arr):
        grs = []
        for i in range(arr.shape[0]):
            grp = arr[i, :, :].squeeze()
            if grp.max() == 0:
                break
            else:
                grs.append(GraspRectangle(grp))
        return cls(grs)

    @classmethod
    def load_from_cornell_file(cls, mask_name, center_name):
        grs = []
        grs.append(GraspRectangle(np.load(mask_name), np.load(center_name)))
        return cls(grs)

    @classmethod
    def load_from_jacquard_file(cls, fname, scale=1.0):
        grs = []
        with open(fname) as f:
            for l in f:
                x, y, theta, w, h = [float(v) for v in l[:-1].split(';')]
                # index based on row, column (y,x), and the Jacquard dataset's angles are flipped around an axis.
                grs.append(Grasp(np.array([y, x]), -theta / 180.0 * np.pi, w, h).as_gr)
        grs = cls(grs)
        grs.scale(scale)
        return grs

    def append(self, gr):
        self.grs.append(gr)

    def copy(self):
        new_grs = GraspRectangles()
        for gr in self.grs:
            new_grs.append(gr.copy())
        return new_grs

    def show(self, ax=None, shape=None):

        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1, 1, 1)
            ax.imshow(np.zeros(shape))
            ax.axis([0, shape[1], shape[0], 0])
            self.plot(ax)
            plt.show()
        else:
            self.plot(ax)

    def draw(self, shape, position=True, radius=True):
        mask_out = self.grs[0].mask
        pos_out = mask_out / mask_out.max()
        return pos_out, mask_out

    def to_array(self, pad_to=0):
        a = np.stack([gr.points for gr in self.grs])
        if pad_to:
            if pad_to > len(self.grs):
                a = np.concatenate((a, np.zeros((pad_to - len(self.grs), 4, 2))))
        return a.astype(np.int)

    @property
    def center(self):
        return self.grs[0].center.astype(np.int)


class GraspRectangle:
    def __init__(self, mask, center):
        self.mask = mask
        self.center = center

    def __str__(self):
        return str(self.points)

    @property
    def as_grasp(self):
        return Grasp(self.center, self.angle, self.length, self.width)


    def polygon_coords(self, shape=None):

        return disk((self.center[0], self.center[1]), self.mask / 2, shape=(224, 224))

    def compact_polygon_coords(self, shape=None):
        return Grasp(self.center, self.radium).as_gr.polygon_coords(shape)

    def iou(self, gr):
        rr1, cc1 = self.polygon_coords()
        rr2, cc2 = np.where(gr.mask > 1)
        canvas = np.zeros((224, 224))
        canvas[rr1, cc1] += 1
        canvas[rr2, cc2] += 1
        union = np.sum(canvas > 0)
        if union == 0:
            return 0
        intersection = np.sum(canvas == 2)
        return intersection / union

    def copy(self):
        return GraspRectangle(self.points.copy())

    def offset(self, offset):
        self.center += np.array([offset[1], offset[0]])

    def rotate(self, angle, center):
        R = np.array(
            [
                [np.cos(-angle), np.sin(-angle)],
                [-1 * np.sin(-angle), np.cos(-angle)],
            ]
        )
        c = np.array(center).reshape((1, 2))
        center = (center[1], center[0])
        self.mask = rotate(self.mask, angle / np.pi * 180, center=center, preserve_range=True).astype(
            self.mask.dtype)

    def crop(self, top_left, bottom_right):
        self.mask = self.mask[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]

    def scale(self, factor):
        if factor == 1.0:
            return
        self.points *= factor

    def plot(self, ax):
        draw_circle = plt.Circle((self.center[1], self.center[0]), self.mask / 2, fill=False)
        draw_scater = plt.scatter(self.center[1], self.center[0], s=6, c=2, alpha=0.5)
        ax.add_artist(draw_circle)
        ax.add_artist(draw_scater)

    def zoom(self, factor, cen):
        sr = int(self.mask.shape[0] * (1 - factor)) // 2
        sc = int(self.mask.shape[1] * (1 - factor)) // 2
        orig_shape = self.mask.shape
        self.mask = self.mask[sr:self.mask.shape[0] - sr, sc: self.mask.shape[1] - sc].copy()
        self.mask = resize(self.mask, orig_shape, mode='symmetric', preserve_range=True).astype(self.mask.dtype)
        T = np.array(
            [
                [1 / factor, 0],
                [0, 1 / factor]
            ]
        )
        c = np.array(cen).reshape((1, 2))
        self.center = ((np.dot(T, (self.center - c).T)).T + c).astype(np.int)


class Grasp:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    @property
    def as_gr(self):
        return GraspRectangle(self.radius, self.center)

    def max_iou(self, grs):
        self_gr = self.as_gr
        max_iou = 0
        for gr in grs:
            iou = self_gr.iou(gr)
            max_iou = max(max_iou, iou)
        return max_iou

    def plot(self, ax, color=None):
        self.as_gr.plot(ax, color)

def detect_grasps(q_img, width_img=None):
    local_max = peak_local_max(q_img, min_distance=20, threshold_abs=0.2, num_peaks=1)  # 区域的最大值滤波
    grasps = []
    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)

        g = Grasp(grasp_point, width_img[grasp_point])
        grasps.append(g)

    return grasps
