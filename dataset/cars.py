# https://github.com/htdt/hyp_metric/blob/master/proxy_anchor/dataset/cars.py
from .base import *
import scipy.io

class Cars(BaseDataset):
    def __init__(self, root, mode, transform = None):
        self.root = root + '/cars196'
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            self.classes = range(0,98)
        elif self.mode == 'eval':
            self.classes = range(98,196)
                
        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        annos_fn = 'cars_annos.mat'
        cars = scipy.io.loadmat(os.path.join(self.root, annos_fn))
        ys = [int(a[5][0] - 1) for a in cars['annotations'][0]]
        im_paths = [a[0][0] for a in cars['annotations'][0]]
        index = 0
        for im_path, y in zip(im_paths, ys):
            if y in self.classes: # choose only specified classes
                self.im_paths.append(os.path.join(self.root, im_path))
                self.ys.append(y)
                self.I += [index]
                index += 1
