# https://github.com/htdt/hyp_metric/blob/master/proxy_anchor/dataset/SOP.py

from .base import *


class SOP(BaseDataset):
    def __init__(self, root, mode, transform = None):
        self.root = root + '/Stanford_Online_Products'
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            self.classes = range(0,11318)
        elif self.mode == 'eval':
            self.classes = range(11318,22634)  

        BaseDataset.__init__(self, self.root, self.mode, self.transform)
        metadata = open(os.path.join(self.root, 'Ebay_train.txt' if self.classes == range(0, 11318) else 'Ebay_test.txt'))
        for i, (image_id, class_id, _, path) in enumerate(map(str.split, metadata)):
            if i > 0:
                if int(class_id)-1 in self.classes:
                    self.ys += [int(class_id)-1]
                    self.I += [int(image_id)-1]
                    self.im_paths.append(os.path.join(self.root, path))
