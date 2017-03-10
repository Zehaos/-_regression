"""The data base wrapper class"""

import cv2

class imdb(object):
    """Image database."""

    def __init__(self, name):
        self._name = name
        self._classes = []
        self._img_anno_list = []
        self._cur_idx = 0

    @property
    def name(self):
        return self._name

    @property
    def classes(self):
        return self._classes

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def img_anno_list(self):
        return self._img_anno_list

    @property
    def cur_idx(self):
        return self._cur_idx

    def read(self):
        NotImplemented

    def prepare_list(self):
        NotImplemented

    def read_image(self):
        NotImplemented

    def read_annotations(self):
        NotImplemented

    def visualize(self):
        for bbox in self._anno:
            cv2.rectangle(self._img, (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                          (0, 255, 0), 1)
        cv2.imshow('visulization', self._img)
        cv2.waitKey(0)

