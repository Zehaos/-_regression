"""Image data base class for eyelevel5k"""

import os

from imdb import imdb

import cv2

class eyelevel5k(imdb):
  def __init__(self, imgs_path, annos_path):
    imdb.__init__(self, 'eyelevel5k')
    self.imgs_path = imgs_path
    self.annos_path = annos_path
    self._anno = []
    self.prepare_list()
    self._num_images = len(self._img_anno_list)

  @property
  def num_images(self):
    return self._num_images

  def prepare_list(self):
    img_list = os.listdir(self.imgs_path)
    for img in img_list:
      assert os.path.splitext(img)[1] in ['.jpg', '.png', '.bmp']
      name = os.path.basename(img)
      img_path = os.path.join(self.imgs_path, img)
      anno_path = os.path.join(self.annos_path, str.split(name, '.')[0] + '.txt')
      self._img_anno_list.append([img_path, anno_path])

  def read(self):
    self.read_image()
    self.read_anno()
    self._cur_idx += 1
    return self._img, self._anno

  def read_anno(self):
    self._anno = []
    with open(self._img_anno_list[self.cur_idx][1]) as f:
      lines = f.readlines()
      line = str.split(lines[1], ' ')
      bbox = [int(line[1]), int(line[2]), int(line[3]), int(line[4])]
      self._anno.append(bbox)

  def read_image(self):
    self._img = cv2.imread(self._img_anno_list[self._cur_idx][0])
