import os.path
import cv2

from dataset.eyelevel5K import eyelevel5k
from utils import bbox_transform_inv, bbox_jitter, cal_offset

LABEL_OUT_PATH = '/media/zehao/Local2/car_regression_data/labels'
IMAGE_OUT_PATH = '/media/zehao/Local2/car_regression_data/images'

if __name__ == '__main__':
  imdb = eyelevel5k('/media/zehao/WD/Dataset/processed/car_dataset/Rendered/eyelevel5K/images',
             '/media/zehao/WD/Dataset/processed/car_dataset/Rendered/eyelevel5K/annotations')
  for i in range(imdb.num_images):
    img, bboxes = imdb.read()
    jittered_bboxes = bbox_jitter(img, bboxes[0], 0.2, 15)
    gt_bbox = bbox_transform_inv(bboxes[0])
    for j_bbox, j in zip(jittered_bboxes, range(len(jittered_bboxes))):
      j_bbox = bbox_transform_inv(j_bbox)
      j_img = img[j_bbox[1]:j_bbox[3], j_bbox[0]:j_bbox[2], :]
      print os.path.join(IMAGE_OUT_PATH, str(i)+'_'+str(j))+'.jpg'
      cv2.imwrite(os.path.join(IMAGE_OUT_PATH, str(i)+'_'+str(j))+'.jpg', j_img)
      offset = cal_offset(gt_bbox, j_bbox)
      with open(os.path.join(LABEL_OUT_PATH, str(i)+'_'+str(j)+'.txt'), 'w') as f:
        f.write(str(offset[0])+' '+str(offset[1])+' '+str(offset[2])+' '+str(offset[3]))