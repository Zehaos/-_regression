import caffe
import numpy as np
import cv2
from dataset.eyelevel5K import eyelevel5k
from utils import bbox_jitter, bbox_transform_inv, shift

network_proto_path = '/home/zehao/My_Net/Car_Regression/deploy.prototxt'
network_model_path = '/home/zehao/My_Net/Car_Regression/snapshot/train_iter_1021000.caffemodel'
imdb = eyelevel5k('/media/zehao/WD/Dataset/processed/car_dataset/Rendered/eyelevel5K/images',
                  '/media/zehao/WD/Dataset/processed/car_dataset/Rendered/eyelevel5K/annotations')

net = caffe.Net(network_proto_path, network_model_path, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_input_scale('data', 0.1)
transformer.set_mean('data', np.array([104, 117, 123]))
transformer.set_transpose('data', (2, 0, 1))


while True:
  # Generate test images
  img, bboxes = imdb.read()
  jittered_bboxes = bbox_jitter(bboxes[0], 0.1, 3)
  gt_bbox = bbox_transform_inv(bboxes[0])
  j_box = bbox_transform_inv(jittered_bboxes[0])
  test_img = img[j_box[1]:j_box[1] + j_box[3], j_box[0]:j_box[0] + j_box[2], :]
  test_img = caffe.io.resize(test_img, [48, 48, 3])

  offset = net.forward_all(data=np.asarray([transformer.preprocess('data', test_img)]))
  print offset['conv6'][0, :]
  print j_box
  out_box = shift(j_box, offset['conv6'][0, :])
  cv2.rectangle(img, (int(out_box[0]), int(out_box[1])),
                (int(out_box[2]), int(out_box[3])), (0, 255, 0), 1)
  cv2.rectangle(img, (int(j_box[0]), int(j_box[1])),
                (int(j_box[2]), int(j_box[3])), (255, 0, 0), 1)
  cv2.rectangle(img, (int(gt_bbox[0]), int(gt_bbox[1])),
                (int(gt_bbox[2]), int(gt_bbox[3])), (0, 0, 255), 1)
  cv2.imshow('output', img)
  cv2.waitKey(0)


