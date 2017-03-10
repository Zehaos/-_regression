import numpy as np

def bbox_jitter(bbox, j_range, num):
  max_drift_x = bbox[2]*j_range
  max_drift_y = bbox[3]*j_range
  max_drift_w = bbox[2] * j_range
  max_drift_h = bbox[3] * j_range
  j_bboxes = []
  for i in range(num):
    dy = np.random.randint(-max_drift_y, max_drift_y)
    dx = np.random.randint(-max_drift_x, max_drift_x)
    dw = np.random.randint(-max_drift_w, max_drift_w)
    dh = np.random.randint(-max_drift_h, max_drift_h)
    box = [0,0,0,0]
    box[0] = bbox[0] - dx
    box[1] = bbox[1] - dy
    box[2] = bbox[2] - dw
    box[3] = bbox[3] - dh
    j_bboxes.append(box)
  return j_bboxes