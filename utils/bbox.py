def bbox_transform_inv(bbox):
  x, y, w, h = bbox
  out_box = [0, 0, 0, 0]
  out_box[0] = x
  out_box[1] = y
  out_box[2] = x + w
  out_box[3] = y + h
  return out_box


def cal_offset(gt_bbox, j_bbox):
  offset = [0, 0, 0, 0]
  offset[0] = (gt_bbox[0] - j_bbox[0]) / float(j_bbox[2] - j_bbox[0])
  offset[1] = (gt_bbox[1] - j_bbox[1]) / float(j_bbox[3] - j_bbox[1])
  offset[2] = (gt_bbox[2] - j_bbox[2]) / float(j_bbox[2] - j_bbox[0])
  offset[3] = (gt_bbox[3] - j_bbox[3]) / float(j_bbox[3] - j_bbox[1])
  return offset


def shift(box, offset):
  out_box = [0, 0, 0, 0]
  out_box[0] = offset[0] * float(box[2] - box[0]) + box[0]
  out_box[1] = offset[1] * float(box[3] - box[1]) + box[1]
  out_box[2] = offset[2] * float(box[2] - box[0]) + box[2]
  out_box[3] = offset[3] * float(box[3] - box[1]) + box[3]
  return out_box