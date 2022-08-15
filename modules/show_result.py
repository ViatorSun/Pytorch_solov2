#  !/usr/bin/env  python
#  -*- coding:utf-8 -*-
# @Time     :  2022.08
# @Author   :  绿色羽毛
# @Email    :  lvseyumao@foxmail.com
# @Blog     :  https://blog.csdn.net/ViatorSun
# @arXiv    :   
# @version  :   
# @Note     :   
#
#


import cv2 as cv
import numpy as np
from scipy import ndimage
import pycocotools.mask as maskutil
from data.imgutils import imresize
from config import process_funcs_dict



COCO_LABEL_MAP = { 0:  1,  1:  2,  2:  3,  3:  4,  4:  5,  5:  6,  6:  7,  7:  8,
                   8:  9,  9: 10,  10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17,
                   16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25,
                   24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36,
                   32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44,
                   40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53,
                   48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61,
                   56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73,
                   64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81,
                   72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}

COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')


""" ------------------------------------------------------------------------------------------ """
Tomato_LABEL_MAP = {  0: 1 ,  1: 2 ,  2: 3 ,  3: 4 ,  4: 5 ,  5: 6 ,  6: 7,
                      7: 8 ,  8: 9 ,  9: 10, 10: 11, 11: 12, 12: 13,  13: 14,
                      14: 15, 15: 16, 16: 17, 17: 18, 18: 19, 19: 20, 20: 21,
                      21: 22, 22: 23, 23: 24, 24: 25, 25: 26, 26: 27, 27: 28}

Tomato_CLASSES = ("yuan_bad"      ,"yuan_good"    ,"yuan_best"    ,"kezhai_bad"   ,
                "kezhai_good"   ,"kezhai_best"  ,"zhedang_bad"  ,"zhedang_good" ,
                "zhedang_best"  , "yuan_bad"    ,"yuan_good"    ,"yuan_best"    ,
                "kezhai_bad"    ,"kezhai_good"  ,"kezhai_best"  ,"zhedang_bad"  ,
                "zhedang_good"  ,"zhedang_best" ,"yuan_bad"     ,"yuan_good"    ,
                "yuan_best"     , "kezhai_bad"  ,"kezhai_good"  ,"kezhai_best"  ,
                "kezhai_kailie" , "zhedang_bad" ,"zhedang_good" ,"zhedang_best")





def build_process_pipeline(pipeline_confg):
    assert isinstance(pipeline_confg, list)
    process_pipelines = []
    for pipconfig in pipeline_confg:
        assert isinstance(pipconfig, dict) and 'type' in pipconfig
        args = pipconfig.copy()
        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            process_pipelines.append(process_funcs_dict[obj_type](**args))

    return process_pipelines


def result2json(img_id, result):
    rel = []
    seg_pred   = result[0][0].cpu().numpy().astype(np.uint8)
    cate_label = result[0][1].cpu().numpy().astype(np.int)
    cate_score = result[0][2].cpu().numpy().astype(np.float)
    num_ins = seg_pred.shape[0]
    for j in range(num_ins):
        a = cate_label[j]
        # realclass = COCO_LABEL[cate_label[j]]
        realclass = Tomato_LABEL_MAP[cate_label[j]]
        re = {}
        score = cate_score[j]
        re["image_id"] = img_id
        re["category_id"] = int(realclass)
        re["score"] = float(score)
        outmask = np.squeeze(seg_pred[j])
        outmask = outmask.astype(np.uint8)
        outmask = np.asfortranarray(outmask)
        rle = maskutil.encode(outmask)
        rle['counts'] = rle['counts'].decode('ascii')
        re["segmentation"] = rle
        rel.append(re)
    return rel


class LoadImage(object):
    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = cv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


class LoadImageInfo(object):
    def __call__(self, frame):
        results = {}
        results['filename'] = None
        results['img'] = frame
        results['img_shape'] = frame.shape
        results['ori_shape'] = frame.shape
        return results


def show_result_ins(img, result, score_thr=0.2, sort_by_density=False):
    if isinstance(img, str):
        img  = cv.imread(img)
    img_show = img.copy()
    h, w, _  = img.shape

    cur_result = result[0]
    seg_label  = cur_result[0]
    seg_label  = seg_label.cpu().numpy().astype(np.uint8)

    cate_label = cur_result[1]
    cate_label = cate_label.cpu().numpy()
    # print("cate_label is:", cate_label)
    # print("cate_label max is:", cate_label.max())

    score = cur_result[2].cpu().numpy()

    # vis_inds   = score > score_thr
    # seg_label  = seg_label[vis_inds]
    # num_mask   = seg_label.shape[0]
    # cate_label = cate_label[vis_inds]
    # cate_score = score[vis_inds]


    num_mask = seg_label.shape[0]
    cate_score = score

    if sort_by_density:
        mask_density = []
        for idx in range(num_mask):
            cur_mask = seg_label[idx, :, :]
            cur_mask = imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.int32)
            mask_density.append(cur_mask.sum())
        orders = np.argsort(mask_density)
        seg_label  = seg_label[orders]
        cate_label = cate_label[orders]
        cate_score = cate_score[orders]

    np.random.seed(42)
    color_masks = [np.random.randint(0, 256, (1, 3), dtype=np.uint8) for _ in range(num_mask)]

    for idx in range(num_mask):         # 倒序 mask可视化，将概率高的置顶
        idx = -(idx + 1)
        cur_mask = seg_label[idx, :, :]
        cur_mask = imresize(cur_mask, (w, h))
        cur_mask = (cur_mask > 0.5).astype(np.uint8)
        if cur_mask.sum() == 0:
            continue
        color_mask = color_masks[idx]
        cur_mask_bool = cur_mask.astype(np.bool)
        img_show[cur_mask_bool] = img[cur_mask_bool] * 0.5 + color_mask * 0.5

        # 当前实例的类别
        cur_score = cate_score[idx]
        name_idx  = cate_label[idx]
        # print("cur_cate  is:", cur_cate)
        # realclass = COCO_LABEL[cur_cate]
        # print("realclass  is:", realclass)


        # name_idx = COCO_LABEL_MAP[realclass]
        # print("name_idx is:", name_idx)
        label_text = Tomato_CLASSES[name_idx]
        label_text += '|{:.02f}'.format(cur_score)
        center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
        vis_pos = (max(int(center_x) - 10, 0), int(center_y))
        cv.putText(img_show, label_text, vis_pos, cv.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255))  # green
        print("label is:",label_text)
    return img_show