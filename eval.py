#  !/usr/bin/env  python
#  -*- coding:utf-8 -*-
# @Time     :  2022.07
# @Author   :  绿色羽毛
# @Email    :  lvseyumao@foxmail.com
# @Blog     :  https://blog.csdn.net/ViatorSun
# @arXiv    :
# @version  :
# @Note     :
#
#




import os
import json
import time
import torch
import numpy as np
import cv2 as cv
from data.compose import Compose
from glob import glob
import pycocotools.mask as maskutil

from modules.solov2 import SOLOV2
from modules.solo import SOLO

from modules.show_result import LoadImage, LoadImageInfo
from modules.show_result import show_result_ins, build_process_pipeline, result2json
from config import cfg, process_funcs_dict





def eval(valmodel_weight, data_path, benchmark, test_mode, save_imgs=False):
    test_pipeline = []
    transforms = [dict(type='Resize', keep_ratio=True),
                  dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                  dict(type='Pad', size_divisor=32),
                  dict(type='ImageToTensor', keys=['img']),
                  dict(type='TestCollect', keys=['img'])    ]

    transforms_piplines = build_process_pipeline(transforms)
    Multest = process_funcs_dict['MultiScaleFlipAug'](transforms=transforms_piplines, img_scale=(512, 512), flip=False)

    if test_mode == "video":
        test_pipeline.append(LoadImageInfo())
    elif test_mode == "images":
        test_pipeline.append(LoadImage())
    else:
        raise NotImplementedError("not support mode!")
    test_pipeline.append(Multest)
    test_pipeline = Compose(test_pipeline)

    # model = SOLOV2(cfg, pretrained=valmodel_weight, mode='test')

    model = SOLO(pretrained=valmodel_weight, test_cfg=cfg.test_cfg ,mode='test')

    # model.load_state_dict(torch.load(valmodel_weight))
    # model = torch.load(valmodel_weight)

    model = model.cuda()
    model.eval()

    # if test_mode == "video":
    #     vid = cv.VideoCapture(data_path)
    #     target_fps   = round(vid.get(cv.CAP_PROP_FPS))
    #     frame_width  = round(vid.get(cv.CAP_PROP_FRAME_WIDTH))
    #     frame_height = round(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
    #     num_frames   = round(vid.get(cv.CAP_PROP_FRAME_COUNT))
    #
    #     for i in range(num_frames):
    #         if i % 5 != 0:
    #             continue
    #         frame = vid.read()
    #         img = frame[1]
    #         data = test_pipeline(img)
    #         imgs = data['img']
    #
    #         img = imgs[0].cuda().unsqueeze(0)
    #         img_info = data['img_metas']
    #         start = time.time()
    #         with torch.no_grad():
    #             seg_result = model.forward(img=[img], img_meta=[img_info], return_loss=False)
    #
    #         img_show = show_result_ins(frame[1], seg_result)
    #         end = time.time()
    #         print("spend time: ", (end - start))
    #         cv.imshow("watch windows", img_show)
    #         cv.waitKey(1)

    if test_mode == "images":
        img_ids  = []
        images   = []
        use_json = "E:/Data/TomatoInstance1/annotations.json"
        # test_imgpath = data_path
        # if use_json is False:
        #     test_imgpath = test_imgpath + '/*'
        #     images = glob(test_imgpath)
        #     for img in images:
        #         pathname, filename = os.path.split(img)
        #         prefix, suffix = os.path.splitext(filename)
        #         img_id = int(prefix)
        #         img_ids.append(str(img_id))
        # else:
        imgsinfo = json.load(open(use_json, 'r'))
        for i in range(len(imgsinfo['images'])):
            img_id = imgsinfo['images'][i]['id']
            img_path = "E:/Data/TomatoInstance1/" + imgsinfo['images'][i]['file_name']

            img_ids.append(img_id)
            images.append(img_path)

        imgs_nums = len(images)
        results = []
        k = 0
        for imgpath in images:
            img_id = img_ids[k]
            data = dict(img=imgpath)
            data = test_pipeline(data)
            imgs = data['img']

            img = imgs[0].cuda().unsqueeze(0)
            img_info = data['img_metas']

            start = time.time()
            with torch.no_grad():
                seg_result = model.forward(img=[img], img_meta=[img_info], return_loss=False)
            img_show = show_result_ins(imgpath, seg_result)
            end = time.time()
            t_all = end - start
            # print("spend time: ", t_all ,"s")
            print('average time:{:.02f} s'.format(np.mean(t_all) / 1))
            print('average FPS :{:.02f} fps'.format(1 / np.mean(t_all)))

            out_filepath = "results1/" + os.path.basename(imgpath)

            k = k + 1
            if save_imgs:
                cv.imwrite(out_filepath, img_show)
                print("success save img:", out_filepath, '\n')
            if benchmark is True:
                result = result2json(img_id, seg_result)
                results = results + result

        if benchmark is True:
            re_js = json.dumps(results)
            fjson = open("eval_masks.json", "w")
            print("\n>>> success save eval_masks.json")
            fjson.write(re_js)
            fjson.close()


if __name__ == '__main__':

    eval(valmodel_weight='weights/Aug12-21-49-59_solov2_epoch_24.pth',
         data_path="E:/Data/TomatoInstance1/",
         benchmark=True, test_mode="images", save_imgs=True)

# eval(valmodel_weight='pretrained/solov2_448_r18_epoch_36.pth',data_path="cam0.avi", benchmark=False, test_mode="video")

