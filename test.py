import mmcv
import os
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
import numpy as np
from glob import glob
cfg = mmcv.Config.fromfile('configs/myfaster_rcnn_r50_fpn_1x.py')
cfg.model.pretrained = None

# 构建网络，载入模型
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)

#_ = load_checkpoint(model,'./model_work/save_model/epoch1_16_21.pth')# ##'faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth')
_ = load_checkpoint(model,'./model_work/save_model/ac98.4-res50-lr0.008.pth')#'./model_work/epoch_5.pth')#
# #测试一张图片
# img = mmcv.imread('imgTest/rebar1.jpg')
# result = inference_detector(model, img, cfg)
# show_result(img, result,dataset='mydata2',score_thr=0.6)#,outfile='./i mgTest/')
# 测试多张图片
threshold=0.5
results_dir= "./submit/result_%s.csv"%str(threshold)
if not os.path.exists(results_dir):
    os.system(r"touch {}".format(results_dir))
#mmcv.mkdir_or_exist(results_dir)
results_file= open("./submit/result_%s.csv"%str(threshold),"w")

def write_csv(filename,result):
    bboxes = np.vstack(result)
    scores = bboxes[:, -1]
    inds = scores > threshold
    bboxes = bboxes[inds, :]
    for bbox in bboxes:
        bbox_int = bbox.astype(np.int32)
    # left_top = (bbox_int[0], bbox_int[1])
    # right_bottom = (bbox_int[2], bbox_int[3])
        results_file.write(filename.split("/")[-1] + "," + str(int(bbox_int[0])) + " " + str(int(bbox_int[1])) + " "
                       + str(int(bbox_int[2])) + " " + str(int(bbox_int[3])) + "\n")

root='../data/test/'
images_list = glob(root+"*.jpg")
for i, result in enumerate(inference_detector(model, images_list, cfg, device='cuda:0')):
    print(i, images_list[i])
    write_csv(images_list[i], result)
    show_result(images_list[i], result,dataset='mydata2',score_thr=0.5 ,thick=3,imgshow=False,
                     outfile=os.path.join('./imgTest/outnew/', 'out{}.png'.format(i)))

# imgs=[]
# for i in range(1,16):#[ , )前闭后开
#     imgs.append('imgTest/rebar{}.jpg'.format(i))
# #imgs = ['imgTest/rebar3.jpg', 'imgTest/rebar4.jpg', 'imgTest/rebar5.jpg']
# for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
#     print(i, imgs[i])
#     show_result(imgs[i], result,dataset='mydata2',score_thr=0.5 ,thick=3,
#                 outfile=os.path.join('./imgTest/outnew/', 'out{}.png'.format(i)))

