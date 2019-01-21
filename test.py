import mmcv
import os
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result

cfg = mmcv.Config.fromfile('configs/myfaster_rcnn_r101_fpn_1x.py')
cfg.model.pretrained = None

# 构建网络，载入模型
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)

#_ = load_checkpoint(model, 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth')
# 如果通过网盘下载，取消下一行代码的注释，并且注释掉上一行
#_ = load_checkpoint(model,'./model_work/save_model/epoch1_16_21.pth')#'./model_work/epoch_10.pth') ##'faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth')
_ = load_checkpoint(model,'./model_work/save_model/ac98.2-res101-lr0.008.pth')
# #测试一张图片
# img = mmcv.imread('imgTest/rebar1.jpg')
# result = inference_detector(model, img, cfg)
# show_result(img, result,dataset='mydata2',score_thr=0.6)#,outfile='./imgTest/')

#/rebarCount_mmdetection/mmdet/core/evaluation/class_names.py
# 测试多张图片
imgs=[]
for i in range(1,13):#[ , )前闭后开
    imgs.append('imgTest/rebar{}.jpg'.format(i))
#imgs = ['imgTest/rebar3.jpg', 'imgTest/rebar4.jpg', 'imgTest/rebar5.jpg']
for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
    print(i, imgs[i])
    show_result(imgs[i], result,dataset='mydata2',score_thr=0.5 ,thick=3,
                outfile=os.path.join('./imgTest/', 'out{}.png'.format(i)))