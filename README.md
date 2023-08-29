Object detection with dynamic upsamplers, based on [mmdetection](https://github.com/open-mmlab/mmdetection)

For example, to train Faster R-CNN-R50 with [DySample](https://github.com/tiny-smart/dysample):

```shell
bash dist_train.sh configs/dynamic_upsampling/faster-rcnn_r50_fpn_dysample-lpg4_1x_coco.py 4
```
