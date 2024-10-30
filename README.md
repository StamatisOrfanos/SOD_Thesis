# Small-object detection in remote sensing images and video - Master's Thesis


## Introduction
Object detection in remote sensing images has been a challenging problem for the computer vision research community due to the presence of small or tiny objects, which are often difficult to detect because they occupy only a small proportion of the image. These objects can be defined in two main ways: relatively, where an object is considered small if its bounding box covers less than 1% of the image area or absolutely, where small objects are defined by specific pixel dimensions, such as 32x32 pixels in the MS-COCO dataset or 16x16 pixels in the USC-GRAD-STDb. There have been improvements in the mean Average Precision (mAP) of the models using different architectures. Most of the detection models are becoming more complex and bigger, which can cause a problem usually when a detection model is intended for use in a satellite or an Unmanned Aerial Vehicle, since their computation resources are limited. This thesis proposes a new backbone being the Extended Feature Pyramid Network for the visual transformer Masked-Attention Mask Transformer as the detector. This new model utilizes feature maps, bounding boxes and masks as information to effectively localize and classify small objects. This approach has achieved a significant reduction in computational complexity, specifically a 56% decrease in Giga-Floating Point Operations Per Second (GFLOPs) in all cases. The datasets that were used for the evaluation of the models with the proposed method, were the Microsoft Common Object in COntext (MS COCO), VisDrone and Unmanned Aerial Vehicle Small Object Detection (UAV-SOD). On the UAV-SOD dataset the model had a 3.1% mAP improvement, while having an almost identical performance on the most complex dataset the MS COCO with a 6.5% decrease. Lastly on the VisDrone dataset we got a bigger performance decrease of around 13%, since the test data had objects that the model correctly localized and classified but the annotations were incorrectly not included. In this last case we think the performance of the model was better than the result may suggest. The results demonstrate the effectiveness of the proposed method, providing useful intel in multi-task learning and achieving greater accuracy performance and better computational efficiency on a set of challenging datasets.

### License

MIT License

Copyright (c) 2023 Stamatios Orfanos

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
