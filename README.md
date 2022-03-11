# paper整理-deep learning object detection
---
> 處理時間:2022/03/10
> forked from [xialeiliu/Awesome-Incremental-Learning](https://github.com/hoya012/deep_learning_object_detection)

---
```
報告順序:
Object Detection in 20 Years: A Survey

(R-CNN家族)
Fast R-CNN  
Faster R-CNN, RPN
Light-Head R-CNN (改良不太多)
Cascade R-CNN
Libra R-CNN
Dynamic R-CNN
Sparse R-CNN

SSD

OHEM、Soft NMS
R-FCN
FPN 、NAS-FPN

ResNet  (影像分類)
VGG     (影像分類)
CSPNet  (影像分類)

GIoU 、Distance-IoU Loss
RetinaNet
RefineDet
TridentNet

FCOS
M2Det
EGNet
EfficientDet

(YOLO家族)
YOLOv3
YOLOv4
Scaled-YOLOv4
PP-YOLO
YOLOR

[最後加入]
CornerNet (anchor free)
CenterNet (anchor free)
```
---
## 2014

- <font color="#f00">**重要**</font> **[R-CNN]** Rich feature hierarchies for accurate object detection and semantic segmentation | **[CVPR' 14]** |[`[pdf]`](https://arxiv.org/pdf/1311.2524.pdf) [`[official code - caffe]`](https://github.com/rbgirshick/rcnn) 

- <font color="#f00">**重要**</font> **[OverFeat]** OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks | **[ICLR' 14]** |[`[pdf]`](https://arxiv.org/pdf/1312.6229.pdf) [`[official code - torch]`](https://github.com/sermanet/OverFeat) 

- <font color="#f00">**重要**</font> **[SPPnet]** Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition |  |[`[pdf]`](https://arxiv.org/abs/1406.4729)

- <font color="#f00">**重要-影像分類**</font> **[ResNet]** Deep Residual Learning for Image Recognition [`[pdf]`](https://arxiv.org/abs/1512.03385)

- <font color="#f00">**重要-影像分類**</font> **[VGG]** Very Deep Convolutional Networks for Large-Scale Image Recognition [`[pdf]`](https://arxiv.org/abs/1409.1556)

---
## 2015

- <font color="#f00">**重要**</font> **[Fast R-CNN]** Fast R-CNN | **[ICCV' 15]** |[`[pdf]`](https://arxiv.org/pdf/1504.08083.pdf) [`[official code - caffe]`](https://github.com/rbgirshick/fast-rcnn) 

- <font color="#f00">**重要**</font> **[Faster R-CNN, RPN]** Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks | **[NIPS' 15]** |[`[pdf]`](https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf)  [`[official code - caffe]`](https://github.com/rbgirshick/py-faster-rcnn) [`[unofficial code - tensorflow]`](https://github.com/endernewton/tf-faster-rcnn)  [`[unofficial code - pytorch]`](https://github.com/jwyang/faster-rcnn.pytorch) 

---
## 2016

- <font color="#f00">**重要**</font> **[YOLO v1]** You Only Look Once: Unified, Real-Time Object Detection | **[CVPR' 16]** |[`[pdf]`](https://arxiv.org/pdf/1506.02640.pdf) [`[official code - c]`](https://pjreddie.com/darknet/yolo/) 


- <font color="#f00">**重要**</font> **[OHEM]** Training Region-based Object Detectors with Online Hard Example Mining | **[CVPR' 16]** |[`[pdf]`](https://arxiv.org/pdf/1604.03540.pdf) [`[official code - caffe]`](https://github.com/abhi2610/ohem)  

- <font color="#f00">**重要**</font> **[SSD]** SSD: Single Shot MultiBox Detector | **[ECCV' 16]** |[`[pdf]`](https://arxiv.org/pdf/1512.02325.pdf) [`[official code - caffe]`](https://github.com/weiliu89/caffe/tree/ssd) [`[unofficial code - tensorflow]`](https://github.com/balancap/SSD-Tensorflow) [`[unofficial code - pytorch]`](https://github.com/amdegroot/ssd.pytorch) 


- <font color="#f00">**重要**</font> **[R-FCN]** R-FCN: Object Detection via Region-based Fully Convolutional Networks | **[NIPS' 16]** |[`[pdf]`](https://arxiv.org/pdf/1605.06409.pdf) [`[official code - caffe]`](https://github.com/daijifeng001/R-FCN) [`[unofficial code - caffe]`](https://github.com/YuwenXiong/py-R-FCN)

---
## 2017

- <font color="#f00">**重要**</font> **[FPN]** Feature Pyramid Networks for Object Detection  | **[CVPR' 17]** |[`[pdf]`](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf) [`[unofficial code - caffe]`](https://github.com/unsky/FPN)

- <font color="#f00">**重要**</font> **[YOLO v2]** YOLO9000: Better, Faster, Stronger | **[CVPR' 17]** |[`[pdf]`](https://arxiv.org/pdf/1612.08242.pdf) [`[official code - c]`](https://pjreddie.com/darknet/yolo/) [`[unofficial code - caffe]`](https://github.com/quhezheng/caffe_yolo_v2) [`[unofficial code - tensorflow]`](https://github.com/nilboy/tensorflow-yolo) [`[unofficial code - tensorflow]`](https://github.com/sualab/object-detection-yolov2) [`[unofficial code - pytorch]`](https://github.com/longcw/yolo2-pytorch) 

- <font color="#f00">**重要**</font> **[RetinaNet]** Focal Loss for Dense Object Detection | **[ICCV' 17]** |[`[pdf]`](https://arxiv.org/pdf/1708.02002.pdf) [`[official code - keras]`](https://github.com/fizyr/keras-retinanet) [`[unofficial code - pytorch]`](https://github.com/kuangliu/pytorch-retinanet) [`[unofficial code - mxnet]`](https://github.com/unsky/RetinaNet) [`[unofficial code - tensorflow]`](https://github.com/tensorflow/tpu/tree/master/models/official/retinanet)

- **[SMN]** Spatial Memory for Context Reasoning in Object Detection | **[ICCV' 17]** |[`[pdf]`](http://openaccess.thecvf.com/content_ICCV_2017/papers/Chen_Spatial_Memory_for_ICCV_2017_paper.pdf) 、引用次數133

- **[Light-Head R-CNN]** Light-Head R-CNN: In Defense of Two-Stage Object Detector | **[arXiv' 17]** |[`[pdf]`](https://arxiv.org/pdf/1711.07264.pdf) [`[official code - tensorflow]`](https://github.com/zengarden/light_head_rcnn) 、引用次數261

- <font color="#f00">**重要**</font> **[Soft-NMS]** Improving Object Detection With One Line of Code | **[ICCV' 17]** |[`[pdf]`](https://arxiv.org/pdf/1704.04503.pdf) [`[official code - caffe]`](https://github.com/bharatsingh430/soft-nms)、引用次數1055

## 2018

- <font color="#f00">**重要**</font> **[YOLO v3]** YOLOv3: An Incremental Improvement | **[arXiv' 18]** |[`[pdf]`](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [`[official code - c]`](https://pjreddie.com/darknet/yolo/) [`[unofficial code - pytorch]`](https://github.com/ayooshkathuria/pytorch-yolo-v3) [`[unofficial code - pytorch]`](https://github.com/eriklindernoren/PyTorch-YOLOv3) [`[unofficial code - keras]`](https://github.com/qqwweee/keras-yolo3) [`[unofficial code - tensorflow]`](https://github.com/mystic123/tensorflow-yolo-v3)


- <font color="#f00">**重要**</font> **[RefineDet]** Single-Shot Refinement Neural Network for Object Detection | **[CVPR' 18]** |[`[pdf]`](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single-Shot_Refinement_Neural_CVPR_2018_paper.pdf) [`[official code - caffe]`](https://github.com/sfzhang15/RefineDet) [`[unofficial code - chainer]`](https://github.com/fukatani/RefineDet_chainer)  [`[unofficial code - pytorch]`](https://github.com/lzx1413/PytorchSSD)、引用次數1055


- <font color="#f00">**重要**</font> **[Cascade R-CNN]** Cascade R-CNN: Delving into High Quality Object Detection | **[CVPR' 18]** |[`[pdf]`](http://openaccess.thecvf.com/content_cvpr_2018/papers/Cai_Cascade_R-CNN_Delving_CVPR_2018_paper.pdf) [`[official code - caffe]`](https://github.com/zhaoweicai/cascade-rcnn) 、引用次數2129


- <font color="#f00">**重要**</font> **[CornerNet]** CornerNet: Detecting Objects as Paired Keypoints | **[ECCV' 18]** |[`[pdf]`](https://arxiv.org/pdf/1808.01244.pdf) [`[official code - pytorch]`](https://github.com/princeton-vl/CornerNet)、引用次數1806

- **[Softer-NMS]** Softer-NMS: Rethinking Bounding Box Regression for Accurate Object Detection | **[arXiv' 18]** |[`[pdf]`](https://arxiv.org/pdf/1809.08545.pdf) 、引用次數259

---
## 2019

- <font color="#f00">**重要**</font> **[M2Det]** M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network | **[AAAI' 19]** |[`[pdf]`](https://arxiv.org/pdf/1811.04533.pdf) [`[official code - pytorch]`](https://github.com/qijiezhao/M2Det) 、引用次數436

- <font color="#f00">**重要**</font> **[GIoU]** Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1902.09630.pdf)  、引用次數1233

- <font color="#f00">**重要**</font> **[Libra R-CNN]** Libra R-CNN: Towards Balanced Learning for Object Detection | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.02701.pdf) 、引用次數613

- <font color="#f00">**重要**</font> **[NAS-FPN]** NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object Detection | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.07392.pdf) 、引用次數697

- **[Adaptive NMS]** Adaptive NMS: Refining Pedestrian Detection in a Crowd | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.03629.pdf) 、引用次數152

- Diversify and Match: A Domain Adaptive Representation Learning Paradigm for Object Detection | **[CVPR' 19]** |[`[pdf]`](https://arxiv.org/pdf/1905.05396.pdf)、引用次數138

- Multi-adversarial Faster-RCNN for Unrestricted Object Detection | **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1907.10343v1.pdf)、引用次數127

- A Robust Learning Approach to Domain Adaptive Object Detection | **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.02361.pdf) 、引用次數103

- Selectivity or Invariance: Boundary-Aware Salient Object Detection	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1812.10066.pdf) 、引用次數104

- <font color="#f00">**重要**</font> **[TridentNet]** Scale-Aware Trident Networks for Object Detection	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1901.01892.pdf) 、引用次數504

-  <font color="#f00">**重要**</font> **[CenterNet]** CenterNet: Keypoint Triplets for Object Detection	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.08189.pdf)、引用次數962

- **[Auto-FPN]** Auto-FPN: Automatic Network Architecture Adaptation for Object Detection Beyond Classification	| **[ICCV' 19]** |[`[pdf]`](http://openaccess.thecvf.com/content_ICCV_2019/papers/Xu_Auto-FPN_Automatic_Network_Architecture_Adaptation_for_Object_Detection_Beyond_Classification_ICCV_2019_paper.pdf) 、引用次數104

- **[ThunderNet]** ThunderNet: Towards Real-Time Generic Object Detection on Mobile Devices	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1903.11752.pdf)、引用次數142

- **[RDN]** Relation Distillation Networks for Video Object Detection	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1908.09511.pdf)、引用次數99

- **[SCAN]** Stacked Cross Refinement Network for Edge-Aware Salient Object Detection	| **[ICCV' 19]** |[`[official code]`](https://github.com/wuzhe71/SCAN) |[`[pdf]`](https://openaccess.thecvf.com/content_ICCV_2019/html/Wu_Stacked_Cross_Refinement_Network_for_Edge-Aware_Salient_Object_Detection_ICCV_2019_paper.html)、引用次數164

- **[ClusDet]** Clustered Object Detection in Aerial Images	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.08008.pdf)、引用次數100

- Few-Shot Object Detection via Feature Reweighting	 | **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1812.01866.pdf) 、引用次數281

- **[Objects365]** Objects365: A Large-Scale, High-Quality Dataset for Object Detection	| **[ICCV' 19]** |[`[pdf]`](http://openaccess.thecvf.com/content_ICCV_2019/papers/Shao_Objects365_A_Large-Scale_High-Quality_Dataset_for_Object_Detection_ICCV_2019_paper.pdf) 、引用次數104

- <font color="#f00">**重要**</font> **[EGNet]** EGNet: Edge Guidance Network for Salient Object Detection	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1908.08297.pdf) 、引用次數*400*
	

- <font color="#f00">**重要**</font>**[FCOS]** FCOS: Fully Convolutional One-Stage Object Detection	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.01355.pdf)	、引用次數**1694**

- **[RepPoints]** RepPoints: Point Set Representation for Object Detection	| **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.11490.pdf)	、引用次數**318**

- Meta-Learning to Detect Rare Objects	| **[ICCV' 19]** |[`[pdf]`](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Meta-Learning_to_Detect_Rare_Objects_ICCV_2019_paper.pdf)、引用次數117

- **[Gaussian YOLOv3]** Gaussian YOLOv3: An Accurate and Fast Object Detector using Localization Uncertainty for Autonomous Driving | **[ICCV' 19]** |[`[pdf]`](https://arxiv.org/pdf/1904.04620.pdf) [`[official code - c]`](https://github.com/jwchoi384/Gaussian_YOLOv3) 、引用次數204


- **[FreeAnchor]** FreeAnchor: Learning to Match Anchors for Visual Object Detection | **[NeurIPS' 19]** |[`[pdf]`](https://arxiv.org/pdf/1909.02466v1.pdf) 、引用次數169

- **[DetNAS]** DetNAS: Backbone Search for Object Detection | **[NeurIPS' 19]** |[`[pdf]`](https://arxiv.org/pdf/1903.10979v4.pdf) 、引用次數137

- **[AA]** Learning Data Augmentation Strategies for Object Detection | **[arXiv' 19]** |[`[pdf]`](https://arxiv.org/pdf/1906.11172.pdf) 、引用次數241

- **[Spinenet]** Spinenet: Learning scale-permuted backbone for recognition and localization | **[arXiv' 19]** |[`[pdf]`](https://arxiv.org/pdf/1912.05027.pdf) 、引用次數111

- <font color="#f00">**重要**</font> Object Detection in 20 Years: A Survey | **[arXiv' 19]** |[`[pdf]`](https://arxiv.org/pdf/1905.05055.pdf) 、引用次數508

- Salient Object Detection in the Deep Learning Era: An In-Depth Survey | **[CVPR 19]** |[`[pdf]`](https://arxiv.org/abs/1904.09146) 、引用次數288

- <font color="#f00">**重要**</font> **[CSPNet]** CSPNet: A New Backbone that can Enhance Learning Capability of CNN [`[pdf]`](https://arxiv.org/abs/1911.11929) 、引用次數736

---
## 2020

- **[CBnet]** Cbnet: A novel composite backbone network architecture for object detection | **[AAAI' 20]** |[`[pdf]`](https://arxiv.org/pdf/1909.03625.pdf) 、引用次數144

- <font color="#f00">**重要**</font> **[Distance-IoU Loss]** Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression | **[AAAI' 20]** |[`[pdf]`](https://arxiv.org/pdf/1911.08287v1.pdf) 、引用次數594

- <font color="#f00">**重要**</font> **[YOLOv4]** YOLOv4: Optimal Speed and Accuracy of Object Detection | **[arXiv' 20]** |[`[pdf]`](https://arxiv.org/pdf/2004.10934.pdf) 、引用次數3206

- <font color="#f00">**重要**</font> **[Scaled-YOLOv4]** Scaled-YOLOv4: Scaling Cross Stage Partial Network、[`[pdf]`](https://arxiv.org/pdf/2004.10934.pdf) 、引用次數212

- **[PP YOLO]** PP-YOLO: An Effective and Efficient Implementation of Object Detector |[`[pdf]`](https://arxiv.org/abs/2007.12099) 、引用次數57

- Few-Shot Object Detection With Attention-RPN and Multi-Relation Detector | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/1908.01998.pdf) 、引用次數188

- <font color="#f00">**重要**</font> Bridging the Gap Between Anchor-Based and Anchor-Free Detection via Adaptive Training Sample Selection | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/1912.02424.pdf) 、引用次數369

- Rethinking Classification and Localization for Object Detection	 | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/1904.06493.pdf) 、引用次數133

- <font color="#f00">**重要**</font> **[EfficientDet]** EfficientDet: Scalable and Efficient Object Detection | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/1911.09070.pdf) 、引用次數1498

- Dynamic Refinement Network for Oriented and Densely Packed Object Detection | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/2005.09973.pdf) 、引用次數76

- **[D2Det]** D2Det: Towards High Quality Object Detection and Instance Segmentation | **[CVPR' 20]** |[`[pdf]`](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cao_D2Det_Towards_High_Quality_Object_Detection_and_Instance_Segmentation_CVPR_2020_paper.pdf) 、引用次數71

- Prime Sample Attention in Object Detection | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/1904.04821.pdf) 、引用次數88

- Exploring Categorical Regularization for Domain Adaptive Object Detection	 | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/2003.09152.pdf) 、引用次數79

- **[NAS-FCOS]** NAS-FCOS: Fast Neural Architecture Search for Object Detection | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/1906.04423.pdf) 、引用次數79

- **[AugFPN]** AugFPN: Improving Multi-Scale Feature Learning for Object Detection	 | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/1912.05384.pdf) 、引用次數113


- Incremental Few-Shot Object Detection	 | **[CVPR' 20]** |[`[pdf]`](https://arxiv.org/pdf/2003.04668.pdf) 、引用次數90

- DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution	 | **[arXiv' 20]** |[`[pdf]`](https://arxiv.org/pdf/2006.02334v1.pdf) 、引用次數198

- <font color="#f00">**重要**</font> **[DETR]** End-to-End Object Detection with Transformers	| **[ECCV' 20]** |[`[pdf]`](https://arxiv.org/pdf/2005.12872.pdf) 、引用次數1644

- Suppress and Balance: A Simple Gated Network for Salient Object Detection	| **[ECCV' 20]** |[`[code]`](https://github.com/Xiaoqi-Zhao-DLUT/GateNet-RGB-Saliency) 、引用次數111


- **[Chained-Tracker]** Chained-Tracker: Chaining Paired Attentive Regression Results for End-to-End Joint Multiple-Object Detection and Tracking	| **[ECCV' 20]** |[`[pdf]`](https://arxiv.org/pdf/2007.14557.pdf) -引用次數81

- Highly Efficient Salient Object Detection with 100K Parameters | **[ECCV' 20]** |[`[pdf]`](https://arxiv.org/pdf/2003.05643.pdf) -引用次數63

- Arbitrary-Oriented Object Detection with Circular Smooth Label	| **[ECCV' 20]** |[`[pdf]`](https://arxiv.org/pdf/2003.05597.pdf) -引用次數87

- Soft Anchor-Point Object Detection		| **[ECCV' 20]** |[`[pdf]`](https://arxiv.org/pdf/1911.12448.pdf) -引用次數69

- **[Dynamic R-CNN]** Dynamic R-CNN: Towards High Quality Object Detection via Dynamic Training		| **[ECCV' 20]** |[`[pdf]`](https://arxiv.org/pdf/2004.06002.pdf) -引用次數91


- Multi-Scale Positive Sample Refinement for Few-Shot Object Detection		| **[ECCV' 20]** |[`[pdf]`](https://arxiv.org/pdf/2007.09384.pdf) 、引用次數53

- Few-Shot Object Detection and Viewpoint Estimation for Objects in the Wild		| **[ECCV' 20]** |[`[pdf]`](https://arxiv.org/pdf/2007.12107.pdf) 、引用次數60

- Pillar-based Object Detection for Autonomous Driving		| **[ECCV' 20]** |[`[pdf]`](https://arxiv.org/pdf/2007.10323.pdf) 、引用次數52



- Probabilistic Anchor Assignment with IoU Prediction for Object Detection		| **[ECCV' 20]** |[`[pdf]`](https://arxiv.org/pdf/2007.08103.pdf) 、引用次數76

- On the Importance of Data Augmentation for Object Detection		| **[ECCV' 20]** |[`[pdf]`](https://arxiv.org/abs/1906.11172) 、引用次數53



---
### 2021

- **[Generalized Focal Loss V2]** Generalized Focal Loss V2: Learning Reliable Localization Quality Estimation for Dense Object Detection [`[pdf]`]()、引用次數35

- **[Sparse R-CNN]** Sparse R-CNN: End-to-End Object Detection with Learnable Proposals  [`[pdf]`](https://arxiv.org/abs/2011.12450)、引用次數107

- **[Center-based 3D]** Center-based 3D Object Detection and Tracking [`[pdf]`](https://arxiv.org/abs/2006.11275)、引用次數141

- **[YOLOR]** You Only Learn One Representation: Unified Network for Multiple Tasks [`[pdf]`](https://arxiv.org/abs/2105.04206)、引用次數32


---
>補充
### 行人檢測-整理
https://github.com/xingkongliang/Pedestrian-Detection
