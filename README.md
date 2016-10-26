# tf_faster_rcnn

This is an experimental tensorflow implementation of Faster R-CNN, following (Ren, Shaoqing, et al. "Faster R-CNN: Towards real-time object detection with region proposal networks." Advances in neural information processing systems. 2015.)<br />
Layers are based on py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn), with some modifications to suit them to tensorflow's mostly NHWC layers.

Base trunk is a Residual Network, with options for either 50, 101, or 152 layers. Following He, Kaiming, et al. "Deep residual learning for image recognition." arXiv preprint arXiv:1512.03385 (2015)., 
the RPN is set right after conv4, followed by ROI Pooling. Layers conv5 and up serve as the "fully-connected" layers.

created by A. Labao under Pros Naval of CVMIG Lab, University of the Philippines

# Usage
Specify the image and ground-truth folder locations in resnet_faster_rcnn.py, and run it directly in python (no need for setup). Ground truth is a csv file of the same filename number as the image it references. Each row in the csv file corresponds to an object in the referenced image, with the format x1 y1 x2 y2 label for each object bounding box.

Current optimization is done using Adam optimizer. But... results for the moment are not yet able to match the map scores in the original papers - mostly due to lack of ImageNet pre-training. Results should be better though if ImageNet pre-trained weights are used during training.

# Requirements
GTX 1070  <br />
Cuda 7.5+  <br />
Cudnn 5.0+  <br />
tensorflow v10+  <br />
and roi_pooling_op.so installed - check my other git repository [here] (https://github.com/alfonsolink/tensorflow_user_ops) for the girshick roi_pooling tensorflow wrap)
