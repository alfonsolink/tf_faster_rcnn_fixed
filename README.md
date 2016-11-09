# tf_faster_rcnn

-- experimental tensorflow implementation of Faster R-CNN, by (Ren, Shaoqing, et al. "Faster R-CNN: Towards real-time object detection with region proposal networks." Advances in neural information processing systems. 2015.)<br />
Layers are based on py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn), but modified to suit them to tensorflow's mostly NHWC layers.

Base trunk is a Residual Network, with options for either 50, 101, or 152 layers. Following He, Kaiming, et al. "Deep residual learning for image recognition." arXiv preprint arXiv:1512.03385 (2015)., 
the RPN is set right after conv4, followed by ROI Pooling. Layers conv5 and up serve as the "fully-connected" layers.

created by A. Labao under Pros Naval of CVMIG Lab, Univ of the Philippines

# Usage
Specify image and ground-truth folder locations in resnet_faster_rcnn.py, and run it directly in python. Ground truth is a csv file referencing an image. Each row in the csv file has to match an object in the image, with the format (x1 y1 x2 y2 label) for each object bounding box.  Images and bounding boxes are to be rescaled to H X W of 600 X 1000 as pre-processing. Current optimization is done using Momentum Optimizer with Nesterov Momentum

# Requirements
GTX 1070  <br />
OpenCV 3.1 <br />
Cuda 7.5+  <br />
Cudnn 5.0+  <br />
tensorflow v10+  <br />
and roi_pooling_op.so installed - check my other git repository [here] (https://github.com/alfonsolink/tensorflow_user_ops) for the girshick roi_pooling tensorflow wrap.

# Notes:
this has no dynamic tensors accdg. to image size -- all images are rescaled to 600 x 1000, I'll come up with a version that has dynamic tensors soon. 

training results are going to be better if input images have dynamic sizes, which is implemented in the the other repo [here](https://github.com/alfonsolink/tf_faster_rcnn_dynamic)

results are much better if network is loaded with pre-trained imagenet weights, which can be downloaded [here](https://1drv.ms/f/s!AtPFjf_hfC81kUrPD2Kazg1Gtkz6) for a simple saver.restore().
