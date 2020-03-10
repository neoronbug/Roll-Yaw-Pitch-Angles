# Roll-Yaw-Pitch-Angles
Calculations of Roll Yaw and Pitch angles from a give image that can estimate a face posture of any image.

====

Numpy & OpenCV implementation of [Head Pose Estimation on Top of Haar-Like Face Detection: A Study Using the Kinect Sensor](https://www.researchgate.net/figure/The-head-pose-rotation-angles-Yaw-is-the-rotation-around-the-Y-axis-Pitch-around-the_fig1_281587953)

![Alt text](./figure/overallflow.PNG)

Prerequisites
-------------
* [Python 3.5.2](https://www.python.org/downloads/release/python-352/)
* [Numpy](https://pypi.org/project/numpy/)
* [OpenCV 4.1.1](http://opencv.org/releases.html)


We recommend pasting the following instuctions in command prompt after Python Installation.

   pip install numpy
   pip install opencv-python

Usage
-------------

First, download the [landmark file](https://drive.google.com/open?id=1PFh3s8WL6_tmMe-oNXM73526ngXQ51TD) here. When files are download, open "pose_estimation.py" and right at the end of code, provide Landmark file path as well as image path from which you want to calculate roll, yaw and pitch angles.


Then run get_vggface.sh in the SSPP-DAN/pretrained folder to use the pre-trained VGG-Face model.

To train a model with downloaded dataset:
```
$ python train_model.py --dataset='eklfh_s1' --exp_mode='dom_3D' 
```

To test with an existing model:
```
$ python test_model.py --dataset='eklfh_s1' --exp_mode='dom_3D'  --summaries_dir 'exp_eklfh_s1/tuning/exp_2_dom__batch_64__steps_10000__lr_2e-05__embfc7__dr_0.3__ft_fc7' 
```

Results
-------------
Facial feature space (left) and its embedding space after applying DA (right). The subscript “s” and “t” in the
legend refer to the source and target domains, respectively.

![Alt text](./landmarks_detected.png)
![Alt text](./some.jpg)



Author
------------
Zeeshan Badar 
Email: mail@zeeshan.engineer
