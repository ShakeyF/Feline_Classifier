# Description

Feline_Classifier with CNN construction and ImageNet input and blurred image input.

# Requirements

## Deblurring Network

* Tensorflow is needed.<br>
* Install Keras as follow :<br>
```
$ pip install keras -U --pre
```
## Classification Network
* Pytorch is needed.<br>
* Install as follow:
```
$pip3 install http://download.pytorch.org/whl/cu80/torch-0.4.0-cp36-cp36m-win_amd64.whl 
$pip3 install torchvision
```
# Code organization
image_download.ipynb       --  Run to download all the image from urls in the root txt document
data_establish.ipynb       --  Run to resize and blur all the images in the rootpath and pack them into dataset package
VGG16_NET.ipynb            --  Run the training of our classification network with VGG16 model
ALEX_NET.ipynb             --  Run the training of our classification network with AlexNet model
Deblurring_Network.ipynb   --  Run the training of our deblurring network
demo.ipynb                 --  Run a demo of our deblur and classification code (required utils, trained model and images all available on Google drive. Have open to TA Sneha. Link:https://drive.google.com/open?id=18WDIGNO6KX9ycZ1ca7ZeqbkON9mcJIPQ)
utils.py                   --  The functions our demo will need
/data                      --  Three datasets of different blur kernel sizes for model training.
