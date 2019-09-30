# detectsalt

A seismic image produced from imaging the reflection coming from rock boundaries is used to image subsurface of the Earth. Each reflection represents the difference in physical properties on either sides of the interface. Although we can see rock boundaries from seismic images, they do not say much about the rock themselves. One of the challenges is to identify salt in seismic images. Salt has a lower density and higher seismic velocity than the surrounding sediments. These differences in properties create sharp reflections at the salt-sediment interface. Salt is mobile and can be any shape, so it is hard to predict the extensive of the salt in seismic images. <br/>
<br/>
We used seismic images provided by TGS as a part of Kaggle competition (https://www.kaggle.com/c/tgs-salt-identification-challenge). The dataset comprises a set of imageschosen at various locations. The images are 101x101 pixels and each pixel is classified as either salt or sediment. The goal is to segment regions that contain salt. <br/>
<br/>
We used convolutional neural networks to segment regions of salts in seismic images and able to achieve the accuracy of 79.5%.
