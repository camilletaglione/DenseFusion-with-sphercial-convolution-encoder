# DenseFusion-with-sphercial-convolution-encoder

6D pose estimation method developed during a master course. 
This method is based on the architecture of the DenseFusion method with the addition of a spherical convolution based encoder inspired by the DualPoseNet method.
The method has been developed for the YCB-Video database with a possible extension to LineMod in the future.
The initial encoder is coded in TensorFlow, but the Densefusion architecture is coded in PyTorch.
So the new encoder does not use the same spherical convolution function as the original encoder.  
To realise this method a study of many methods was carried out.
The methods to be studied are those present in the citation file.
These methods use visual detection, tactile detection or both.
As well as a tool to visualise the installation in the form of a video on the YCB database.

Architecture :

![Untitled Diagram drawio (1)](https://user-images.githubusercontent.com/61682491/172632084-d7a1215f-6ff2-423f-9a4d-8843fe29d321.png)

Results :
![seg2](https://user-images.githubusercontent.com/61682491/172631251-479a27e0-fd54-4a38-a0ba-7044ffe46cb3.png)
![segmentationresult](https://user-images.githubusercontent.com/61682491/172631256-03c411cf-e6d6-4add-b67b-66d887086ce3.png)
