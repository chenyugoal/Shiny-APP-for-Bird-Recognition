# Bird Recognition APP Using Resnet-50 Transfered From ImageNet

Author: Chenyu Gao

E-mail: cgao17@jhu.edu

---

![Image of Birds](https://github.com/ds4ph-bme/capstone-project-chenyugoal/blob/main/bird_recognition/www/collage.jpg)

This is a shiny app for bird recognition using convolutional neural network (CNN). **Please do not attempt to sell the app for commercial use because the accuracy is still unsatisfying!** I will optimize the model and re-publish it in the future.

### Dataset & Model
Caltech-UCSD Birds-200-2011 (CUB-200-2011) is used as the dataset. It contains 11,788 images of 200 bird species, including Johns Hopkins's mascot, Blue Jay!!!!!!

To form the classification model, the Resnet-50 and two fully conneted layers are concatenated together. The Resnet-50 was pretrained on ImageNet by Keras. All of its parameters are freezed during training. Data augmentation techniques are used, including rotation, shifting, shearing, zooming, flipping, to avoid overfitting.

On the validation set (20% of the dataset), the model achieves an accuracy of ~ 0.35, which is far lower than those of the benchmarks. The simple and shallow structure of the fully connected layers is to blame. I will take some deep learning courses to build a more robust model in the future.



### Link to the Shiny App: 

https://m250.shinyapps.io/bird_recognition/

### Link to the presentation Video:

https://drive.google.com/file/d/1g8K4ayceLNup0YJtLo-WEvCNUNZjNav1/view?usp=sharing

### References
1. https://gerinberg.com/2019/12/10/image-recognition-keras/

2. https://www.r-bloggers.com/2018/02/deep-learning-image-classification-with-keras-and-shiny/

3. https://keras.io/api/applications/





*If you are not able to download the CNN model from this repository, you can also download it from here:*

*https://drive.google.com/file/d/11Zu8vG3UStsquyEWjs-hwYo-qkzPCXF_/view?usp=sharing*

---
