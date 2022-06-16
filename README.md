# Free-hand sketch classification problem

Bozitao Zhong    220030910014

Course project for SJTU CS420 Machine Learning course

## Introduction

Free-hand sketch drawing is a valuable way to convey messages and express emotions throughout human history. Sketches contain not only vivid categorical features of the target objects but also abstract, variant visual appearances. In this project, our target is to construct deep learning models for sketch image classification.



### Useful links

dataset source: https://github.com/magenta/magenta/tree/main/magenta/models/sketch_rnn



## Dataset

**QuickDraw** is one of the largest free-hand sketch datasets. It includes 345 categories of common objects, and each one contains 70 thousand training, 2.5 thousand validation and 2.5 thousand test samples. The dataset is available at https://magenta.tensorflow.org/sketch_rnn. The original sketches in QuickDraw are described as vectorized sequences, which can be further translated into sketch images. In this project, our dataset is created by **pix2seq** https://github.com/CMACH508/RPCL-pix2seq which offers an approach to create the pixel-formed sketch images.
In this project, we choose 25 categories (cow, panda, lion, tiger, raccoon, monkey, hedgehog, zebra, horse, owl, elephant, squirrel, sheep, dog, bear, kangaroo, whale, crocodile, rhinoceros, penguin, camel, flamingo, giraffe, pig, cat) from **QuickDraw** for the sketch classification problem. Each sketch individual is translated to a 28*28 sketch image as the model input.

## Main Ideas



## Methods and algorithms





## Experimental settings





## Conclusion



## Group member and contribution

My group only have one member: Bozitao Zhong (student ID: 220030910014), all work is done by me, including designing and building models, conduct experiments, writing report and visualization.

The code of this project is available at https://github.com/Zuricho/Free_Hand_Sketch
