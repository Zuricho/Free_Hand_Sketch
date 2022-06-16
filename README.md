# Free-hand sketch classification problem

Bozitao Zhong    220030910014

Course project for SJTU CS420 Machine Learning course

## Main Ideas

### Background

**QuickDraw** is one of the largest free-hand sketch datasets. It includes 345 categories of common objects, and each one contains 70 thousand training, 2.5 thousand validation and 2.5 thousand test samples. The dataset is available at https://magenta.tensorflow.org/sketch_rnn. 

In this project, we choose 25 categories (cow, panda, lion, tiger, raccoon, monkey, hedgehog, zebra, horse, owl, elephant, squirrel, sheep, dog, bear, kangaroo, whale, crocodile, rhinoceros, penguin, camel, flamingo, giraffe, pig, cat) from **QuickDraw** for the sketch classification problem. Each sketch individual is translated to a 28*28 sketch image as the model input.

### Transform free-hand sketch dataset into pixel image dataset

The original sketches in QuickDraw are described as vectorized sequences, which we want to further translated into sketch pixel images. 

In this project, I used some functions from **pix2seq** https://github.com/CMACH508/RPCL-pix2seq which offers an approach to create the pixel-formed sketch images to build mine dataset transform tools `data_transform.py`. Why I rebuild that code is because that code has too much setting which are too fuzzy for me, and I'm using PyTorch rather than Tensorflow.

### Baseline models

For baseline models, I selected support vector machine and simple fully-connected neural network as baseline. These models are relative simple for this complicated classification problem. (All baseline models are built with scikit-learn package)

### Deep learning models

For deep learning models, I used a series of models including:

- Fully connected neural network
- Convolution neural network
- Variational Auto-Encoder
- Masked CNN

## Methods and Algorithms

### Baseline models

Our 2 baseline models are built with `scikit-learn`. 





## Experimental settings

### Dataset

Dataset has 25 classifications, each image is classified into 1 of the 25 classifications. Each pixel image is 28*28 large unless have specific setting.

### Training setting





## Conclusion



## Group member and contribution

My group only have one member: Bozitao Zhong (student ID: 220030910014), all work is done by me, including designing and building models, conduct experiments, writing report and visualization.

The code of this project is available at https://github.com/Zuricho/Free_Hand_Sketch
