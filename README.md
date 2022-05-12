# A Pytorch Implementation of Mean Teacher Semi-Supervised Deep Learning
## Introduction
Follow the implementation of [mean-teacheer](https://github.com/CuriousAI/mean-teacher). I only tested performance on the cifar-10 dataset and the result is as follows:
- using resnet model and 1000 labels (50000 images), in 180 epochs, the accuracy rate can reach 80%, which is close to the effect in the paper
![Figure_1](https://user-images.githubusercontent.com/34528863/168013626-2b007b7f-5b86-493e-9b7e-06958ba0368b.png)
## Run
- torch environment: 1.8.1+cu111
- configure the location of cifar-10 dataset and run run.py
## Other
- The code is more friendly and suitable for torch beginners
- File train.py stores the core part of the algorithm, you can modify it according to your needs
- datasets folder is the interface part of the data and you can add your own data reading program
