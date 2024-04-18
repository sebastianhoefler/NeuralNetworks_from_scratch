# Neural Networks and Deep Learning from scratch with NumPy

<p><strong><span style="font-size: larger;">Author:</span></strong> <a href="https://github.com/sebastianhoefler" style="font-size: larger;">Sebastian Hoefler</a></p>

This repository contains the implementation of different Neural Network layers from scratch using only NumPy. It is based on the programming exercises from the Deep Learning course (SS 2023) offered at the [Pattern Recognition Lab](URL "(https://lme.tf.fau.de/)") Lab of the Computer Science Department at the Friedrich-Alexander-University Erlangen-Nuernberg (FAU). The course was offered by Prof. Dr.-Ing. habil. Andreas Maier.

## Features

The primary objective of these exercises was to construct the essential components of neural networks, focusing on the development and implementation of various types of layers, optimizers, and activation functions. 

The implementation is split into several units:

1. **Feed Forward layer**: Construction of a basic `Fully Connected` layer, `ReLU` and `Softmax` activiation functions, an `Optimizer` and `Cross Entropy` loss.
2. **Convolutional layer**: Construction of the basic functions of a CNN (Convolutional Neural Network) `Conv`, `Pooling` and `Flatten` layer. Implementation of two advanced optimizers `Adam` and `SgdWithMomentum`
3. **Regularization**: Implementation of of different regularization techniques to combat overfitting and imporve performance. `Dropout`, `L1` and `L2` as well as `Batch Normalization`.
4. **Recurrent layer**: Construction of a basic `RNN` (Recurrent Neural Network) cell, as well as `TanH` and `Sigmoid` activation functions.
5. **PyTorch PV-cell defect classification**: The final assignment which included classification of defects in PV (photovoltaic) units, can be found in a different repository here (include link).

Everything can be run from `NeuralNetwork.py`. 

## Requirements

Everything was developed in `Python 3.10`. The only library needed is `NumPy`. 