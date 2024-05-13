# GAN

<p align="center">
    <img src = './img/01.png' width = 500px height = 340px>
</p>


<p align="center">
    <img src = './img/02.png' width = 500px height = 340px>
</p>

## Basic idea of GAN

<p align="center">
    <img src = './img/03.png' width = 500px height = 340px>
</p>


<p align="center">
    <img src = './img/04.png' width = 500px height = 340px>
</p>


## Animal Face Generation

**100 updates**

<p align="center">
    <img src = './img/05.png' width = 361px height = 340px>
</p>

**1000 updates**

<p align="center">
    <img src = './img/06.png' width = 361px height = 340px>
</p>


**5000 updates**

<p align="center">
    <img src = './img/07.png' width = 361px height = 340px>
</p>

**10000 updates**
<p align="center">
    <img src = './img/08.png' width = 361px height = 340px>
</p>

**20000 updates**

<p align="center">
    <img src = './img/09.png' width = 361px height = 340px>
</p>

**50000 updates**

<p align="center">
    <img src = './img/10.png' width = 361px height = 340px>
</p>


## GAN Structure


<p align="center">
    <img src = './img/11.png' width =600px height = 340px>
</p>


### Generator

**ConvTranspose2d**

ConvTranspose2d, often called transposed convolution or deconvolution, is the inverse process of the convolution operation and is widely used in image generation tasks such as generative adversarial networks (GANs) and autoencoders. Although it is called deconvolution, it is not the inverse operation of the convolution operation in the strict sense, but a special convolution operation used to increase the spatial size of the input data.


<p align="center">
    <img src = './img/12.png' width = 361px height = 600px>
</p>


https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md


**upsample**

https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html


### Discrimator

The discriminator in a GAN is simply a classifier. It tries to distinguish real data from the data created by the generator.


## Why cannot we use sample Generator?

To be continue...


## GANS

### CGAN

A Conditional Generative Adversarial Network (CGAN) is an extension of the basic Generative Adversarial Network (GAN) framework that allows the generation of synthetic data conditioned on certain inputs. This conditioning can be in the form of labels, tags, or other types of data that direct the generation process, enabling the model to produce more specific or targeted outputs


