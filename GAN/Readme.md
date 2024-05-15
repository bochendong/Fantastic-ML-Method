# GAN

<p align="center">
    <img src = 'img/01.png' width = 500px height = 340px>
</p>


<p align="center">
    <img src = 'img/02.png' width = 500px height = 340px>
</p>

## What can GAN Do?

<p align="center">
    <img src = 'img/13.png' width =600px height = 340px>
</p>

Choi, Y., Choi, M.-J., Kim, M., Ha, J.-W., Kim, S., & Choo, J. (2017). StarGAN: Unified generative adversarial networks for multi-domain image-to-image translation. CoRR, abs/1711.09020. http://arxiv.org/abs/1711.09020

<p align="center">
    <img src = 'img/14.png' width =600px height = 340px>
</p>

Wu, J., Zhang, C., Xue, T., Freeman, W. T., & Tenenbaum, J. B. (2016). Learning a probabilistic latent space of object shapes via 3D generative-adversarial modeling. CoRR, abs/1610.07584. Retrieved from http://arxiv.org/abs/1610.07584


## Basic idea of GAN

<p align="center">
    <img src = 'img/03.png' width = 500px height = 340px>
</p>


<p align="center">
    <img src = 'img/04.png' width = 500px height = 340px>
</p>


## Animal Face Generation

**100 updates**

<p align="center">
    <img src = 'img/05.png' width = 361px height = 340px>
</p>

**1000 updates**

<p align="center">
    <img src = 'img/06.png' width = 361px height = 340px>
</p>


**5000 updates**

<p align="center">
    <img src = 'img/07.png' width = 361px height = 340px>
</p>

**10000 updates**
<p align="center">
    <img src = 'img/08.png' width = 361px height = 340px>
</p>

**20000 updates**

<p align="center">
    <img src = 'img/09.png' width = 361px height = 340px>
</p>

**50000 updates**

<p align="center">
    <img src = 'img/10.png' width = 361px height = 340px>
</p>


## GAN Structure

<p align="center">
    <img src = 'img/11.png' width =600px height = 340px>
</p>

**训练过程：**
- 从真实数据集中采样m个**真实样本**，假设每个样本的大小为 (3, 32, 32)。
- 生成m个随机噪声样本，假设每个**噪声样本**大小为 （128, 1, 1）。
- 将随机噪声放入**生成器G**中，生成m个生成样本，每个**生成样本**的大小为 (3, 32, 32)。
- 同时将**真实样本**与**生成样本**放入判别器中，判别器需要通过梯度下降（gradient descent）的方法，将自己的权重更新，使得**判别器D**能尽可能的分辨出**真实样本**与**生成样本**。
- 训练K次**判别器D**后，使用较小的学习率来更新一次**生成器G**的参数，

```python
for batch in range(batch_size):)
    for 轮数 in range(判别器训练轮数):
        步骤一 训练D
    步骤二 训练G
```

<p align="center">
    <img src = 'img/15.png' width =600px height = 300px>
</p>


**Question:**
- 什么时候我们认为生成样本是足够好的？
- 理想状况下，判别器判别*真实样本*和*生成样本*的准确率是多少？


**Sample Code**

```python
for epoch in range(num_epochs):
    for images, lables in train_loader:
        real_images = images.to(device)                         # Move the real images to the device (GPU or CPU)
        real_labels = torch.ones(batch_size, 1).to(device)      # Create real labels (1s) with the current batch size
        fake_labels = torch.zeros(batch_size, 1).to(device)     # Create fake labels (0s) with the current batch size

        # --------- Train the Teacher (D) --------- #
        d_optimizer.zero_grad()                                 # Zero the gradients of the Teacher optimizer

        outputs = Teacher(real_images)                          # Pass real images through the Teacher
        d_real_loss = loss_fn(outputs, real_labels)             # Calculate the loss for real images 
                                                                # (how well does it recognize real images)

        z = torch.randn(batch_size, noise_size).to(device)      # Generate random noise to produce fake images

        fake_images = Student(z)                                # Generate fake images from noise
        outputs = Teacher(fake_images.detach())                 # Pass the fake images through the Teacher [1]
        d_fake_loss = loss_fn(outputs, fake_labels)             # Calculate the loss for fake images 
                                                                # (how well does it recognize fake images)

        d_loss = d_real_loss + d_fake_loss                      # Calculate total Teacher loss

        d_loss.backward()                                       # Update Teacher's network weights
        d_optimizer.step()

        # --------- Train the Student (G) --------- #
        g_optimizer.zero_grad()                                 # Zero the gradients of the Student optimizer

        outputs = Teacher(fake_images)                          # Pass the fake images to the Teacher again
        g_loss = loss_fn(outputs, real_labels)                  # Calculate the Student loss 
                                                                # (how well does it fool the Teacher)

        g_loss.backward()                                       # Update Student's network weights
        g_optimizer.step()
```

**Note:**

Why we use `fake_images.detach()` in [1]?

If we do not detach() fake_images, then the update of Teacher will also update the student.


<p align="center">
    <img src = 'img/16.png' width = 600px height = 130px>
</p>


## Why cannot we use sample Generator?

<p align="center">
    <img src = 'img/17.png' width = 600px height = 400px>
</p>

## GANS

### CGAN

A Conditional Generative Adversarial Network (CGAN) is an extension of the basic Generative Adversarial Network (GAN) framework that allows the generation of synthetic data conditioned on certain inputs. This conditioning can be in the form of labels, tags, or other types of data that direct the generation process, enabling the model to produce more specific or targeted outputs

<p align="center">
    <img src = 'img/18.png' width = 600px height = 350px>
</p>


Here is an exmaple of Generator and Discrimator on MNIST

```python
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, image_size),
            nn.Tanh()
        )

    def forward(self, noise, context):
        noise = noise.view(-1, latent_dim)
        context_feature = self.label_emb(context)
        x = torch.cat([noise, context_feature], 1)

        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.model = nn.Sequential(
            nn.Linear(image_size + num_classes, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, context):
        img = img.view(-1, image_size)
        context_feature = self.label_emb(context)

        x = torch.cat((img, context_feature), dim=1)
        return self.model(x)
```


### Generator

**ConvTranspose2d**

ConvTranspose2d, often called transposed convolution or deconvolution, is the inverse process of the convolution operation and is widely used in image generation tasks such as generative adversarial networks (GANs) and autoencoders. Although it is called deconvolution, it is not the inverse operation of the convolution operation in the strict sense, but a special convolution operation used to increase the spatial size of the input data.


<p align="center">
    <img src = 'img/12.png' width = 361px height = 600px>
</p>


https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md


**upsample**

https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html


### Discrimator

The discriminator in a GAN is simply a classifier. It tries to distinguish real data from the data created by the generator.



