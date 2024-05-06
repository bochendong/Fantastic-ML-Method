# Image Classification


## Linear Classifier

<p align="center">
    <img src = './img/01.png' width = 600 height = 240>
</p>

**Interpretation of linear classifiers as template matching**

A interpretation for the weights W is that each row of W corresponds to a template for one of the classes

<p align="center">
    <img src = './img/03.png' width = 600 height = 90>
</p>

**Analogy of images as high-dimensional points**. 

Since the images are stretched into high-dimensional column vectors, we can interpret each image as a single point in this space (e.g. each image in CIFAR-10 is a point in 3072-dimensional space of 32x32x3 pixels). Analogously, the entire dataset is a (labeled) set of points.

<p align="center">
    <img src = './img/02.png' width = 400 height = 330>
</p>

## Neural Network

## Multi-Linear Layer

<p align="center">
    <img src = './img/04.png' width = 800px height = 300px>
</p>


```python
class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear (32 * 32 * 3, 1000),          # 3072 -> 1000          /3
            nn.ReLU(inplace=True),
            nn.Linear (1000, 100),                  # 1000 -> 100       /10
            nn.ReLU(inplace=True),
            nn.Linear (100, 10),                    # 100 -> 10         /10
        )
    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = self.classifier(x)

        return x
```

Notice that the `ReLU` between each linear layer helps improve accuracy. These `ReLU` activations introduce non-linearity to the model, enabling it to learn more complex patterns in the data.


|Models| Test Accuracy|
|:--|:--:|
|KNN| 22%|
|Linear Model| 40%|
|Linear Model with Relu| 56%|


## Conv Layer

Convolutional layers are responsible for learning spatial hierarchies in the input data. Each convolutional layer applies a set of learnable filters to the input, producing feature maps. These feature maps capture different aspects of the input data.

<p align="center">
    <img src = './img/05.png' width = 800px height = 350px>
</p>


## MaxPool Layer

Max-pooling layers are used to downsample the feature maps, reducing their spatial dimensions while retaining important information. Max-pooling helps in achieving translation invariance and reduces computational complexity.

<p align="center">
    <img src = './img/06.png' width = 800px height = 350px>
</p>


```python
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),   
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                    

            nn.Conv2d(64, 128, kernel_size=3, padding=1),         
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),       
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(4 * 4 * 256, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 4 * 4 * 256)  # Adjusted for the added depth
        x = self.classifier(x)
        return x
```

<p align="center">
    <img src = './img/13.png' width = 800px height = 400px>
</p>


**Summary**

- **Filter Size**:
  - A Filter size of 3x3 is commonly used in CNNs, 
  - If we are using big Filter size, will lead to shrink too fast. Shrinking too fast is not good, does not work well.
  - Smaller kernel sizes tend to capture finer details, while larger kernel sizes capture more global features.
- **Padding**:
  - Padding helps in preserving spatial information at the borders of the image. 
  - Without padding, the spatial dimensions would shrink with each convolutional layer.
- **Multiple Convolutions Followed by Max Pooling**:
  - Multiple convolutional layers followed by a max-pooling layer is a common design pattern.
  - Convolutional layers extract features from the input images, while max-pooling layers downsample the feature maps, reducing computational complexity and the spatial dimensions of the feature maps.
  - This pattern helps the model learn hierarchical features by gradually reducing the spatial dimensions while increasing the depth (number of channels) of the feature maps.


|Models| Test Accuracy|
|:--|:--:|
|KNN| 22%|
|Linear Model| 56%|
|Simple Conv | 65%|
|Conv with maxpooling| 75%|
|Conv with maxpooling and padding| 78%|

# Overfitting

Overfitting occurs when a machine learning model learns to perform exceptionally well on the training data but fails to generalize to unseen or new data.

<p align="center">
    <img src = './img/09.png' width = 500px height = 200px>
</p>


## Dropout Layer

Dropout is a regularization technique commonly used in neural networks, especially deep learning models, to **prevent overfitting**.

The idea behind dropout is to randomly "drop out" (i.e., set to zero) a proportion of neurons in the neural network during each training iteration. 

This means that the neurons selected for dropout do not contribute to the forward pass, nor do they participate in the backpropagation of gradients during training.


<p align="center">
    <img src = './img/07.png' width = 800px height = 400px>
</p>


```python
self.classifier = nn.Sequential(
    nn.Linear(4 * 4 * 256, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),            # <-- used after activation function
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(4096, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.2),
    nn.Linear(512, 10)
)
```

## Data Augmentation

Data augmentation is a technique used to artificially increase the size and diversity of a dataset by applying various transformations to the existing data samples. These transformations include but are not limited to:

- **Geometric transformations**: Such as rotation, translation, scaling, flipping, and cropping.
- **Color transformations**: Such as brightness adjustment, contrast adjustment, hue variation, and saturation adjustment.
- **Noise injection**: Adding random noise to the data, which helps the model become more robust to variations in the input.


<p align="center">
    <img src = './img/08.png' width = 800px height = 400px>
</p>


```python
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])
```

|Models| Test Accuracy|
|:--|:--:|
|KNN| 22%|
|Linear Model| 56%|
|Simple Conv | 65%|
|Conv with maxpooling| 75%|
|Conv with maxpooling and padding| 78%|
|Conv with Dropout and Data Augmentation| 82%|

# Learning Rate Decay

Learning rate decay, also known as learning rate annealing or learning rate scheduling, is a technique used to improve the training of machine learning models, particularly deep neural networks. 

<p align="center">
    <img src = './img/11.png' width = 800px height = 350px>
</p>

There are several reasons why learning rate decay is beneficial:

- **Convergence**: Initially, a high learning rate helps the model converge quickly towards a good solution. However, as training progresses and the model approaches a local minimum, a lower learning rate helps the optimization process converge more accurately and efficiently.
- **Stability**: A decreasing learning rate stabilizes the training process by reducing the likelihood of oscillations or divergence in the loss landscape. It allows the optimization algorithm to navigate more smoothly towards the global or local minimum.
- **Improved Generalization**: Learning rate decay can help prevent overfitting by regularizing the optimization process. By gradually reducing the learning rate, the model becomes less sensitive to small fluctuations in the training data and is more likely to generalize well to unseen data.
- **Fine-Tuning**: Learning rate decay enables fine-tuning of the model parameters during the later stages of training. This allows the model to make smaller, more precise adjustments to the parameters, leading to better convergence and performance.

<p align="center">
    <img src = './img/10.png' width = 800px height = 400px>
</p>

```python
for epoch in range(num_epochs):

    ... training ...

    scheduler.step()


scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
```


```python
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = 0.05 * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(num_epochs):

    ... training ...

    adjust_learning_rate(optimizer, epoch)

```


# Batch Norm

Batch Normalization (BN) is a technique used in neural networks to improve the training speed, stability, and performance.

It normalizes the activations of each layer by adjusting and scaling them so that they have a mean of zero and a standard deviation of one. 

This normalization is performed over the mini-batches during training.
<p align="center">
    <img src = './img/12.png' width = 600px height = 300px>
</p>


```python
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
            nn.Softmax(dim=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512)
        x = self.classifier(x)
        return x
```


<p align="center">
    <img src = './img/14.PNG' width = 600px height = 300px>
</p>


|#|Models| Test Accuracy| Year|
|:--:|:--|:--:|:--|
|1|KNN| 22%|1951|
|2|Linear Model| 56%|1958|
|3|Conv with big filter size | 65%|2012|
|4|Conv with small filter size and maxpooling| 75%|2013 - 2015|
|5|[4] with padding| 80%|2013 - 2015|
|6|[5] with Dropout and Data Augmentation| 82%|2015 - 2017|
|7|[6] with Learning rate Deacy| 86%|2015 - 2017|
|8|[8] with Batch Norm| 93%|2015 - 2017|

