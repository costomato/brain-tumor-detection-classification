# Brain Tumor Detection using MRI Images

This project aims to classify brain MRI images into four categories: **glioma**, **meningioma**, **pituitary**, and **no tumor**. We use Convolutional Neural Networks (CNNs) and PyTorch for training the model. This is a deep learning-based approach to help in the early detection of brain tumors, which can aid in better diagnosis and treatment.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Setup](#setup)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [How to Use the Model](#how-to-use-the-model)
- [Results](#results)
- [License](#license)

## Introduction

Brain tumors are masses of abnormal cells that can be malignant or benign. Detecting and classifying brain tumors at an early stage is critical for effective treatment. In this project, we apply deep learning techniques to classify brain tumor MRI images. This can help doctors in diagnosing brain tumors quickly and accurately.

## Dataset

The dataset used in this project is a combination of three datasets:

- **SARTAJ dataset**
- **Br35H dataset**
- **figshare dataset**

The dataset contains 7023 images of brain MRIs, classified into four categories:

- **Glioma**
- **Meningioma**
- **Pituitary**
- **No tumor**

The images in the dataset have varying sizes, and we perform necessary preprocessing steps to ensure that the model receives consistent input.

**Dataset Source**: [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

## Technologies Used

- **Python**: Programming language used for implementation.
- **PyTorch**: Deep learning framework for building and training the model.
- **OpenCV**: Used for image preprocessing.
- **Matplotlib**: Used for displaying images.
- **Kaggle API**: For downloading the dataset directly from Kaggle.
- **Google Colab**: For training the model in the cloud with GPU support.

## Setup

Follow these steps to set up the project locally or on Google Colab.

### 1. Clone the repository

```bash
git clone https://github.com/costomato/brain-tumor-detection-classification.git
cd brain-tumor-detection-classification
```

### 2. Install dependencies

You can use a virtual environment or install dependencies globally.

```bash
pip install -r requirements.txt
```

Here is the content of `requirements.txt`:

```
torch==1.13.1
torchvision==0.14.1
numpy==1.21.2
opencv-python==4.5.3.56
PIL==9.0.1
tqdm==4.62.3
requests==2.26.0
matplotlib==3.4.3
```

### 3. Download the dataset

Follow the instructions below to download the dataset directly into your Google Colab environment.

```python
!pip install kaggle
!mkdir ~/.kaggle
!cp /path/to/your/kaggle.json ~/.kaggle/

# Download the dataset using Kaggle API
!kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
```

## Preprocessing

Before feeding the images into the model, we need to preprocess them. The following steps are performed:

1. **Resizing**: All images are resized to a consistent size (256x256 pixels).
2. **Grayscale Conversion**: Images are converted to grayscale and then back to 3-channel images to match the input size expected by the model.
3. **Normalization**: The pixel values are normalized to have a mean of `[0.485, 0.456, 0.406]` and a standard deviation of `[0.229, 0.224, 0.225]`.

Here is the code for preprocessing:

```python
from PIL import Image, ImageOps
import requests
from io import BytesIO

def preprocess_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # Convert to grayscale
    img = img.convert('L')

    # Convert grayscale to 3 channels
    img = Image.merge("RGB", (img, img, img))

    # Resize to the desired size
    img = img.resize((256, 256))

    return img
```

## Model Architecture

For this project, we use a **ResNet-18** architecture, which is a popular CNN model known for its depth and ability to learn complex features. We load the pre-trained ResNet model and replace the final fully connected layer to classify the images into four categories.

```python
from torchvision import models
import torch.nn as nn

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 4)  # 4 classes: glioma, meningioma, pituitary, no tumor
```

## Training the Model

### Hyperparameters

- **Learning rate**: 0.001
- **Batch size**: 32
- **Epochs**: 10

We use the **Adam optimizer** and **cross-entropy loss** for training the model.

### Training Code:

```python
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set up data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load training and validation datasets
train_dataset = datasets.ImageFolder('path_to_training_data', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the model and optimizer
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 4)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch [{epoch+1}/10], Loss: {running_loss/len(train_loader)}, Accuracy: {100 * correct / total}%")
```

## Evaluation

After training the model, evaluate its performance on the validation dataset. This involves measuring the **accuracy** and **loss** of the model on unseen data.

```python
# Evaluate the model on validation data
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in validation_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Validation Accuracy: {100 * correct / total}%")
```

## Prediction

You can use the trained model to predict the class of a new image. Here's the function to make predictions from an image URL:

```python
def predict_image(image_url):
    img = preprocess_image(image_url)
    img_tensor = transforms.ToTensor()(img).unsqueeze(0)  # Add batch dimension
    outputs = model(img_tensor)
    _, predicted = torch.max(outputs, 1)
    class_names = ['glioma', 'meningioma', 'no tumor', 'pituitary']
    predicted_class = class_names[predicted.item()]
    return predicted_class
```

## How to Use the Model

1. Preprocess the image (resize and normalize).
2. Pass the preprocessed image to the model.
3. Display the predicted class.

```python
image_url = "https://example.com/path/to/image.png"
predicted_class = predict_image(image_url)
print(f"Predicted Class: {predicted_class}")
```

## Results

The model achieves an accuracy of **99.24%** on the validation set, which indicates the effectiveness of the model in classifying brain tumors. This result demonstrates the potential for using deep learning techniques in medical image classification.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: Always ensure that the images used for prediction are correctly preprocessed and fit the model’s input requirements for accurate results.
