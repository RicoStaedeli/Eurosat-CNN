# ML-Eurosat
With this project I want to create a CNN to classify images from the official [EuroSAT](https://github.com/phelber/EuroSAT) dataset. 
This dataset has Sentinel-2 images from 10 different classes. 

This project was done during the [Kaggle coding competition](https://www.kaggle.com/competitions/8-860-1-00-coding-challenge-2025).


## Table of Contents

1. [Project Description](#project-description)
2. [Project Structure](#project-structure)
3. [Running the Project](#running-the-project)
4. [Results & Evaluation](#results--evaluation)

---

## Project Description

### 1. Data Preprocessing 
First step of the project is the creation of a valid training dataset. In the test dataset on Kaggle the images are in another format. 
They used the format Level-1C with only 12 of the original 13 bands. Therefore I create with this notebook a valid training dataset with .npy images without B10.

### 2. Track_1
In Track 1 I create a CNN from Scratch. The model has three convolutional blocks with batch normalization and a ReLU Pool. 
```python
class EuroSatClassifier(nn.Module):
    def __init__(self, num_classes=10, input_channels=4):
        super(EuroSatClassifier, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # -> 64x32x32
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # -> 128x16x16
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # -> 256x8x8
        )

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
```


### 3. Track 2
Track two is a CNN enhanced with a pretrained backbone. For this I used the model **efficientnet_v2_m**. 
Data preparation is the same as in Track 1. 


---

## Project Structure

```
.
├── Generate_Training_Dataset.ipynb/    #Generate Dataset from original EuroSAT Dataset
├── Track_1_Training.ipynb
├── Track_2_Training.ipynb
└── README.MD
```

---



## Results & Evaluation

### Track 1
- **Validation Performance:** 98.22%
- **Test Performance:** 67.243%

### Track 2
- **Validation Performance:** 96.28%
- **Test Performance:** 65.699%

---
