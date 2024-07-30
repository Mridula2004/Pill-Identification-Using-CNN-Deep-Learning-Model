# Pill Identification Using Deep Learning

This project demonstrates how to build an image classification model using TensorFlow, Keras, and a dataset of 10 different items: Alaxan, Bactidol, Bioflu, Biogesic, DayZinc, Decolgen, Fish Oil, Kremil S, Medicol, and Neozep. The model is trained to classify these items using convolutional neural networks (CNNs).

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Setup](#setup)
- [Contributing](#contributing)

## Project Overview
This project uses TensorFlow and Keras to build and train a convolutional neural network (CNN) for image classification. The model is designed to classify images of 10 different items. The dataset is divided into training and validation sets with an 80-20 split.

## Dataset
The dataset consists of images stored in Google Drive. The images are loaded and converted into a TensorFlow dataset.

## Setup
### Prerequisites
- Python 3.x
- TensorFlow
- Keras
- Google Colab (if using Google Drive)

### Installing Dependencies
Install the required Python packages:
```bash
pip install tensorflow pandas pillow
```

### Mounting Google Drive (if using Google Colab)
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### Set the Path to Your Image Directory
```python
images = "/content/drive/My Drive/images2/"
```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.
