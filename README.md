# Pill Identification Using Deep Learning

This project demonstrates how to build an image classification model using TensorFlow, Keras, and a dataset of 10 different items: Alaxan, Bactidol, Bioflu, Biogesic, DayZinc, Decolgen, Fish Oil, Kremil S, Medicol, and Neozep. The model is trained to classify these items using convolutional neural networks (CNNs).

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Setup](#setup)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Predictions](#predictions)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

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

## Training the Model
The model is defined using Keras' Sequential API. It includes data augmentation, convolutional layers, and dense layers.

### Load and Preprocess the Data
```python
train = tf.keras.utils.image_dataset_from_directory(
    images,
    subset="training",
    **args
)

test = tf.keras.utils.image_dataset_from_directory(
    images,
    subset="validation",
    **args
)
```

### Data Caching and Prefetching
```python
train = train.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test = test.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
```

### Define the Model
```python
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

network = [
  tf.keras.layers.Rescaling(1./255),
  layers.Conv2D(16, 4, padding='same', activation='relu', input_shape=(256,256,3)),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 4, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 4, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(len(items))
]
```

### Compile and Train the Model
```python
history_df, model = train_model(network)
```

## Evaluating the Model
### Model Summary
```python
model.summary()
```

### Training History
```python
import pandas as pd

history_df = pd.DataFrame.from_dict(history.history)
history_df[["accuracy", "val_accuracy"]].plot()
```

## Predictions
Generate predictions on the test dataset and compare them with the actual labels.
```python
preds = model.predict(test)
import numpy as np

predicted_class = np.argmax(preds, axis=1)
actual_labels = np.concatenate([y for x, y in test], axis=0)
```

## Results
Display the predictions along with the actual labels.
```python
import itertools
from PIL import Image

actual_image = [x.numpy().astype("uint8") for x, y in test]
actual_image = list(itertools.chain.from_iterable(actual_image))
actual_image = [Image.fromarray(a) for a in actual_image]

pred_df = pd.DataFrame(zip(predicted_class, actual_class, actual_image), columns=["prediction", "actual", "image"])
pred_df["prediction"] = pred_df["prediction"].apply(lambda x: items[x])
pred_df["actual"] = pred_df["actual"].apply(lambda x: items[x])
```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.
