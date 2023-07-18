# Pneumonia-Classification

## Overview

This project was insprired in part of the RSNA Pneumonia Detection [Kaggle](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/overview) challenge, where the challenge was to create an algorithm that automatically detects potential pneumonia cases.

Here’s the backstory and why solving this problem matters.

Pneumonia accounts for over 15% of all deaths of children under 5 years old internationally. In 2015, 920,000 children under the age of 5 died from the disease. In the United States, pneumonia accounts for over 500,000 visits to emergency departments and over 50,000 deaths in 2015, keeping the ailment on the list of top 10 causes of death in the country.

### Libraries Used

The following libraries are utilized in this project:

* os: For handling file paths and directories.
* torch: For building and training deep learning models using PyTorch.
* pydicom: For reading DICOM files, a common format for medical images.
* torchvision: For computer vision-related utilities, including image transformations.
* torchmetrics: For calculating evaluation metrics like accuracy, 
* precision, recall, etc.
* pandas: For data manipulation and handling CSV files.
* numpy: For numerical operations.
* matplotlib: For data visualization.

The project code consists of two main files: pneumonia_dataset.py and pneumonia_classification.ipynb.

pneumonia_dataset.py

This file contains a custom dataset class called PneumoniaDataset, which is responsible for loading and preprocessing the medical images and their corresponding labels. The dataset class uses the pydicom library to read DICOM files and applies specified transforms to preprocess the images.

### Notebook

pneumonia_classification.ipynb

This Jupyter Notebook demonstrates the entire workflow of the pneumonia classification project. It starts by importing necessary libraries and defining data preparation steps, including reading the data and splitting it into training and validation sets. The notebook then defines data transformation pipelines for preprocessing the images. It also visualizes sample images from the dataset.

The next step involves creating and training two different classification models: a custom model and a modified resnet18 model. The custom model is a simple convolutional neural network, while the modified resnet18 model uses transfer learning with pre-trained weights.

Finally, the notebook evaluates both models using accuracy, precision, recall, and confusion matrix metrics on the validation set.

### Data Preprocessing

The data preparation section of the notebook involves reading the DICOM files and extracting image pixel arrays to create the dataset. It also includes defining data transformation pipelines for both training and validation sets. The transforms consist of resizing, normalization, and random augmentations to enhance model generalization.

### Model Creation and Training

Two models are created and trained in the notebook:

1. Custom Model: A simple CNN architecture designed for pneumonia classification.
2. Modified Resnet8: A pre-trained Resnet18 model with adjustments for single-channel images.

Both models are trained using the PyTorch Lightning framework, which simplifies the training process by providing a high-level interface for training loops and callbacks.

### Model Evaluation

After training, both models are evaluated on the validation set to measure their performance. The evaluation metrics include accuracy, precision, recall, and confusion matrix. The results demonstrate the performance of each model in classifying pneumonia cases.

#### Results

Custom Classification Model:

Val Accuracy: 0.6470
Val Precision: 0.4511
Val Recall: 0.7256

Resnet18 Model:

Val Accuracy: 0.8098
Val Precision: 0.6602
Val Recall: 0.7755


### Model Interpretation

To gain insights into how the models make decisions, we employed Class Activation Mapping (CAM) techniques. CAM visually highlights the regions of the image that contribute most to the model's classification decision. By analyzing these regions, we can understand what features the model focuses on when making predictions.

Below, you will find the CAM visualizations. These visualizations provide valuable information on how the models recognize pneumonia patterns in the medical images.

![](Images/CAM.png)

### Conclusion

Th goal was to develop an effective classification model to identify potential pneumonia cases from medical images. We created and evaluated two models: a custom classification model and a modified resnet18 model.

The results from our evaluation indicate that the resnet18 model outperformed the custom model in classifying pneumonia cases. The resnet18 model achieved an accuracy of approximately 81%, significantly higher than the custom model's accuracy of around 65%. Moreover, the resnet18 model demonstrated a higher precision of about 66% and a higher recall of approximately 78%. These results suggest that the resnet18 model is more effective in correctly identifying pneumonia cases.

The implications of these precision and recall results are crucial in the context of medical image classification. A higher precision indicates that when the resnet18 model predicts a positive pneumonia case, it is correct about 66% of the time. This is important for ensuring accurate diagnoses and reducing false positives, which can lead to unnecessary treatments or medical interventions.

On the other hand, a higher recall means that the resnet18 model is capable of identifying around 78% of actual pneumonia cases present in the dataset. This high recall rate is especially critical in medical tasks like pneumonia classification, as it ensures that true positive cases are not missed, allowing for early detection and timely treatment.

In conclusion, the resnet18 model proved to be the better performing model for pneumonia classification in our project. Its superior accuracy, precision, and recall make it a more reliable tool for detecting potential pneumonia cases from medical images. The high recall rate of the resnet18 model is promising for accurate and timely identification of pneumonia patients, potentially leading to improved patient outcomes and reduced mortality rates.