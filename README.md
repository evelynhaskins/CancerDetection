# Image Classification Using Convolutional Neural Networks (CNNs) for Cancer Detection

## Project Overview
This project focuses on building a Convolutional Neural Network (CNN) to classify histopathological images for cancer detection. The goal is to train a model that can predict whether a given image contains signs of cancer, which is a critical step in medical diagnostics.

I utilized a dataset from the **Histopathologic Cancer Detection** competition on Kaggle, which provides labeled images of cancerous and non-cancerous tissues. The model was trained to distinguish between the two categories using a CNN architecture.

**Note:** This project is based on the Kaggle competition [Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection/code), but I should not be included in the competition results as I relied heavily on external resources and prior work shared by the Kaggle community.

## Key Concepts and Learning
This project was my **first time working with image processing and convolutional neural networks (CNNs)**. I took advantage of various resources available on Kaggle, including kernels, discussions, and other notebooks, to understand the different steps involved in training CNNs for image classification.

I followed a structured learning process:
- **Exploratory Data Analysis (EDA)**: I began by analyzing and exploring the dataset to understand its structure and the distribution of labels.
- **Image Preprocessing**: Resizing and normalizing images to prepare them for the neural network.
- **CNN Architecture**: Built a CNN model using TensorFlow and Keras, with layers such as convolution, pooling, and dense layers to extract features and make predictions.
- **Model Evaluation**: Evaluated the model using accuracy, loss metrics, and confusion matrix. The model performance was validated using a separate validation set.

## Exploratory Data Analysis (EDA)
Before training the model, I performed some exploratory data analysis (EDA) to better understand the dataset:
- The dataset contains histopathological images with associated labels indicating whether the tissue is cancerous (1) or not cancerous (0).
- I checked the distribution of labels to ensure balanced classes, which is important for model training.

After performing the EDA, I proceeded with preprocessing the data, including resizing images to the required dimensions and scaling pixel values between 0 and 1.

## Model Results
The model was evaluated on the validation set, and here are the results:
- **Validation Loss**: 0.2923
- **Validation Accuracy**: 0.8867

These results suggest that the model is performing well in terms of classification accuracy, with a relatively low loss on the validation set.

## Resources Used
Since this was my first time implementing CNNs for image classification, I used external resources from Kaggle to guide me through the process:
- Kaggle Notebooks: I explored and built upon kernels shared by other participants in the competition.
- Kaggle Discussions: Found useful insights on image preprocessing, augmentation, and model improvement techniques.
- Kaggle Datasets: Leveraged the **Histopathologic Cancer Detection** dataset to train and test my model.

