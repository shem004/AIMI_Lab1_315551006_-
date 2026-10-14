# Pneumonia Classification (Lab 1)

This repository contains the implementation of a binary image classifier designed to detect pneumonia in chest X-ray images using PyTorch. 

## Requirements
* Python 3.8+
* PyTorch & Torchvision
* Kaggle API (`kaggle`)
* OpenCV, Seaborn, Matplotlib, scikit-learn

## Repository Structure
* `train.py`: Downloads the Kaggle dataset, processes data augmentation, applies weighted CrossEntropyLoss for class imbalance, and trains the ResNet architecture. Saves the best checkpoint.
* `inference.py`: Loads the saved `.pt` weights and evaluates the model against the testing dataset. Generates the final accuracy metrics and plots the confusion matrix.

## Instructions for Execution

1. **Setup Credentials:**
   Ensure your Kaggle API key is active. In order to find your key, you have to open your Kaggle account and click one your profile. Then select settings and API key. After that generate your Kaggle API key. The script automatically uses the credentials provided within `train.py` to stream the dataset.

2. **Train the Model:**
   ```bash
   python train.py

3. **Run the Inference:**
   ```bash
   python inference.py