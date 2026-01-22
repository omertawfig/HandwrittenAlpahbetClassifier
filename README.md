# Handwritten Alphabet Classifier

A comprehensive machine learning project for classifying handwritten alphabets (A-Z) using multiple classification algorithms including SVM, Logistic Regression, and Neural Networks.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architectures](#model-architectures)
- [Results](#results)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Output Files](#output-files)
- [Contributors](#contributors)
- [License](#license)

## Overview

This project implements and compares various machine learning algorithms for handwritten alphabet recognition. The system processes grayscale images of handwritten letters (A-Z) and classifies them using different approaches, providing a comprehensive analysis of each method's performance.

## Features

- **Multiple Classification Algorithms**:
  - Support Vector Machine (Linear and RBF kernels)
  - Logistic Regression (implemented from scratch with One-vs-All approach)
  - Neural Networks (two architectures with varying complexity)

- **Data Analysis & Visualization**:
  - Class distribution analysis
  - Sample image visualization
  - Confusion matrices for all models
  - Learning curves and performance metrics

- **Custom Letter Generation**:
  - Synthetic letter generation for testing
  - Handwritten-style letter generation with jitter
  - Model validation on specific test cases

- **Model Optimization**:
  - Early stopping for logistic regression
  - Learning rate scheduling
  - L2 regularization
  - Mini-batch gradient descent

## Dataset

- **Source**: A_Z Handwritten Data (CSV format)
- **Image Size**: 28x28 pixels (784 features)
- **Classes**: 26 (A-Z)
- **Format**: Grayscale images flattened to 1D arrays
- **Preprocessing**: MinMax normalization (0-1 scaling)

### Data Split
- **Training Set**: 64% of total data
- **Validation Set**: 16% of total data
- **Test Set**: 20% of total data

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
opencv-python
```

## Usage

Run the main script to train all models and generate results:

```bash
python main.py
```

The script will:
1. Load and preprocess the dataset
2. Generate data visualizations
3. Train SVM models (Linear and RBF)
4. Train Logistic Regression model from scratch
5. Train Neural Network models
6. Evaluate all models on the test set
7. Test specific letters (A, H, M, E, D, O, R, I, Z, L, N)
8. Save results and visualizations

## Model Architectures

### 1. Support Vector Machine (SVM)
- **Linear Kernel**: For linearly separable patterns
- **RBF Kernel**: For complex, non-linear decision boundaries
- **Optimization**: Limited iterations (1000) with increased cache size (1000)

### 2. Logistic Regression (One-vs-All)
- **Implementation**: From scratch using NumPy
- **Features**:
  - Sigmoid activation function
  - L2 regularization
  - Learning rate scheduling
  - Mini-batch gradient descent (batch size: 32)
  - Early stopping (patience: 5 epochs)
- **Hyperparameters**:
  - Initial learning rate: 0.01
  - Regularization: 0.01
  - Max epochs: 100

### 3. Neural Network 1 (Simple Architecture)
```
Input (784) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(26, Softmax)
```

### 4. Neural Network 2 (Complex Architecture)
```
Input (784) → Dense(256, ReLU) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(26, Softmax)
```

- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Training**: 10 epochs with 20% validation split

## Results

The project generates comprehensive evaluation metrics for all models:

- **F1 Scores** (Macro and Weighted averaging)
- **Confusion Matrices** for visual performance analysis
- **Learning Curves** showing training and validation metrics
- **Best Model Selection** based on F1 scores

### Output Metrics
- `svm_results.txt`: F1 scores for Linear and RBF SVM
- `logistic_regression_results.txt`: F1 scores for Logistic Regression
- `nn_results.txt`: F1 scores for both Neural Networks and best model indicator

### Letter Testing
The system tests the following letters (representing group members' names):
- A, H, M, E, D, O, R, I, Z, L, N

## Project Structure

```
HandwrittenAlpahbetClassifier/
│
├── main.py                                    # Main script
├── A_Z_Handwritten_Data.csv                   # Dataset (not included in repo)
├── Report.pdf                                 # Project report
├── README.md                                  # This file
│
├── Generated Files:
│   ├── class_distribution.png                 # Class distribution visualization
│   ├── sample_images.png                      # Sample handwritten letters
│   ├── svm_confusion_matrices.png             # SVM confusion matrices
│   ├── svm_results.txt                        # SVM performance metrics
│   ├── logistic_regression_curves.png         # LR learning curves
│   ├── logistic_regression_confusion.png      # LR confusion matrix
│   ├── logistic_regression_results.txt        # LR performance metrics
│   ├── neural_network_curves.png              # NN learning curves
│   ├── best_nn_confusion.png                  # Best NN confusion matrix
│   ├── nn_results.txt                         # NN performance metrics
│   └── best_model.h5                          # Saved best neural network model
```

## Methodology

1. **Data Loading & Preprocessing**
   - Load CSV data without headers
   - Separate labels (column 0) from features (columns 1-784)
   - Apply MinMax normalization
   - Reshape for visualization (28x28)

2. **Data Analysis**
   - Analyze class distribution
   - Visualize sample images
   - Ensure balanced dataset

3. **Model Training**
   - Train multiple models in parallel
   - Apply appropriate hyperparameters
   - Use validation sets for optimization

4. **Evaluation**
   - Generate confusion matrices
   - Calculate F1 scores (macro and weighted)
   - Compare model performances
   - Select best performing model

5. **Testing**
   - Generate synthetic test letters
   - Validate model on custom inputs
   - Calculate accuracy on specific letters

## Output Files

### Visualizations
- `class_distribution.png`: Distribution of samples across all 26 classes
- `sample_images.png`: 25 sample handwritten letters with labels
- `svm_confusion_matrices.png`: Side-by-side confusion matrices for Linear and RBF SVM
- `logistic_regression_curves.png`: Training and validation loss/accuracy curves
- `logistic_regression_confusion.png`: Confusion matrix for Logistic Regression
- `neural_network_curves.png`: Learning curves for both NN architectures
- `best_nn_confusion.png`: Confusion matrix for the best-performing neural network

### Results Files
- `svm_results.txt`: SVM model F1 scores
- `logistic_regression_results.txt`: Logistic Regression F1 scores
- `nn_results.txt`: Neural Network F1 scores and best model identifier
- `best_model.h5`: Saved Keras model (best performing NN)

## Contributers:
- Omer Tawfig
- Ahmed Yousif Alzain
- Izzaldin Salah