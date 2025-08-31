# Machine Learning Lab 4: Banknote Authentication Dataset Analysis

## Overview
This lab implements a complete machine learning pipeline for the Banknote Authentication Dataset, demonstrating various aspects of model selection, evaluation, and ensemble methods.

## Files in this Directory
- `ML_Lab4_1.ipynb`: Jupyter notebook containing the first part of the lab assignment
- `ML_Lab4_2.ipynb`: Jupyter notebook implementing a comprehensive ML pipeline including:
  - Manual Grid Search Implementation
  - Built-in GridSearchCV Implementation
  - Model Comparison and Evaluation
  - Ensemble Methods (Voting Classifier)

## Dataset Description
The Banknote Authentication dataset contains features extracted from images of genuine and forged banknotes. The task is to classify banknotes as authentic or forged.

### Features
1. Variance of Wavelet Transformed image
2. Skewness of Wavelet Transformed image
3. Curtosis of Wavelet Transformed image
4. Entropy of image
5. Class (0: Authentic, 1: Forged)

## Implementation Details

### Models Implemented
1. Decision Tree Classifier
2. k-Nearest Neighbors (k-NN)
3. Logistic Regression
4. Voting Classifier (Ensemble)

### Key Components
1. **Data Preprocessing**
   - Feature scaling
   - Feature selection
   - Train-test split

2. **Model Selection**
   - Manual Grid Search implementation
   - Built-in GridSearchCV implementation
   - Cross-validation

3. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - ROC curves and AUC scores
   - Confusion matrices

4. **Visualizations**
   - ROC curves for all models
   - Confusion matrices for voting classifiers
   - Performance comparison plots

## Results
The notebooks contain detailed comparisons between:
- Manual vs Built-in Grid Search implementations
- Individual model performances
- Ensemble method effectiveness
- Feature importance analysis

## Requirements
- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib
- jupyter notebook

## Usage
1. Open the Jupyter notebooks in VS Code or Jupyter Lab
2. Run all cells in sequence
3. Compare the results between manual and built-in implementations
4. Analyze the visualizations and performance metrics

## Note
Make sure to have all required libraries installed before running the notebooks. The implementation includes proper error handling and comprehensive documentation within the code.
