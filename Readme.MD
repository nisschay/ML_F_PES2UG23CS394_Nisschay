---

# 📊 ML-Lab

This repository contains my **Machine Learning Laboratory coursework**, including weekly assignments and future mini-projects.
It serves as a personal record of my work, experiments, and learning progress throughout the course.

---

## 📂 Repository Structure

```
ML-Lab/
├── ML_LAB1_Assignment/
│   ├── ML_LAB_1.pdf                  # Week 1 lab instructions
│   ├── ML_Lab_Assignment_1_1.ipynb   # Notebook for Assignment 1.1
│   ├── ML_Lab_Assignment_1_2.ipynb   # Notebook for Assignment 1.2
│   ├── customer_churn_data.csv       # Dataset for classification task
│   └── house_price_data.csv          # Dataset for regression task
├── ML_LAB3_Assignment/
│   └── EC_F_PES2UG23CS394_Lab3.py    # Information Gain & Decision Tree implementation
├── ML_LAB4_Assignment/
│   ├── ML_LAB4.pdf                   # Week 4 lab instructions
│   ├── ML_Lab4_1.ipynb               # Wine Quality Classification
│   └── ML_Lab4_2.ipynb               # Banknote Authentication Classification
├── ML_LAB6_Assignment/
│   ├── ML_LAB_6.pdf                  # Week 6 lab report
│   └── ML_LAB_6.ipynb                # Neural Networks implementation
├── Ml_LAB7_Assignment/
│   ├── PES2UG23CS394_F.ipynb.ipynb   # SVM kernels, margin experiments
│   └── PES2UG23CS394_F_Report.pdf.docx # Week 7 report draft
├── ML_LAB12_Assignment/
│   ├── Naive_Bayes_Classifier_Boilerplate.ipynb # Naive Bayes implementation
│   ├── Lab Report.pdf                # Week 12 lab report
│   ├── Lab Instructions.pdf          # Week 12 lab instructions
│   └── Dataset/                      # PubMed 20k RCT dataset
├── ML_LAB13_Assignment/
│   ├── PES2UG23CS394_F.ipynb         # K-means clustering implementation
│   ├── PES2UG23CS394_F_Report.pdf    # Week 13 lab report
│   ├── ML Lab Week 13 Clustering Lab Instructions.pdf # Lab instructions
│   └── bank-full.csv                 # Bank marketing dataset
├── ML_LAB14_Assignment/
│   ├── PES2UG23CS394_LAB14.ipynb     # CNN for image classification
│   ├── PES2UG23CS394_LAB14.pdf       # Week 14 lab report
│   └── Week_14_Lab_Instructions.pdf  # Lab instructions
└── (Upcoming assignments & projects)
```

---

## 📌 Contents

### **Week 1**

* **Topics Covered**:

  * Data loading and exploration with Pandas
  * Basic data preprocessing
  * Exploratory Data Analysis (EDA)
  * Building simple ML models for classification & regression
* **Datasets**:

  * *Customer Churn Data* – for binary classification
  * *House Price Data* – for regression analysis

### **Week 2**

* **Status**: ❌ No assignment (missed week)

### **Week 3**

* **Topics Covered**:

  * Information Theory fundamentals
  * Entropy calculation for datasets
  * Information Gain computation
  * Decision Tree attribute selection algorithms
  * PyTorch tensor operations for ML computations
* **Implementation**:

  * `get_entropy_of_dataset()` - Calculate dataset entropy using Shannon's formula
  * `get_avg_info_of_attribute()` - Compute weighted average information for attribute splits
  * `get_information_gain()` - Calculate information gain for feature selection
  * `get_selected_attribute()` - Select best attribute based on highest information gain
* **Datasets Tested**:

  * *Mushrooms Dataset* – mushroom classification
  * *Nursery Dataset* – nursery school ranking
  * *Tic-Tac-Toe Dataset* – game outcome prediction

### **Week 4**

* **Topics Covered**:

  * Classification algorithms implementation and comparison
  * Manual vs Built-in classifier implementations
  * Hyperparameter tuning techniques
  * Model performance evaluation and analysis
* **Algorithms Implemented**:

  * **K-Nearest Neighbors (KNN)** - Both manual and built-in versions
  * **Logistic Regression** - Both manual and built-in implementations
  * **Decision Tree** - Both manual and built-in classifiers
  * **Voting Classifier** - Ensemble method combining multiple algorithms
* **Hyperparameter Optimization**:

  * **GridSearchCV** - Automated hyperparameter search
  * **Manual Search** - Custom hyperparameter tuning implementation
* **Datasets**:

  * *Wine Quality Dataset* – Multi-class classification of wine quality
  * *Banknote Authentication Dataset* – Binary classification for fraud detection
* **Performance Metrics**:

  * Accuracy, Precision, Recall, F1-Score
  * ROC-AUC for model comparison
  * Comprehensive performance analysis tables

### **Week 5**

* **Status**: ❌ No assignment (missed week)

### **Week 6**

* **Topics Covered**:

  * Neural Networks for Function Approximation
  * Implementation of forward propagation, backpropagation, and gradient descent
  * Weight initialization using Xavier initialization
  * Early stopping to prevent overfitting
  * Visualization of training loss curves and predicted vs actual values
* **Deliverables**:

  * `ML_LAB_6.ipynb` – Complete notebook with ANN implementation, experiments, and plots
  * `ML_LAB_6.pdf` – Lab report summarizing dataset, methodology, results, and conclusions
* **Experiments**:

  * Baseline model training
  * Hyperparameter exploration (learning rate, epochs, batch size)
  * Comparative results table with training/test losses and R² scores

### **Week 7**

* **Topics Covered**:

  * Support Vector Machines (SVM) with linear, RBF, and polynomial kernels
  * Comparative evaluation across synthetic and real-world datasets
  * Margin control via hard (high `C`) and soft (low `C`) configurations
  * Visual diagnostics of separating hyperplanes in two dimensions
* **Datasets**:

  * *Moons Dataset* – Non-linear synthetic classification for kernel comparison
  * *Banknote Authentication* – Real-world binary classification using variance and skewness
  * *Make Blobs (with noise)* – Linearly separable data to contrast hard vs soft margins
* **Deliverables**:

  * `PES2UG23CS394_F.ipynb.ipynb` – End-to-end SVM experiments, metrics, and plots
  * `PES2UG23CS394_F_Report.pdf` – Draft write-up summarizing observations and takeaways

### **Week 12**

* **Topics Covered**:

  * Naive Bayes classification from scratch
  * Count/Frequency-based Multinomial Naive Bayes implementation
  * Text classification using TF-IDF vectorization
  * Hyperparameter tuning with GridSearchCV
  * Comparison of custom vs. scikit-learn implementations
* **Dataset**:

  * *PubMed 20k RCT* – Medical abstract classification into categories (BACKGROUND, OBJECTIVE, METHODS, RESULTS, CONCLUSIONS)
* **Deliverables**:

  * `Naive_Bayes_Classifier_Boilerplate.ipynb` – Custom Naive Bayes implementation and scikit-learn pipeline
  * `Lab Report.pdf` – Comprehensive analysis of model performance and comparisons

### **Week 13**

* **Topics Covered**:

  * K-means clustering algorithm from scratch
  * Customer segmentation and market analysis
  * Principal Component Analysis (PCA) for dimensionality reduction
  * Elbow method for optimal cluster selection
  * Agglomerative hierarchical clustering
  * Dendrogram visualization for hierarchical clustering
  * Clustering evaluation metrics (Silhouette Score, Inertia)
* **Dataset**:

  * *Bank Marketing Dataset* – Customer data for segmentation analysis (age, balance, job, education, etc.)
* **Deliverables**:

  * `PES2UG23CS394_F.ipynb` – K-means implementation from scratch, PCA, and hierarchical clustering
  * `PES2UG23CS394_F_Report.pdf` – Analysis of customer segments and clustering evaluation

### **Week 14**

* **Topics Covered**:

  * Convolutional Neural Networks (CNN) architecture
  * Image classification using deep learning
  * PyTorch implementation of CNNs
  * Convolutional layers, pooling layers, and fully connected layers
  * Data augmentation and image preprocessing
  * Model training with GPU acceleration
  * Performance evaluation on image datasets
* **Dataset**:

  * *Rock Paper Scissors Dataset* – Hand gesture images for multi-class classification (rock, paper, scissors)
* **Deliverables**:

  * `PES2UG23CS394_LAB14.ipynb` – Complete CNN implementation using PyTorch for image classification
  * `PES2UG23CS394_LAB14.pdf` – Lab report with model architecture, training results, and analysis

### **Future Work**

* Additional weekly lab assignments
* One or more **mini-projects**, applying ML techniques to real-world datasets

---

## 📅 Timeline

| Week         | Status      | Notes                                                     |
| ------------ | ----------- | --------------------------------------------------------- |
| 1            | ✅ Completed | Introductory ML tasks                                     |
| 2            | ❌ Missed    | No assignment                                             |
| 3            | ✅ Completed | Information Gain & Decision Tree theory                   |
| 4            | ✅ Completed | Classification algorithms & hyperparameter tuning         |
| 5            | ❌ Missed    | No assignment                                             |
| 6            | ✅ Completed | Neural Networks for Function Approximation (from scratch) |
| 7            | ✅ Completed | SVM kernels, hard vs soft margin analysis                 |
| 12           | ✅ Completed | Naive Bayes classifier from scratch, text classification  |
| 13           | ✅ Completed | K-means clustering from scratch, customer segmentation    |
| 14           | ✅ Completed | CNN image classification with PyTorch                     |
| Mini-Project | ⏳ Pending   | Planned for later in the semester                         |

---


