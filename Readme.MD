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
| Mini-Project | ⏳ Pending   | Planned for later in the semester                         |

---


