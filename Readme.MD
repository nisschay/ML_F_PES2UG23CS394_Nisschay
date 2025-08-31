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

### **Future Work**

* Additional weekly lab assignments
* One or more **mini-projects**, applying ML techniques to real-world datasets

---

## 🚀 Getting Started

### **1️⃣ Clone the repository**

```bash
git clone https://github.com/nisschay/ML-Lab.git
cd ML-Lab
```

### **2️⃣ Install dependencies**

It is recommended to use a virtual environment:

```bash
pip install -r requirements.txt
```

### **3️⃣ Open notebooks**

Launch Jupyter Notebook or JupyterLab:

```bash
jupyter notebook
```

### **4️⃣ Run Week 3 code**

For the decision tree implementation:

```python
python ML_LAB3_Assignment/EC_F_PES2UG23CS394_Lab3.py
```

### **5️⃣ Explore Week 4 assignments**

Navigate to the ML_LAB4_Assignment folder and open the Jupyter notebooks:

```bash
cd ML_LAB4_Assignment
jupyter notebook ML_Lab4_1.ipynb  # Wine Quality Analysis
jupyter notebook ML_Lab4_2.ipynb  # Banknote Authentication
```

---

## 🛠 Tech Stack

* **Python 3.10+**
* **Jupyter Notebook**
* **NumPy**, **Pandas**, **Matplotlib**, **Seaborn**
* **Scikit-Learn**
* **PyTorch** (added for Week 3)

---

## 📅 Timeline

| Week         | Status      | Notes                                    |
| ------------ | ----------- | ---------------------------------------- |
| 1            | ✅ Completed | Introductory ML tasks                    |
| 2            | ❌ Missed    | No assignment                           |
| 3            | ✅ Completed | Information Gain & Decision Tree theory  |
| 4            | ✅ Completed | Classification algorithms & hyperparameter tuning |
| 5+           | ⏳ Pending   | Will be uploaded as completed           |
| Mini-Project | ⏳ Pending   | Planned for later in the semester       |

---

## 🎯 Key Learning Outcomes

### Week 1
- Data manipulation and visualization techniques
- Basic ML model implementation and evaluation

### Week 3
- Understanding information theory in machine learning
- Implementing entropy and information gain calculations
- Attribute selection algorithms for decision trees
- Working with PyTorch tensors for ML computations

### Week 4
- Manual implementation of core classification algorithms
- Understanding the mathematics behind KNN, Logistic Regression, and Decision Trees
- Comparison between manual and built-in implementations
- Ensemble methods and voting classifiers
- Hyperparameter optimization strategies (GridSearchCV vs Manual Search)
- Comprehensive model evaluation using multiple metrics
- Performance analysis across different datasets and problem types

---

## 📜 License

This repository is intended for **academic and learning purposes only**.
Feel free to refer to the code, but **do not submit as your own work** if you are enrolled in a similar course.

---