# ATS-Submitter-Prediction
This repository contains Python code for multi-label text classification using the One-vs-Rest (OvR) strategy. The code uses the scikit-learn library for machine learning tasks and demonstrates a complete workflow for preprocessing, training, and evaluating a multi-label text classification model. In this README, we will provide a detailed explanation of the code and its various components.

Table of Contents
Introduction
Requirements
Getting Started
Code Explanation
1. Loading and Preprocessing Data
2. MultiLabelBinarizer
3. Text Preprocessing with TF-IDF
4. Building the Model Pipeline
5. Training and Evaluation
Usage
References
License
Introduction
Multi-label text classification is a task where each document can belong to multiple categories simultaneously. The One-vs-Rest (OvR) strategy is a common approach for handling multi-label classification problems. In this strategy, a separate binary classifier is trained for each label, treating it as a positive class while treating all other labels as negative. This code demonstrates how to implement OvR for multi-label text classification using scikit-learn.

Requirements
Before running the code, make sure you have the following libraries installed:

pandas
scikit-learn
matplotlib
joblib
You can install these libraries using pip:

bash
Copy code
pip install pandas scikit-learn matplotlib joblib
Getting Started
Clone this repository:
bash
Copy code
git clone https://github.com/Parsayarya/ATS-Submitter-Prediction.git
cd ovr-multi-label-text-classification
Ensure you have the required dataset (e.g., 'FinalCorpus4.csv') available in the 'Data' folder. That is the information and working papers from the ATS archive.

Run the code using a Python environment (Python 3.6 or higher):

bash
Copy code
python ovr_classification.py
Code Explanation
The code is organized into several sections, each serving a specific purpose:

1. Loading and Preprocessing Data
The data is loaded from a CSV file ('FinalCorpus4.csv') using pandas. We filter and preprocess the data to select relevant labels and handle the 'Submitted By' column to make it suitable for multi-label classification.

2. MultiLabelBinarizer
We use the MultiLabelBinarizer to transform the multi-label classification problem into a binary classification problem for each label. This step is essential for OvR.

3. Text Preprocessing with TF-IDF
We preprocess the text data using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. This technique converts text data into numerical features suitable for machine learning models.

4. Building the Model Pipeline
We create a scikit-learn pipeline that includes a TF-IDF vectorizer and a RandomForestClassifier as the base classifier for each label. The OneVsRestClassifier is not used in this version.

5. Training and Evaluation
The model is trained on the preprocessed data. Evaluation metrics such as ROC curves, confusion matrices, and recall scores are calculated to assess the model's performance.

Usage
To use this code for your multi-label text classification task:

Prepare your dataset in a CSV format similar to 'FinalCorpus4.csv'.
Ensure that your dataset includes text data and corresponding labels.
Modify the code's configuration (e.g., threshold values, model hyperparameters) to suit your specific task.
Run the code using the instructions provided in the "Getting Started" section.
Feel free to customize and extend the code as needed for your project.

