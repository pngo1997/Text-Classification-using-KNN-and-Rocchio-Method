# 🏗️ Text Classification using K-Nearest Neighbors and Rocchio Method

## 📜 Overview  
This project implements a **K-Nearest Neighbor (KNN) classifier** and **Rocchio Method** to classify text documents into two categories:  
1. **Hockey (class label = 1)**  
2. **Microsoft Windows (class label = 0)**  

The dataset is a subset of the **20 Newsgroups dataset**, containing **1,000 documents** represented in a **term x document** format with **5,500 vocabulary terms**. The dataset has already been preprocessed (tokenized, stop words removed, and stemmed). Both training and test data are provided, with documents represented by **raw term frequency (TF) values**.

## 🎯 Problem Explanation
The goal is to classify documents using **custom implementations** of KNN and Rocchio classifiers, without relying on scikit-learn for classification. Tasks include:  

1. **Implementing a custom KNN classifier** (with both **Euclidean distance** and **Cosine similarity**).  
2. **Evaluating classification accuracy** of the KNN model across different values of K (5 to 100).  
3. **Visualizing accuracy trends** across K values for both **distance metrics**.  
4. **Applying TF-IDF weighting** instead of raw term frequencies and re-evaluating the classifier.  
5. **Implementing the Rocchio Method** (nearest centroid classifier) and comparing it to KNN.  
6. **Using Scikit-learn's Nearest Centroid classifier** and benchmarking results.  

## 🛠️ Implementation Details  
### **1. Custom KNN Classifier**  
- Implements two versions:  
  - **Euclidean Distance:** Measures similarity based on squared differences.  
  - **Cosine Similarity:** Uses `1 - Cosine similarity` as the distance metric.  
- The classifier takes as input:  
  - **Training data matrix**  
  - **Training labels**  
  - **Test instance to classify**  
  - **Value of K (number of neighbors)**  
  - **Distance metric (Euclidean or Cosine similarity)**  
- Returns:  
  - **Predicted class label**  
  - **Indices of top K nearest neighbors**  

### **2. Model Evaluation**  
- A custom **evaluation function** computes classification accuracy.  
- Accuracy is tested for **K values from 5 to 100 (increments of 5)**.  
- Results are plotted for **Euclidean vs. Cosine similarity KNN classifiers**.  

### **3. Applying TF-IDF Transformation**  
- Converts raw term frequencies into **TF-IDF weights**.  
- The classifier is re-evaluated using **only the Cosine similarity version**.  
- Results are compared **with and without TF-IDF weighting**.  

### **4. Rocchio Method (Nearest Centroid Classifier)**  
- **Training function:**  
  - Computes a **prototype vector** for each class using mean TF-IDF values.  
- **Classification function:**  
  - Measures **Cosine similarity** between test instances and class prototypes.  
  - Assigns the class with the highest similarity score.  
- **Evaluation:**  
  - Accuracy is compared against **best-performing KNN model (with TF-IDF weights)**.  

### **5. Scikit-learn’s Nearest Centroid Classifier**  
- Uses `NearestCentroid()` from scikit-learn.  
- Results are compared with the **custom Rocchio classifier**.  

## 🚀 Technologies Used  
- **Python** (for implementing classifiers and evaluating performance).  
- **Pandas & NumPy** (for data manipulation and matrix operations).  
- **Matplotlib** (for visualization of accuracy trends).  
- **Scikit-learn** (only for comparison with built-in Nearest Centroid classifier).  

## 📊 20 Newsgroups dataset
The dataset is a subset of the 20 newsgroup corpus http://qwone.com/~jason/20Newsgroups/  in term-document format. This subset has been taken from http://mlg.ucd.ie/content/view/22/ (this data was modified to remove terms that did not appear in any of the documents). Each document belong to one of the two classes {Windows, Hockey}. The original data has been divided into test and train (20%, 80%) subsets.

The files contained in the archive file are as follows:
1. **trainMatrixModified.txt**: the term-document frequency matrix for the training documents. Each row of this matrix corresponds to one the terms and each column corresponds to one the documents and the (i,j)th element of the matrix shows the frequency of the ith term in **the jth document. This matrix contains 5500 rows and 800 columns.
2. **testMatrixModified.txt**: the term-document frequency for the test documents. The matrix contains 5500 rows and 200 columns.
3. **trainClasses.txt**: This file contains the labels associated with each training document. Each line is in the format of documentIndex \t classId where the documentIndex is in the range of [0,800) and refers to the index of the document in the term-document frequency matrix for train documents. The classId refers to one of the two classes and takes one of the values 0 (for Windows) or 1 (for Hockey).
4. **testClasses.txt**: This file contains the labels associated with each test document. Each line is in the format of documentIndex \t classId where the documentIndex is in the range of [0,200) and refers to the index of the document in the term-document frequency matrix for test documents  
5. **modifiedterms.txt**: This file contains the set of 5500 terms in the vocabulary. Each line contains a term and corresponds to the corresponding rows in term-document frequency matrices. 
