# Machine Learning Projects

## Table of Contents
1. [Overview](#overview)
2. [Part 1: Supervised Machine Learning](#part-1-supervised-machine-learning)
   - [Data Preparation](#data-preparation)
   - [Modeling Techniques](#modeling-techniques)
   - [Model Assessment](#model-assessment)
3. [Part 2: Unsupervised Machine Learning](#part-2-unsupervised-machine-learning)
   - [Data Loading](#data-loading)
   - [Principal Component Analysis](#principal-component-analysis)
   - [Clustering](#clustering)
   - [Cluster Analysis](#cluster-analysis)
4. [Conclusion](#conclusion)

## Overview
This repository contains two projects focusing on different machine learning techniques: supervised and unsupervised learning. The first project applies various supervised learning algorithms to predict drug usage based on health and demographic data. The second project utilizes principal component analysis and clustering methods to analyze country-level data, exploring relationships between economic and health indicators.

## Part 1: Supervised Machine Learning

### Data Preparation
In this project, the NHANES dataset is utilized to train models predicting hard drug usage. Data is cleaned and prepared, ensuring that missing values are handled appropriately and features are selected for model training.

### Modeling Techniques
Multiple algorithms are employed, including decision trees, random forests, and naive Bayes. The models are trained and evaluated using various metrics, including misclassification rates and confusion matrices. Hyperparameter tuning is performed for the random forest model to optimize performance.

### Model Assessment
The models are assessed based on their classification performance, and metrics such as sensitivity and specificity are calculated. The analysis highlights the trade-offs between model complexity and accuracy, providing insights into the best-performing models.

## Part 2: Unsupervised Machine Learning

### Data Loading
This project involves the analysis of country-level data to uncover patterns through unsupervised learning techniques. The dataset is loaded, and initial exploratory data analysis is performed.

### Principal Component Analysis
Principal component analysis (PCA) is conducted to reduce dimensionality and identify key components that explain the most variance in the data. A scree plot is created to visualize the proportion of variance explained by each principal component, and cumulative variance is assessed to determine the number of components to retain.

### Clustering
Using the first few principal components, k-means clustering and Partitioning Around Medoids (PAM) are applied to identify clusters within the data. The optimal number of clusters is determined through methods such as the gap statistic, and differences between clustering methods are evaluated.

### Cluster Analysis
The analysis of clusters focuses on economic indicators like GDP per capita and health metrics such as life expectancy. Boxplots are used to visualize differences across clusters, and median life expectancy is calculated for each cluster, revealing significant disparities.

## Conclusion
These projects demonstrate the application of both supervised and unsupervised machine learning techniques to real-world datasets. Through careful data preparation, model selection, and evaluation, valuable insights are derived that contribute to understanding complex relationships within the data. This comprehensive approach showcases the strengths of machine learning in predictive analytics and exploratory data analysis.
