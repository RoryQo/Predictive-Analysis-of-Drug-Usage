# Predictive Analysis of Drug Usage Mini Project   
 
## Table of Contents
1. [Overview](#overview)
2. [Data Preparation](#data-preparation)
3. [Modeling Techniques](#modeling-techniques)
4. [Model Assessment](#model-assessment)  
5. [Conclusion](#conclusion)

## Overview
In this project, we utilize the NHANES dataset to predict hard drug usage based on various health and demographic features. The project involves data cleaning, model training using supervised learning algorithms, and evaluating the performance of these models.

## Data Preparation
The NHANES dataset is processed to select relevant features such as age, gender, race, education, and health indicators. Missing values are handled to ensure a clean dataset suitable for model training. The data is split into training (80%) and testing (20%) sets for effective model evaluation.

## Modeling Techniques
Several supervised learning algorithms are applied:
- **Decision Trees:** A tree-based model that makes predictions based on feature splits.
- **Random Forests:** An ensemble method using multiple decision trees to improve accuracy.
- **Naive Bayes:** A probabilistic classifier based on Bayes' theorem.

Each model is trained on the training dataset, and predictions are made on the testing set.

## Model Assessment
The performance of the models is assessed using various metrics:
- **Misclassification Rate:** Calculated to evaluate the percentage of incorrect predictions.
- **Confusion Matrix:** Provides a detailed breakdown of true positives, true negatives, false positives, and false negatives.
```
# Create function to make a confusion matrix 
# create new columns in data with predictions from model
# select actual target values (y) and predicted y values
# Table them
confusion_matrix <- function(data,y,mod){
  confusion_matrix <- data %>% 
  mutate(pred = predict(mod, newdata = data, type = "class"),
         y=y) %>%
  select(y,pred) %>% table()
}
```

Hyperparameter tuning is performed on the random forest model to optimize performance, ensuring the best model is selected based on accuracy and error rates.

## Conclusion
This project demonstrates the effectiveness of supervised learning techniques in predicting drug usage based on health and demographic data. The analysis highlights the importance of model selection and evaluation, providing insights into the best-performing algorithms for classification tasks.
