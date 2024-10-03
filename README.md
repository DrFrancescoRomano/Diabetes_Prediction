# Diabetes_Prediction

# Diabetes Prediction Project

## Overview

The **Diabetes Prediction Project** aims to develop a machine learning model to predict the likelihood of diabetes in patients based on their medical and demographic data. By leveraging various classification techniques, this project intends to provide valuable insights for healthcare professionals in identifying at-risk individuals and formulating personalized treatment plans.

## About the Dataset

The **Diabetes Prediction Dataset** is a collection of medical and demographic information from 100,000 patients, along with their diabetes status (positive or negative). The dataset includes the following features:

- **Age**: Age of the patient (ranging from 0 to 80 years).
- **Gender**: Gender of the patient (categories: Male, Female, Other).
- **Hypertension**: Indicator for hypertension status (0 = No, 1 = Yes).
- **Heart Disease**: Indicator for heart disease status (0 = No, 1 = Yes).
- **Smoking History**: Categorized smoking status (Not Current, Former, No Info, Current, Never).
- **BMI**: Body Mass Index, a measure of body fat based on weight and height.
- **HbA1c Level**: A measure of average blood sugar levels over the past 2-3 months.
- **Blood Glucose Level**: Current blood glucose measurement.
- **Diabetes**: Target variable indicating the presence of diabetes (1 = Positive, 0 = Negative).

This dataset is instrumental in building predictive models for diabetes, aiding in early diagnosis and intervention.

## Project Objectives

1. **Data Exploration and Preprocessing**:
   - Load and explore the dataset to understand its structure and features.
   - Perform data cleaning, including handling missing values and duplicates.
   - Conduct univariate and multivariate analysis to uncover relationships between features.

2. **Data Balancing**:
   - Utilize techniques such as SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance in the dataset.

3. **Model Development and Evaluation**:
   - Implement multiple classification algorithms: Random Forest, Support Vector Machine (SVM), and Logistic Regression.
   - Optimize models using cross-validation and hyperparameter tuning.
   - Evaluate model performance using accuracy, precision, recall, and F1-score.

The required packages are as follows:
```r
# Load required libraries
library(dplyr)
library(caret)
library(readr)
library(yardstick)
library(pROC)
library(corrplot)
library(Hmisc)
library(boot)
library(zoo)
library(ggplot2)
library(lubridate)
library(xgboost)
library(glmnet)
library(nnet)
library(scales)
library(plotly)
library(ggthemes)
library(GGally)
library(e1071)
library(smotefamily)
```


The R code is located in the repository and is divided into the following sections:

1. **Data Loading & Exploration**
   - 1.1. Data exploration  
   - 1.2. Remove duplicate entries and filter out 'Other' from gender

2. **Univariate Analysis**
   - 2.1. Visualization code for univariate analysis

3. **Multivariate Analysis**
   - 3.1. Visualization code for multivariate analysis

4. **Data Preprocessing**
   - 4.1. Recode and encode categorical variables  
   - 4.2. One-Hot Encoding for categorical variables

5. **Data Balancing (SMOTE)**
   - 5.1. Apply SMOTE for data balancing

6. **Model Training**
   - 6.1. Split the data into training and testing sets and train models

