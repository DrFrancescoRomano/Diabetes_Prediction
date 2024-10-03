# Diabetes_Prediction

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

## Project Phases

### 1. **Loading Libraries**
At the beginning of the project, we load the necessary libraries for data analysis, visualization, and model building. The main libraries include:
- **dplyr**: for data manipulation.
- **caret**: for model training and evaluation.
- **ggplot2**: for data visualization.
- **pROC**: for ROC curve analysis.

### 2. **Loading the Dataset**
The dataset is loaded from a CSV file. During this phase, preliminary checks on the dataset structure, including dimensions and variables present, are performed. This initial information helps us understand the dataset's content and plan subsequent analysis steps.

### 3. **Initial Data Exploration**
An analysis of the dataset's structure is performed using functions such as `str()` and `summary()`. These functions provide information on variable types, their distribution, and descriptive statistics, allowing us to identify potential anomalies and outliers.

### 4. **Checking for Missing Values**
The presence of missing values in each column of the dataset is checked. The absence of null values is crucial, as incomplete data can negatively impact the effectiveness of the predictive model. If missing values are present, strategies for managing them, such as imputation or row removal, are evaluated.

### 5. **Removing Duplicates**
We proceed to remove any duplicate rows in the dataset to ensure the uniqueness of observations. This operation is performed using the `duplicated()` function, which identifies repeated rows. By removing duplicates, we reduce the risk of introducing bias into the model, ensuring that each sample contributes uniquely to the analysis.

### 6. **Filtering Data**
In the specific case of our dataset, it was decided to exclude rows where the gender is classified as "Other." This operation is carried out using a filter to ensure that the dataset contains only significant and useful categories for analysis.

### 7. **Univariate Analysis**
In this phase, individual variables are analyzed through the creation of charts and descriptive statistics. 

#### 1. Age Distribution Histogram
![Age Distribution Histogram](https://github.com/DrFrancescoRomano/Diabetes_Prediction/blob/main/Plot_Diabetes_Rplot/Rplot_Age.png)

The age distribution histogram displays how the ages of patients are distributed within the dataset. The bars, grouped in 5-year intervals, provide a clear representation of the frequency of each age range. This graph helps identify any peaks or anomalies in the age distribution.

#### 2. Gender Distribution
![Gender Distribution](https://github.com/DrFrancescoRomano/Diabetes_Prediction/blob/main/Plot_Diabetes_Rplot/Rplot_Gender_Distribution.png)

The bar chart for gender distribution visualizes the number of patients divided by gender. The colored bars allow for a quick observation of the sample composition, highlighting any imbalances between genders. This graph is useful for understanding the representation of each gender in the dataset.

#### 3. BMI Distribution Histogram and Density

![BMI Distribution Histogram and Density]([insert_image_path_here](https://github.com/DrFrancescoRomano/Diabetes_Prediction/blob/main/Plot_Diabetes_Rplot/Rplot_BMI_Distribution.png)
This graph combines a histogram and a density curve to show the distribution of BMI. The histogram represents the normalized BMI values, while the overlaid density curve provides an overall view of the distribution. The combination of these two elements allows for better identification of trends and shapes in the distribution of BMI values.

#### 4. Smoking History Distribution
![Smoking History Distribution](https://github.com/DrFrancescoRomano/Diabetes_Prediction/blob/main/Plot_Diabetes_Rplot/Rplot_Smoking_History_Distribution.png)

The bar chart for smoking history distribution shows the number of patients categorized by their smoking habits. The use of different colors for each smoking category makes the graph visually appealing and facilitates data interpretation. The rotated labels on the x-axis enhance readability, especially when there are multiple categories.

#### 5. Hypertension Distribution
![Hypertension Distribution]([insert_image_path_here](https://github.com/DrFrancescoRomano/Diabetes_Prediction/blob/main/Plot_Diabetes_Rplot/Rplot_Hypertension_Distribution.png)

The bar chart for hypertension distribution visualizes how many patients have or do not have hypertension. This graph provides useful information about the prevalence of hypertension in the sample, highlighting the proportion of affected patients compared to those who are not affected.

#### 6. Heart Disease Distribution
![Heart Disease Distribution]([insert_image_path_here](https://github.com/DrFrancescoRomano/Diabetes_Prediction/blob/main/Plot_Diabetes_Rplot/Rplot_Heart_Disease_Distribution.png)

The graph for heart disease distribution shows how many patients have a history of heart disease compared to those who do not. This visualization helps understand the incidence of heart disease in the sample, highlighting patients at risk.

#### 7. Diabetes Distribution
![Diabetes Distribution](https://github.com/DrFrancescoRomano/Diabetes_Prediction/blob/main/Plot_Diabetes_Rplot/Rplot_Diabetes_Distribution.png)

The bar chart for diabetes distribution represents the number of patients diagnosed with diabetes compared to those not diagnosed. This graph provides a clear view of the prevalence of diabetes in the dataset and offers valuable insights for further analysis.


#### 8. **Multivariate Analysis**
After understanding individual variables, we analyze the interactions between them. 
- Graphs are created to show how the target variable (diabetes) relates to other variables, such as gender and age. This helps identify significant correlations and potential indicators for diabetes prediction.

##### 1. Gender vs Diabetes
![Gender vs Diabetes]([insert_image_path_here](https://github.com/DrFrancescoRomano/Diabetes_Prediction/blob/main/Plot_Diabetes_Rplot/Rplot_GendervsDiabetes.png)

This bar chart visualizes the relationship between gender and diabetes diagnosis. The bars are positioned side by side (dodge) to compare the counts of diabetes cases (0 for non-diabetic and 1 for diabetic) across different genders. The use of blue and orange colors helps differentiate between the two classes. This visualization provides insight into how diabetes prevalence varies between genders in the dataset.

##### 2. Age vs BMI Colored by Diabetes Classification

![Age vs BMI Colored by Diabetes Classification]([insert_image_path_here](https://github.com/DrFrancescoRomano/Diabetes_Prediction/blob/main/Plot_Diabetes_Rplot/Rplot_AgevsBMI.png)
This scatter plot depicts the relationship between age and BMI, with points colored according to diabetes classification (0 for non-diabetic and 1 for diabetic). Each point represents an individual patient, allowing for the identification of patterns or clusters in the data. The transparency (alpha = 0.6) and size of the points enhance visibility, especially in areas of high density. This graph helps to visualize how age and BMI might correlate with diabetes status, providing valuable insights for analysis.


#### 9. **Data Preprocessing**
This phase is crucial to prepare the data for the predictive model.
- **Recategorization of Smoking History**: the "smoking_history" variable is recategorized into three classes: "non.smoker", "current", and "past.smoker". This is important for simplifying categories and improving data understanding.
- **One-Hot Encoding**: one-hot encoding is applied to categorical variables (such as gender and smoking history), transforming them into numerical variables. This makes the data suitable for machine learning models.

#### 10. **Correlation Analysis**
The correlation matrix for numerical variables in the dataset is calculated. 
- A heatmap is created to visualize the correlations between variables, helping to identify which variables have a significant relationship with the target variable (diabetes). Variables with high correlations are considered more relevant for prediction.

### 11. **Data Balancing**
The original dataset shows an imbalance between the classes of the target variable (diabetes positive and negative).
- The SMOTE (Synthetic Minority Over-sampling Technique) method is used to balance the dataset. This technique generates new synthetic samples for the minority class (diabetes positive), thereby reducing the impact of the imbalance. 

| Class Distribution Before SMOTE |
|----------------------------------|
|    0     |     1     |
|----------|-----------|
| 87646    |  8482     |

| Class Distribution After SMOTE |
|--------------------------------|
|    0     |     1     |
|----------|-----------|
| 87646    |  25446    |

- Before applying SMOTE, the distribution of classes is checked, and it is ensured that the target variable is of factor type. After applying SMOTE, the new distribution of classes is confirmed to verify that the balancing was effective.

### 12. **Feature Scaling**
To ensure that all numerical variables are on a similar scale, standardization is applied to selected columns (BMI, HbA1c_level, blood_glucose_level, and age). 
- Standardization is important because many machine learning algorithms, such as logistic regression and neural networks, are sensitive to the scale of variables.

### 13. **Splitting Data into Training and Testing Sets**
The balanced dataset is split into a training set and a testing set. 
- A stratified partitioning method is used to ensure that the proportion of classes is maintained in both sets. This helps evaluate model performance on data that reflects the same proportions present in the original dataset.

### 14. **Model Building and Optimization**
Several machine learning models are constructed and optimized:
- **Random Forest**: parameter optimization (mtry and maxnodes) is performed using cross-validation. The number of trees is also limited to improve efficiency.
- **Support Vector Machine (SVM)**: different combinations of parameters (C and sigma) are explored to optimize the model.
- **Logistic Regression**: a logistic regression model is trained for binary classification.

### 15. **Model Performance Evaluation**
After training the models, their performance is evaluated using the following metrics:
- **Accuracy**: the percentage of correct predictions out of total predictions.
- **Precision**: the ability of the model to correctly identify positives among all predicted positive results.
- **Recall (Sensitivity)**: the ability of the model to identify true positives among all true positives.
- **F1 Score**: a measure of the balance between precision and recall.

### 16. **Ensemble Models**
To further improve predictive performance, an ensemble model is created that combines the predictions from the previous models using a majority vote. 
- This approach can leverage the strengths of each model, enhancing overall prediction accuracy.

### 17. **Final Results**
At the end of the project, a detailed summary of the performance of each model and the ensemble model is presented. 

### Combined Performance Metrics (Train & Test)
| Model                  | Accuracy | Precision | Recall | F1_Score | Train_Accuracy | Train_Precision | Train_Recall | Train_F1_Score |
|------------------------|----------|-----------|--------|----------|-----------------|------------------|--------------|-----------------|
| Random Forest          | 0.9373434| 0.9274924 | 0.9967532| 0.9608764| 0.9562773       | 0.9463190        | 1.0000000    | 0.9724192       |
| SVM                    | 0.9197995| 0.9156627 | 0.9870130| 0.9500000| 0.9569019       | 0.9505027        | 0.9959481    | 0.9726949       |
| Logistic Regression     | 0.9122807| 0.9148936 | 0.9772727| 0.9450549| 0.9163023       | 0.9303599        | 0.9635332    | 0.9466561       |

### Performance Metrics of the Ensemble Model
| Model          | Accuracy | Precision | Recall | F1_Score |
|----------------|----------|-----------|--------|----------|
| Random Forest  | 0.9373434| 0.9274924 | 0.9967532| 0.9608764|
| SVM            | 0.9197995| 0.9156627 | 0.9870130| 0.9500000|
| Logistic Regression | 0.9122807| 0.9148936 | 0.9772727| 0.9450549|
| **Ensemble**   | **0.9248120** | **0.9186747** | **0.9902597** | **0.9531250** |

## Conclusion
This data science project has allowed us to develop a predictive model for diabetes diagnosis, following a rigorous and systematic approach. Through data manipulation and analysis, we were able to build machine learning models that can support clinicians in early diagnosis of the disease, thus improving patient intervention and treatment possibilities.


## Contact

If you have any questions or suggestions, feel free to contact me:

- **Name**: Francesco Romano
- **LinkedIn**: [Francesco Romano](https://www.linkedin.com/in/francescoromano03/)
