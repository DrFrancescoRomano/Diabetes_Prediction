# ---------------------- Load Required Libraries -----------------------
required_packages <- c("dplyr", "caret", "readr", "yardstick", "pROC", "corrplot", 
                       "Hmisc", "boot", "zoo", "ggplot2", "lubridate", "xgboost", 
                       "glmnet", "nnet", "scales", "plotly", "ggthemes", "GGally", 
                       "e1071", "caretEnsemble", "ROSE", "caretStack")

installed_packages <- rownames(installed.packages())
for(pkg in required_packages){
  if(!pkg %in% installed_packages){
    install.packages(pkg, dependencies = TRUE)
  }
}

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
library(caretEnsemble)
library(ROSE)
library(caretStack)

# --------------------- Data Loading & Exploration ---------------------
diabetes <- read_csv("diabetes_prediction_dataset.csv", show_col_types = FALSE)
str(diabetes)
dim(diabetes)
summary(diabetes)
colSums(is.na(diabetes))
diabetes <- diabetes[!duplicated(diabetes), ]
diabetes <- diabetes %>% filter(gender != "Other")

# -------------------- Univariate Analysis ---------------------
ggplot(diabetes, aes(x = "", y = bmi)) + 
  geom_boxplot(fill = "lightblue") +
  labs(title = "BMI Boxplot") +
  theme_minimal()

ggplot(diabetes, aes(x = age)) +
  geom_histogram(binwidth = 5, fill = "lightblue", color = "black", alpha = 0.7) +
  labs(title = "Age Distribution", x = "Age", y = "Frequency") +
  theme_minimal()

ggplot(diabetes, aes(x = gender, fill = gender)) +
  geom_bar(color = "black", width = 0.7) +
  labs(title = "Gender Distribution", x = "Gender", y = "Count") +
  theme_minimal()

ggplot(diabetes, aes(x = bmi)) +
  geom_histogram(aes(y = ..density..), bins = 30, fill = "lightblue", color = "black", alpha = 0.6) + 
  geom_density(color = "blue", size = 1.2) + 
  labs(title = "BMI Distribution", x = "BMI", y = "Density") +
  theme_minimal()

ggplot(diabetes, aes(x = smoking_history, fill = smoking_history)) +
  geom_bar() +
  scale_fill_manual(values = c("blue", "orange", "green", "red", "purple", "brown")) +  
  labs(title = "Smoking History Distribution", x = "Smoking History", y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none")

ggplot(diabetes, aes(x = factor(hypertension), fill = factor(hypertension))) +
  geom_bar() +
  labs(title = "Hypertension Distribution", x = "Hypertension", y = "Count") +
  theme_minimal()

ggplot(diabetes, aes(x = factor(heart_disease), fill = factor(heart_disease))) +
  geom_bar() +
  labs(title = "Heart Disease Distribution", x = "Heart Disease", y = "Count") +
  theme_minimal()

ggplot(diabetes, aes(x = factor(diabetes), fill = factor(diabetes))) +
  geom_bar() +
  labs(title = "Diabetes Distribution", x = "Diabetes", y = "Count") +
  theme_minimal()

# --------------------- Multivariate Analysis ---------------------
ggplot(diabetes, aes(x = gender, fill = factor(diabetes))) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("blue", "orange"), name = "Diabetes", labels = c("0", "1")) +
  labs(title = "Gender vs Diabetes", x = "Gender", y = "Count") +
  theme_minimal()

ggplot(diabetes, aes(x = age, y = bmi, color = factor(diabetes))) +
  geom_point(alpha = 0.6, size = 3) +  
  labs(title = "Age vs BMI colored by Diabetes classification", x = "Age", y = "BMI", color = "Diabetes") +
  scale_color_manual(values = c("lightblue", "orange"), labels = c("0", "1")) +  
  theme_minimal()

# ---------------------- Data Preprocessing -----------------------
recategorize_smoking <- function(smoking_status) {
  if (smoking_status %in% c('never', 'No Info')) {
    return('non.smoker')
  } else if (smoking_status == 'current') {
    return('current')
  } else {
    return('past.smoker')
  }
}

diabetes$smoking_history <- sapply(diabetes$smoking_history, recategorize_smoking)

dummy <- dummyVars(~ gender + smoking_history, data = diabetes, fullRank = FALSE)
dummy_vars <- predict(dummy, newdata = diabetes)
diabetes_processed <- cbind(diabetes, dummy_vars)
diabetes_processed$gender <- NULL
diabetes_processed$smoking_history <- NULL

# ---------------------- Correlation Matrix -----------------------
numeric_columns <- diabetes_processed[, sapply(diabetes_processed, is.numeric)]
correlation_matrix <- cor(numeric_columns, use = "complete.obs")
corrplot(correlation_matrix, method = "color", 
         col = colorRampPalette(c("blue", "white", "red"))(200),
         type = "upper", tl.col = "black", tl.srt = 45)

correlations <- cor(numeric_columns, use = "complete.obs")['diabetes', ]
print(correlations)

correlation_df <- data.frame(Variable = names(correlations)[-which(names(correlations) == "diabetes")], 
                             Correlation = correlations[-which(names(correlations) == "diabetes")])

ggplot(correlation_df, aes(x = reorder(Variable, Correlation), y = Correlation, fill = Correlation)) +
  geom_bar(stat = "identity", color = "black") +
  scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0, 
                       limits = c(-1, 1), space = "Lab", name = "Correlation") +  
  coord_flip() +  
  labs(title = "Correlation with Diabetes", x = "Variables", y = "Correlation") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 12),
        axis.text = element_text(size = 10))

# ------------------- Data Balancing (SMOTE & Under-sampling) ------------------
library(smotefamily)
table(diabetes_processed$diabetes)
diabetes_processed$diabetes <- as.factor(diabetes_processed$diabetes)

set.seed(123)
smote_result <- SMOTE(X = diabetes_processed[, -which(names(diabetes_processed) == "diabetes")], 
                      target = diabetes_processed$diabetes, 
                      K = 5, dup_size = 2)

diabetes_smote <- data.frame(smote_result$data)
names(diabetes_smote)[ncol(diabetes_smote)] <- "diabetes"
diabetes_smote$diabetes <- as.factor(diabetes_smote$diabetes)

table(diabetes_smote$diabetes)

original_rows <- nrow(diabetes_processed)
smote_rows <- nrow(diabetes_smote)

cat("Number of rows in the original dataset:", original_rows, "\n")
cat("Number of rows in the dataset after SMOTE:", smote_rows, "\n")

new_rows <- setdiff(diabetes_smote, diabetes_processed)
new_rows_count <- nrow(new_rows)

cat("Number of synthetic rows added:", new_rows_count, "\n")

if (new_rows_count > 0) {
  cat("First rows of the new synthetic instances:\n")
  print(head(new_rows))
} else {
  cat("No new synthetic rows were added.\n")
}

# ----------------------------- Feature Scaling --------------------------------
columns_to_standardize <- c("bmi", "HbA1c_level", "blood_glucose_level", "age")
diabetes_smote[columns_to_standardize] <- scale(diabetes_smote[columns_to_standardize])
head(diabetes_smote)

# ---------------------------- Model Training ---------------------------------
set.seed(123)
diabetes_sample <- diabetes_smote[sample(nrow(diabetes_smote), 2000), ]
trainIndex <- createDataPartition(diabetes_sample$diabetes, p = .8, list = FALSE, times = 1)
diabetes_train <- diabetes_sample[trainIndex, ]
diabetes_test <- diabetes_sample[-trainIndex, ]

set.seed(123)
rf_grid <- expand.grid(mtry = 2:4)
rf_control <- trainControl(method = "cv", number = 3)
rf_model <- train(diabetes ~ ., data = diabetes_train, method = "rf", 
                  trControl = rf_control, tuneGrid = rf_grid, ntree = 150, maxnodes = 30)
print(rf_model)

rf_pred <- predict(rf_model, diabetes_test)
rf_confusion <- confusionMatrix(rf_pred, diabetes_test$diabetes)

rf_train_pred <- predict(rf_model, diabetes_train)
rf_train_confusion <- confusionMatrix(rf_train_pred, diabetes_train$diabetes)

svm_grid <- expand.grid(C = c(0.1, 1, 10), sigma = c(0.01, 0.1, 1))
svm_control <- trainControl(method = "cv", number = 3)
svm_model <- train(diabetes ~ ., data = diabetes_train, method = "svmRadial", 
                   trControl = svm_control, tuneGrid = svm_grid)
print(svm_model)

svm_pred <- predict(svm_model, diabetes_test)
svm_confusion <- confusionMatrix(svm_pred, diabetes_test$diabetes)

svm_train_pred <- predict(svm_model, diabetes_train)
svm_train_confusion <- confusionMatrix(svm_train_pred, diabetes_train$diabetes)

logistic_model <- glm(diabetes ~ ., data = diabetes_train, family = binomial)
logistic_prob <- predict(logistic_model, diabetes_test, type = "response")
logistic_pred <- ifelse(logistic_prob > 0.5, 1, 0)
logistic_pred <- as.factor(logistic_pred)
logistic_confusion <- confusionMatrix(logistic_pred, diabetes_test$diabetes)

logistic_train_prob <- predict(logistic_model, diabetes_train, type = "response")
logistic_train_pred <- ifelse(logistic_train_prob > 0.5, 1, 0)
logistic_train_pred <- as.factor(logistic_train_pred)
logistic_train_confusion <- confusionMatrix(logistic_train_pred, diabetes_train$diabetes)

results <- data.frame(
  Model = c("Random Forest", "SVM", "Logistic Regression"),
  Accuracy = c(rf_confusion$overall['Accuracy'],
               svm_confusion$overall['Accuracy'],
               logistic_confusion$overall['Accuracy']),
  Precision = c(rf_confusion$byClass['Precision'][1],
                svm_confusion$byClass['Precision'][1],
                logistic_confusion$byClass['Precision'][1]), 
  Recall = c(rf_confusion$byClass['Sensitivity'][1],
             svm_confusion$byClass['Sensitivity'][1],
             logistic_confusion$byClass['Sensitivity'][1]),
  F1_Score = c(2 * (rf_confusion$byClass['Precision'][1] * rf_confusion$byClass['Sensitivity'][1]) / 
                 (rf_confusion$byClass['Precision'][1] + rf_confusion$byClass['Sensitivity'][1]),
               2 * (svm_confusion$byClass['Precision'][1] * svm_confusion$byClass['Sensitivity'][1]) / 
                 (svm_confusion$byClass['Precision'][1] + svm_confusion$byClass['Sensitivity'][1]),
               2 * (logistic_confusion$byClass['Precision'][1] * 
                      logistic_confusion$byClass['Sensitivity'][1]) / 
                 (logistic_confusion$byClass['Precision'][1] + logistic_confusion$byClass['Sensitivity'][1])
  ))

train_results <- data.frame(
  Model = c("Random Forest", "SVM", "Logistic Regression"),
  Train_Accuracy = c(rf_train_confusion$overall['Accuracy'],
                     svm_train_confusion$overall['Accuracy'],
                     logistic_train_confusion$overall['Accuracy']),
  Train_Precision = c(rf_train_confusion$byClass['Precision'][1],
                      svm_train_confusion$byClass['Precision'][1],
                      logistic_train_confusion$byClass['Precision'][1]), 
  Train_Recall = c(rf_train_confusion$byClass['Sensitivity'][1],
                   svm_train_confusion$byClass['Sensitivity'][1],
                   logistic_train_confusion$byClass['Sensitivity'][1]),
  Train_F1_Score = c(2 * (rf_train_confusion$byClass['Precision'][1] * rf_train_confusion$byClass['Sensitivity'][1]) / 
                       (rf_train_confusion$byClass['Precision'][1] + rf_train_confusion$byClass['Sensitivity'][1]),
                     2 * (svm_train_confusion$byClass['Precision'][1] * svm_train_confusion$byClass['Sensitivity'][1]) / 
                       (svm_train_confusion$byClass['Precision'][1] + svm_train_confusion$byClass['Sensitivity'][1]),
                     2 * (logistic_train_confusion$byClass['Precision'][1] * 
                            logistic_train_confusion$byClass['Sensitivity'][1]) / 
                       (logistic_train_confusion$byClass['Precision'][1] + logistic_train_confusion$byClass['Sensitivity'][1])
  )
)

final_results_with_train <- cbind(results, train_results)
print("Combined Performance Metrics (Train & Test):")
print(final_results_with_train)

predictions <- data.frame(
  RF = rf_pred,
  SVM = svm_pred,
  Logistic = logistic_pred
)

ensemble_pred <- apply(predictions, 1, function(x) {
  as.character(names(sort(table(x), decreasing = TRUE)[1]))
})

ensemble_pred <- as.factor(ensemble_pred)
ensemble_confusion <- confusionMatrix(ensemble_pred, diabetes_test$diabetes)

ensemble_results <- data.frame(
  Model = "Ensemble",
  Accuracy = ensemble_confusion$overall['Accuracy'],
  Precision = ensemble_confusion$byClass['Precision'][1],
  Recall = ensemble_confusion$byClass['Sensitivity'][1],
  F1_Score = 2 * (ensemble_confusion$byClass['Precision'][1] * ensemble_confusion$byClass['Sensitivity'][1]) / 
    (ensemble_confusion$byClass['Precision'][1] + ensemble_confusion$byClass['Sensitivity'][1])
)

final_results <- rbind(results, ensemble_results)
print("Performance Metrics:")
print(final_results)
