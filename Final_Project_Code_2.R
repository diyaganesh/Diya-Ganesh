# =====================
# GN 428 Final Project
# Heart Disease Prediction
# =====================

# Hypothesis:
# Clinical and physiological features are associated with heart disease 
# classification, and machine learning models can accurately predict the 
# presence of heart disease.

# ------------------------
# 1. Load data
# ------------------------
data <- read.csv("heart.csv")

str(data)
head(data)
summary(data)

# ------------------------
# 2. Clean and prepare data
# ------------------------

# Convert outcome variable to factor
data$HeartDisease <- as.factor(data$HeartDisease)

# Convert categorical variables to factors
data$Sex <- as.factor(data$Sex)
data$ChestPainType <- as.factor(data$ChestPainType)
data$RestingECG <- as.factor(data$RestingECG)
data$ExerciseAngina <- as.factor(data$ExerciseAngina)
data$ST_Slope <- as.factor(data$ST_Slope)

# Check for missing values
colSums(is.na(data))

# Remove duplicate rows if any
data <- unique(data)

str(data)

# ------------------------
# 3. Split data into train and test sets
# ------------------------
set.seed(123)

n <- nrow(data)
train_index <- sample(1:n, size = 0.7 * n)

train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

# ------------------------
# 4. Decision Tree Model
# ------------------------
library(rpart)
library(rpart.plot)

tree_model <- rpart(HeartDisease ~ ., data = train_data, method = "class")

print(tree_model)
summary(tree_model)

rpart.plot(tree_model)

tree_pred <- predict(tree_model, newdata = test_data, type = "class")

tree_conf_matrix <- table(Predicted = tree_pred, Actual = test_data$HeartDisease)
print(tree_conf_matrix)

tree_accuracy <- mean(tree_pred == test_data$HeartDisease)
print(tree_accuracy)

# For heart disease:
# 1 = heart disease
# 0 = no heart disease

TP_tree <- tree_conf_matrix["1", "1"]
TN_tree <- tree_conf_matrix["0", "0"]
FP_tree <- tree_conf_matrix["1", "0"]
FN_tree <- tree_conf_matrix["0", "1"]

tree_sensitivity <- TP_tree / (TP_tree + FN_tree)
tree_specificity <- TN_tree / (TN_tree + FP_tree)

print(tree_sensitivity)
print(tree_specificity)

tree_model$variable.importance

# ------------------------
# 5. Logistic Regression Model
# ------------------------
log_model <- glm(HeartDisease ~ ., data = train_data, family = "binomial")

log_prob <- predict(log_model, newdata = test_data, type = "response")

log_pred <- ifelse(log_prob > 0.5, "1", "0")
log_pred <- factor(log_pred, levels = c("0", "1"))

log_conf_matrix <- table(Predicted = log_pred, Actual = test_data$HeartDisease)
print(log_conf_matrix)

log_accuracy <- mean(log_pred == test_data$HeartDisease)
print(log_accuracy)

TP_log <- log_conf_matrix["1", "1"]
TN_log <- log_conf_matrix["0", "0"]
FP_log <- log_conf_matrix["1", "0"]
FN_log <- log_conf_matrix["0", "1"]

log_sensitivity <- TP_log / (TP_log + FN_log)
log_specificity <- TN_log / (TN_log + FP_log)

print(log_sensitivity)
print(log_specificity)

# ------------------------
# 6. KNN Model
# ------------------------
library(class)

# Create dummy variables for categorical predictors
x_data <- model.matrix(HeartDisease ~ ., data = data)[, -1]
y_data <- data$HeartDisease

train_x <- x_data[train_index, ]
test_x  <- x_data[-train_index, ]

train_y <- y_data[train_index]
test_y  <- y_data[-train_index]

# Normalize features for KNN
train_x_scaled <- scale(train_x)

test_x_scaled <- scale(test_x,
                       center = attr(train_x_scaled, "scaled:center"),
                       scale = attr(train_x_scaled, "scaled:scale"))

knn_pred <- knn(train = train_x_scaled, test = test_x_scaled, cl = train_y, k = 5)

knn_conf_matrix <- table(Predicted = knn_pred, Actual = test_y)
print(knn_conf_matrix)

knn_accuracy <- mean(knn_pred == test_y)
print(knn_accuracy)

TP_knn <- knn_conf_matrix["1", "1"]
TN_knn <- knn_conf_matrix["0", "0"]
FP_knn <- knn_conf_matrix["1", "0"]
FN_knn <- knn_conf_matrix["0", "1"]

knn_sensitivity <- TP_knn / (TP_knn + FN_knn)
knn_specificity <- TN_knn / (TN_knn + FP_knn)

print(knn_sensitivity)
print(knn_specificity)

# ------------------------
# 7. Compare model performance
# ------------------------
model_results <- data.frame(
  Model = c("Decision Tree", "Logistic Regression", "KNN"),
  Accuracy = c(tree_accuracy, log_accuracy, knn_accuracy),
  Sensitivity = c(tree_sensitivity, log_sensitivity, knn_sensitivity),
  Specificity = c(tree_specificity, log_specificity, knn_specificity)
)

print(model_results)

# ------------------------
# 8. Save objects for submission
# ------------------------
save(data, train_data, test_data,
     tree_model, tree_pred, tree_conf_matrix, tree_accuracy, tree_sensitivity, 
     tree_specificity, log_model, log_pred, log_conf_matrix, log_accuracy,
     log_sensitivity, log_specificity,knn_pred, knn_conf_matrix, knn_accuracy, 
     knn_sensitivity, knn_specificity, model_results,
     file = "Final_Project_Heart_Disease.RData")
