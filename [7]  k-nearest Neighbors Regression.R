# Install required packages if not already installed
if (!require("FNN")) install.packages("FNN")
if (!require("caret")) install.packages("caret")
if (!require("dplyr")) install.packages("dplyr")
if (!require("e1071")) install.packages("e1071")

# Load required libraries
library(FNN)
library(caret)
library(dplyr)
library(e1071)

# Read the Titanic dataset
titanic_data <- read.csv("Titanic_Cleaned.csv")

# Display basic information about the dataset
cat("Dataset dimensions:", dim(titanic_data), "\n")
cat("\nFirst few rows of the dataset:\n")
print(head(titanic_data))
cat("\nSummary of the dataset:\n")
print(summary(titanic_data))

# Check for the 'No.' column and remove it if it exists
if ("No." %in% colnames(titanic_data)) {
  titanic_data <- titanic_data %>% select(-`No.`)
  cat("\nRemoved 'No.' column from the dataset.\n")
}

# Convert 'Sex' to a numeric variable for kNN (0 for male, 1 for female)
titanic_data$Sex <- ifelse(titanic_data$Sex == "female", 1, 0)

# For regression, keep Survived as numeric (0 or 1)
# If Survived is already a factor, convert it to numeric
if (is.factor(titanic_data$Survived)) {
  titanic_data$Survived <- as.numeric(as.character(titanic_data$Survived))
}

# Create a function to normalize the data
normalize <- function(x) {
  return((x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE)))
}

# Apply normalization to all numeric columns
titanic_norm <- titanic_data
numeric_cols <- sapply(titanic_norm, is.numeric)
titanic_norm[numeric_cols] <- lapply(titanic_norm[numeric_cols], normalize)

# To make results fluctuate, create a random seed based on current time
seed_value <- as.numeric(format(Sys.time(), "%H%M%S")) %% 1000
set.seed(seed_value)
cat("\nUsing seed value:", seed_value, "for this run\n")

# Calculate the sample sizes
n <- nrow(titanic_norm)
train_size <- floor(0.8 * n)
val_size <- floor(0.1 * n)
test_size <- n - train_size - val_size

# Create indices for the different sets
indices <- sample(1:n, n)
train_indices <- indices[1:train_size]
val_indices <- indices[(train_size + 1):(train_size + val_size)]
test_indices <- indices[(train_size + val_size + 1):n]

# Split the data
train_data <- titanic_norm[train_indices, ]
val_data <- titanic_norm[val_indices, ]
test_data <- titanic_norm[test_indices, ]

# Verify the split sizes
cat("\nSplit sizes:\n")
cat("Training set:", nrow(train_data), "observations (", round(nrow(train_data)/n * 100, 2), "%)\n")
cat("Validation set:", nrow(val_data), "observations (", round(nrow(val_data)/n * 100, 2), "%)\n")
cat("Test set:", nrow(test_data), "observations (", round(nrow(test_data)/n * 100, 2), "%)\n")

# Separate predictors and target variable
train_x <- train_data %>% select(-Survived)
train_y <- train_data$Survived
val_x <- val_data %>% select(-Survived)
val_y <- val_data$Survived
test_x <- test_data %>% select(-Survived)
test_y <- test_data$Survived

# Function to tune k value using validation set
tune_knn <- function(train_x, train_y, val_x, val_y, k_values = 1:20) {
  k_results <- data.frame(k = integer(), mse = numeric())
  
  for (k in k_values) {
    knn_pred <- knn.reg(train = as.matrix(train_x), 
                        test = as.matrix(val_x), 
                        y = train_y, 
                        k = k)
    
    mse <- mean((knn_pred$pred - val_y)^2)
    k_results <- rbind(k_results, data.frame(k = k, mse = mse))
  }
  
  best_k <- k_results$k[which.min(k_results$mse)]
  cat("\nBest k value:", best_k, "\n")
  cat("Validation MSE for best k:", min(k_results$mse), "\n")
  
  return(best_k)
}

# Find the best k value
k_values <- 1:min(30, nrow(train_data) - 1)  # Ensure k doesn't exceed sample size - 1
best_k <- tune_knn(train_x, train_y, val_x, val_y, k_values)

# Small random variation to k can cause fluctuation in results
# Adding a small random offset (+/- 2) to the best k value
k_offset <- sample(-2:2, 1)
final_k <- max(1, best_k + k_offset)  # Ensure k is at least 1
cat("Using k =", final_k, "for final model (best_k", best_k, "+ offset", k_offset, ")\n")

# Train the KNN regression model on the combined training and validation data
combined_train_x <- rbind(train_x, val_x)
combined_train_y <- c(train_y, val_y)

# Function to evaluate KNN regression model performance
evaluate_knn_model <- function(train_x, train_y, test_x, test_y, k, dataset_name) {
  # Make predictions using KNN
  knn_pred <- knn.reg(train = as.matrix(train_x), 
                      test = as.matrix(test_x), 
                      y = train_y, 
                      k = k)
  
  predictions <- knn_pred$pred
  actual <- test_y
  
  # Convert predictions to binary classification (for accuracy calculation)
  binary_predictions <- ifelse(predictions >= 0.5, 1, 0)
  
  # Calculate metrics
  mse <- mean((predictions - actual)^2)
  rmse <- sqrt(mse)
  mae <- mean(abs(predictions - actual))
  r_squared <- 1 - sum((actual - predictions)^2) / sum((actual - mean(actual))^2)
  accuracy <- mean(binary_predictions == actual)
  
  # Calculate sensitivity, specificity, precision
  true_pos <- sum(binary_predictions == 1 & actual == 1)
  true_neg <- sum(binary_predictions == 0 & actual == 0)
  false_pos <- sum(binary_predictions == 1 & actual == 0)
  false_neg <- sum(binary_predictions == 0 & actual == 1)
  
  sensitivity <- true_pos / (true_pos + false_neg)
  specificity <- true_neg / (true_neg + false_pos)
  precision <- true_pos / (true_pos + false_pos)
  f1_score <- 2 * precision * sensitivity / (precision + sensitivity)
  
  # Print results
  cat("\n=== ", dataset_name, " Set KNN Regression Evaluation ===\n")
  cat("k value:", k, "\n")
  cat("MSE:", round(mse, 4), "\n")
  cat("RMSE:", round(rmse, 4), "\n")
  cat("MAE:", round(mae, 4), "\n")
  cat("R-squared:", round(r_squared, 4), "\n")
  cat("Accuracy (using 0.5 threshold):", round(accuracy * 100, 2), "%\n")
  cat("Sensitivity/Recall:", round(sensitivity, 4), "\n")
  cat("Specificity:", round(specificity, 4), "\n")
  cat("Precision:", round(precision, 4), "\n")
  cat("F1 Score:", round(f1_score, 4), "\n")
  
  # Confusion matrix
  conf_matrix <- table(Actual = actual, Predicted = binary_predictions)
  cat("\nConfusion Matrix:\n")
  print(conf_matrix)
  
  return(list(
    k = k,
    mse = mse,
    rmse = rmse,
    mae = mae,
    r_squared = r_squared,
    accuracy = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    precision = precision,
    f1_score = f1_score,
    conf_matrix = conf_matrix,
    predictions = predictions
  ))
}

# Evaluate on training data
train_eval <- evaluate_knn_model(train_x, train_y, train_x, train_y, final_k, "Training")

# Evaluate on validation data
val_eval <- evaluate_knn_model(train_x, train_y, val_x, val_y, final_k, "Validation")

# Evaluate on test data
test_eval <- evaluate_knn_model(combined_train_x, combined_train_y, test_x, test_y, final_k, "Test")

# Summary of KNN regression model performance
cat("\n=== Summary of KNN Regression Model Performance ===\n")
cat("  k value:", final_k, "\n")
cat("  Training Accuracy:", round(train_eval$accuracy * 100, 2), "%\n")
cat("  Validation Accuracy:", round(val_eval$accuracy * 100, 2), "%\n")
cat("  Test Accuracy:", round(test_eval$accuracy * 100, 2), "%\n")
cat("  Training RMSE:", round(train_eval$rmse, 4), "\n")
cat("  Validation RMSE:", round(val_eval$rmse, 4), "\n")
cat("  Test RMSE:", round(test_eval$rmse, 4), "\n")

# Create a data frame with the run results
results_df <- data.frame(
  Timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  RunID = format(Sys.time(), "%Y%m%d%H%M%S"),
  SeedValue = seed_value,
  KValue = final_k,
  ModelType = "KNN_Regression",
  TrainingAccuracy = train_eval$accuracy,
  ValidationAccuracy = val_eval$accuracy,
  TestAccuracy = test_eval$accuracy,
  TrainingRMSE = train_eval$rmse,
  ValidationRMSE = val_eval$rmse,
  TestRMSE = test_eval$rmse,
  TrainingR2 = train_eval$r_squared,
  ValidationR2 = val_eval$r_squared,
  TestR2 = test_eval$r_squared,
  TrainingF1 = train_eval$f1_score,
  ValidationF1 = val_eval$f1_score,
  TestF1 = test_eval$f1_score,
  TrainSize = nrow(train_data),
  ValidationSize = nrow(val_data),
  TestSize = nrow(test_data)
)

# Define the output file name
results_file <- "titanic_knn_regression_performance.csv"

# Check if the file already exists
if (!file.exists(results_file)) {
  # If the file doesn't exist, write the data frame with headers
  write.csv(results_df, results_file, row.names = FALSE)
  cat("\nCreated new results file:", results_file, "\n")
} else {
  # If the file exists, append the new results without headers
  write.table(results_df, results_file, append = TRUE, sep = ",", 
              row.names = FALSE, col.names = FALSE)
  cat("\nAppended results to existing file:", results_file, "\n")
}

# Print the data that was saved to the CSV
cat("\nSaved the following data to", results_file, ":\n")
print(results_df)

# Create a model object to save
knn_model <- list(
  train_data = train_data,
  train_x = combined_train_x,
  train_y = combined_train_y,
  k = final_k,
  seed = seed_value,
  normalization_function = normalize
)

# Save the model information
saveRDS(knn_model, "titanic_knn_reg_model.rds")
cat("\nSaved model to titanic_knn_reg_model.rds\n")