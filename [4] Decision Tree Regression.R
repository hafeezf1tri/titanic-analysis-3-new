# Install required packages if not already installed
if (!require("rpart")) install.packages("rpart")
if (!require("rpart.plot")) install.packages("rpart.plot")
if (!require("caret")) install.packages("caret")
if (!require("dplyr")) install.packages("dplyr")
if (!require("e1071")) install.packages("e1071")

# Load required libraries
library(rpart)
library(rpart.plot)
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

# Convert 'Sex' to a factor variable
titanic_data$Sex <- as.factor(titanic_data$Sex)

# For regression, keep Survived as numeric (0 or 1)
# If Survived is already a factor, convert it to numeric
if (is.factor(titanic_data$Survived)) {
  titanic_data$Survived <- as.numeric(as.character(titanic_data$Survived))
}

# Create indices for splitting the data into train (80%), validation (10%), and test (10%)
set.seed(42) # For reproducibility

# Calculate the sample sizes
n <- nrow(titanic_data)
train_size <- floor(0.8 * n)
val_size <- floor(0.1 * n)
test_size <- n - train_size - val_size

# Create indices for the different sets
indices <- sample(1:n, n)
train_indices <- indices[1:train_size]
val_indices <- indices[(train_size + 1):(train_size + val_size)]
test_indices <- indices[(train_size + val_size + 1):n]

# Split the data
train_data <- titanic_data[train_indices, ]
val_data <- titanic_data[val_indices, ]
test_data <- titanic_data[test_indices, ]

# Verify the split sizes
cat("\nSplit sizes:\n")
cat("Training set:", nrow(train_data), "observations (", round(nrow(train_data)/n * 100, 2), "%)\n")
cat("Validation set:", nrow(val_data), "observations (", round(nrow(val_data)/n * 100, 2), "%)\n")
cat("Test set:", nrow(test_data), "observations (", round(nrow(test_data)/n * 100, 2), "%)\n")

# Train the decision tree regression model
dt_reg_model <- rpart(Survived ~ ., 
                      data = train_data, 
                      method = "anova",  # Use anova for regression
                      control = rpart.control(minsplit = 20, 
                                              minbucket = 7, 
                                              maxdepth = 30, 
                                              cp = 0.01))

# Print model summary
cat("\nDecision Tree Regression Model Summary:\n")
print(dt_reg_model)

# Plot the decision tree
pdf("decision_tree_regression_plot.pdf", width = 12, height = 8)
rpart.plot(dt_reg_model, extra = 101, box.palette = "RdBu", shadow.col = "gray")
dev.off()

# Variable importance
var_importance <- dt_reg_model$variable.importance
var_importance_df <- data.frame(
  Feature = names(var_importance),
  Importance = var_importance
)
var_importance_df <- var_importance_df[order(-var_importance_df$Importance), ]
cat("\nVariable Importance:\n")
print(var_importance_df)

# Function to evaluate regression model performance
evaluate_reg_model <- function(model, data, dataset_name) {
  # Get actual target values
  actual <- data$Survived
  
  # Make predictions
  predictions <- predict(model, data)
  
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
  cat("\n=== ", dataset_name, " Set Regression Evaluation ===\n")
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
    mse = mse,
    rmse = rmse,
    mae = mae,
    r_squared = r_squared,
    accuracy = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    precision = precision,
    f1_score = f1_score,
    conf_matrix = conf_matrix
  ))
}

# Evaluate the model on all datasets
train_eval <- evaluate_reg_model(dt_reg_model, train_data, "Training")
val_eval <- evaluate_reg_model(dt_reg_model, val_data, "Validation")
test_eval <- evaluate_reg_model(dt_reg_model, test_data, "Test")

# Summary of regression model performance
cat("\n=== Summary of Regression Model Performance ===\n")
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
  ModelType = "Regression",
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

# Define the model type and output file name
model_type <- "dt_regression"
results_file <- paste0("titanic_", model_type, "_performance.csv")

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

# Save the model
saveRDS(dt_reg_model, "titanic_decision_tree_reg_model.rds")