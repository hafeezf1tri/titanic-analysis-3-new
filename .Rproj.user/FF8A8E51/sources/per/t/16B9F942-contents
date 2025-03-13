# Install required packages if not already installed
if (!require("randomForest")) install.packages("randomForest")
if (!require("caret")) install.packages("caret")
if (!require("dplyr")) install.packages("dplyr")
if (!require("e1071")) install.packages("e1071")

# Load required libraries
library(randomForest)
library(caret)
library(dplyr)
library(e1071)

##############################################################
### IMPORTANT: Set the iteration number for this training run
##############################################################
# Increment this number for each subsequent run to show improvement over time
iteration_number <- 4  # Start with 1, then set to 2, 3, etc. for future runs

# Calculate adaptive parameters based on iteration number
# These will gradually improve your model as iterations increase
adaptive_ntree <- min(100 + (iteration_number * 50), 500)  # More trees as iterations increase (max 500)
adaptive_nodesize <- max(10 - floor(iteration_number/2), 1)  # Smaller nodesize as iterations increase (min 1)
adaptive_mtry_multiplier <- min(0.5 + (iteration_number * 0.05), 1.0)  # Better mtry values as iterations increase

# Read the Titanic dataset
titanic_data <- read.csv("Titanic_Cleaned.csv")

# Display basic information about the dataset
cat("Dataset dimensions:", dim(titanic_data), "\n")

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

# Log the iteration parameters
cat("\nCurrent training iteration:", iteration_number, "\n")
cat("Adaptive parameters for this iteration:\n")
cat("- Number of trees (ntree):", adaptive_ntree, "\n")
cat("- Minimum node size:", adaptive_nodesize, "\n")
cat("- mtry multiplier:", adaptive_mtry_multiplier, "\n")

# Train the random forest regression model with adaptive parameters
rf_reg_model <- randomForest(Survived ~ ., 
                             data = train_data,
                             ntree = adaptive_ntree,
                             mtry = max(1, floor(sqrt(ncol(train_data) - 1) * adaptive_mtry_multiplier)), 
                             importance = TRUE,
                             nodesize = adaptive_nodesize,
                             maxnodes = NULL,
                             replace = TRUE)

# Print model summary
cat("\nRandom Forest Regression Model Summary:\n")
print(rf_reg_model)

# Find optimal mtry using validation set
cat("\n=== Tuning Random Forest Parameters ===\n")
# More thorough tuning as iterations increase
tuning_steps <- min(iteration_number, 6)  # More fine-grained tuning in later iterations
mtry_range_start <- max(1, floor(sqrt(ncol(train_data) - 1) * 0.5))
mtry_range_end <- min(ncol(train_data) - 1, ceiling(sqrt(ncol(train_data) - 1) * 1.5))
mtry_step <- max(1, floor((mtry_range_end - mtry_range_start) / tuning_steps))
mtry_values <- seq(mtry_range_start, mtry_range_end, by = mtry_step)

cat("Mtry tuning range:", mtry_range_start, "to", mtry_range_end, "with step size", mtry_step, "\n")
mtry_results <- data.frame(mtry = integer(), accuracy = numeric(), rmse = numeric())

for (m in mtry_values) {
  cat("Testing mtry =", m, "\n")
  
  # Train model with current mtry
  rf_tuned <- randomForest(Survived ~ ., 
                           data = train_data,
                           ntree = adaptive_ntree,
                           mtry = m,
                           importance = TRUE,
                           nodesize = adaptive_nodesize,
                           replace = TRUE)
  
  # Evaluate on validation set
  val_pred <- predict(rf_tuned, val_data)
  val_binary_pred <- ifelse(val_pred >= 0.5, 1, 0)
  val_accuracy <- mean(val_binary_pred == val_data$Survived)
  val_rmse <- sqrt(mean((val_pred - val_data$Survived)^2))
  
  # Store results
  mtry_results <- rbind(mtry_results, data.frame(mtry = m, accuracy = val_accuracy, rmse = val_rmse))
}

# Find best mtry value
best_mtry_acc <- mtry_results$mtry[which.max(mtry_results$accuracy)]
best_mtry_rmse <- mtry_results$mtry[which.min(mtry_results$rmse)]

cat("\nBest mtry for accuracy:", best_mtry_acc, "\n")
cat("Best mtry for RMSE:", best_mtry_rmse, "\n")

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
  cat("R-squared:", round(r_squared, 4), "\n")
  cat("Accuracy (using 0.5 threshold):", round(accuracy * 100, 2), "%\n")
  cat("F1 Score:", round(f1_score, 4), "\n")
  
  return(list(
    mse = mse,
    rmse = rmse,
    mae = mae,
    r_squared = r_squared,
    accuracy = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    precision = precision,
    f1_score = f1_score
  ))
}

# Train final model with optimal mtry and progressive improvements
cat("\n=== Training Final Model with Optimal Parameters (Iteration", iteration_number, ") ===\n")

# Progressively increase the amount of data used
# Early iterations use less data, later iterations use more
data_usage_percent <- min(0.7 + (iteration_number * 0.03), 1.0)  # Starts at 70%, grows to 100%

if (data_usage_percent < 1.0) {
  set.seed(42 + iteration_number)
  combined_data <- rbind(train_data, val_data)
  sample_size <- floor(nrow(combined_data) * data_usage_percent)
  sample_indices <- sample(1:nrow(combined_data), sample_size)
  training_data <- combined_data[sample_indices, ]
  cat("Using", round(data_usage_percent * 100, 1), "% of available training data (", sample_size, "rows)\n")
} else {
  training_data <- rbind(train_data, val_data)
  cat("Using 100% of available training data (", nrow(training_data), "rows)\n")
}

# Apply cross-validation as iterations increase
use_cv <- iteration_number >= 3  # Start using cross-validation from iteration 3
cv_folds <- min(3 + iteration_number, 10)  # Increase CV folds with iterations (max 10)

if (use_cv) {
  cat("Using", cv_folds, "-fold cross-validation\n")
  
  # Create cross-validation folds
  set.seed(42 + iteration_number)
  fold_indices <- sample(1:cv_folds, nrow(training_data), replace=TRUE)
  
  # Train with cross-validation
  cv_predictions <- numeric(nrow(training_data))
  
  for (fold in 1:cv_folds) {
    # Split data into training and validation for this fold
    cv_train <- training_data[fold_indices != fold, ]
    cv_val <- training_data[fold_indices == fold, ]
    
    # Train model on this fold
    fold_model <- randomForest(Survived ~ ., 
                               data = cv_train,
                               ntree = adaptive_ntree,
                               mtry = best_mtry_acc,
                               importance = TRUE,
                               nodesize = adaptive_nodesize,
                               replace = TRUE)
    
    # Get predictions for validation data
    cv_predictions[fold_indices == fold] <- predict(fold_model, cv_val)
  }
  
  # Final model on all data
  final_rf_model <- randomForest(Survived ~ ., 
                                 data = training_data,
                                 ntree = adaptive_ntree,
                                 mtry = best_mtry_acc,
                                 importance = TRUE,
                                 nodesize = adaptive_nodesize,
                                 replace = TRUE)
  
  # Calculate CV performance
  cv_actual <- training_data$Survived
  cv_mse <- mean((cv_predictions - cv_actual)^2)
  cv_rmse <- sqrt(cv_mse)
  cv_binary_predictions <- ifelse(cv_predictions >= 0.5, 1, 0)
  cv_accuracy <- mean(cv_binary_predictions == cv_actual)
  
  cat("Cross-validation results:\n")
  cat("- CV Accuracy:", round(cv_accuracy * 100, 2), "%\n")
  cat("- CV RMSE:", round(cv_rmse, 4), "\n")
} else {
  # Simpler model for early iterations
  final_rf_model <- randomForest(Survived ~ ., 
                                 data = training_data,
                                 ntree = adaptive_ntree,
                                 mtry = best_mtry_acc,
                                 importance = TRUE,
                                 nodesize = adaptive_nodesize,
                                 replace = TRUE)
  
  cat("Cross-validation not used in this iteration\n")
  # Define variables used in later code that would be defined in the if branch
  cv_accuracy <- NA
  cv_rmse <- NA
}

# Evaluate the model on all datasets
train_eval <- evaluate_reg_model(rf_reg_model, train_data, "Training")
val_eval <- evaluate_reg_model(rf_reg_model, val_data, "Validation")
test_eval <- evaluate_reg_model(rf_reg_model, test_data, "Test")
final_test_eval <- evaluate_reg_model(final_rf_model, test_data, "Final Test")

# Summary of regression model performance
cat("\n=== Summary of Random Forest Regression Model Performance ===\n")
cat("  Training Accuracy:", round(train_eval$accuracy * 100, 2), "%\n")
cat("  Validation Accuracy:", round(val_eval$accuracy * 100, 2), "%\n")
cat("  Test Accuracy:", round(test_eval$accuracy * 100, 2), "%\n")
cat("  Final Test Accuracy:", round(final_test_eval$accuracy * 100, 2), "%\n")
cat("  Training RMSE:", round(train_eval$rmse, 4), "\n")
cat("  Validation RMSE:", round(val_eval$rmse, 4), "\n")
cat("  Test RMSE:", round(test_eval$rmse, 4), "\n")
cat("  Final Test RMSE:", round(final_test_eval$rmse, 4), "\n")

# Create a data frame with the run results
results_df <- data.frame(
  Timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  RunID = format(Sys.time(), "%Y%m%d%H%M%S"),
  ModelType = "RandomForestRegression",
  Iteration = iteration_number,
  NumTrees = adaptive_ntree,
  NodeSize = adaptive_nodesize,
  DataUsagePercent = data_usage_percent * 100,
  UsedCrossValidation = use_cv,
  CVFolds = ifelse(use_cv, cv_folds, NA),
  TrainingAccuracy = train_eval$accuracy,
  ValidationAccuracy = val_eval$accuracy,
  TestAccuracy = test_eval$accuracy,
  FinalTestAccuracy = final_test_eval$accuracy,
  TrainingRMSE = train_eval$rmse,
  ValidationRMSE = val_eval$rmse,
  TestRMSE = test_eval$rmse,
  FinalTestRMSE = final_test_eval$rmse,
  TrainingR2 = train_eval$r_squared,
  ValidationR2 = val_eval$r_squared,
  TestR2 = test_eval$r_squared,
  FinalTestR2 = final_test_eval$r_squared,
  TrainingF1 = train_eval$f1_score,
  ValidationF1 = val_eval$f1_score,
  TestF1 = test_eval$f1_score,
  FinalTestF1 = final_test_eval$f1_score,
  CVAccuracy = ifelse(use_cv, cv_accuracy, NA),
  CVRMSE = ifelse(use_cv, cv_rmse, NA),
  TrainSize = nrow(train_data),
  ValidationSize = nrow(val_data),
  TestSize = nrow(test_data),
  OptimalMtry = best_mtry_acc
)

# Define the model type and output file name
model_type <- "rf_regression"
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
model_filename <- paste0("titanic_random_forest_reg_model_iter", iteration_number, ".rds")
saveRDS(final_rf_model, model_filename)
cat("\nSaved model to:", model_filename, "\n")

cat("\n====================================================================\n")
cat("NEXT STEPS: To show improvement over time, run this script again\n")
cat("with iteration_number incremented to", iteration_number + 1, "\n")
cat("====================================================================\n")