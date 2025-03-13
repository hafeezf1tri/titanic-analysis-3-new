# Install required packages if not already installed
if (!require("e1071")) install.packages("e1071")  # For SVM/SVR
if (!require("caret")) install.packages("caret")
if (!require("dplyr")) install.packages("dplyr")
if (!require("ggplot2")) install.packages("ggplot2")

# Load required libraries
library(e1071)  # For SVR
library(caret)
library(dplyr)
library(ggplot2)

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

# Tune SVR hyperparameters using cross-validation on the training set
cat("\nTuning SVR hyperparameters using cross-validation...\n")
tune_results <- tune.svm(Survived ~ ., 
                         data = train_data,
                         type = "eps-regression",
                         kernel = "radial",
                         gamma = c(0.01, 0.05, 0.1, 0.5),
                         cost = c(1, 5, 10, 50),
                         epsilon = c(0.01, 0.05, 0.1, 0.5),
                         tunecontrol = tune.control(cross = 5))  # 5-fold cross-validation

# Print tuning results
cat("\nHyperparameter tuning results:\n")
print(tune_results)

# Get the best model parameters
best_params <- tune_results$best.parameters
cat("\nBest parameters:\n")
print(best_params)

# Train the final SVR model with the best parameters
svr_model <- svm(Survived ~ ., 
                 data = train_data,
                 type = "eps-regression",
                 kernel = "radial",
                 gamma = best_params$gamma,
                 cost = best_params$cost,
                 epsilon = best_params$epsilon)

# Print model summary
cat("\nSupport Vector Regression Model Summary:\n")
print(svr_model)

# Function to evaluate regression model performance
evaluate_reg_model <- function(model, data, dataset_name, noise_level = 0) {
  # Get actual target values
  actual <- data$Survived
  
  # Make predictions with optional noise
  raw_predictions <- predict(model, data)
  
  # Add random noise to predictions to simulate fluctuation
  if (noise_level > 0) {
    set.seed(as.integer(Sys.time())) # Different seed each time
    noise <- rnorm(length(raw_predictions), mean = 0, sd = noise_level)
    predictions <- raw_predictions + noise
  } else {
    predictions <- raw_predictions
  }
  
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

# Function to randomly sample a subset of data
sample_data <- function(data, sample_ratio = 0.8) {
  set.seed(as.integer(Sys.time())) # Different seed each time
  n <- nrow(data)
  sample_size <- floor(sample_ratio * n)
  sample_indices <- sample(1:n, sample_size)
  return(data[sample_indices, ])
}

# Generate a random sampling ratio (between 70% and 95% of the test data)
test_sample_ratio <- runif(1, 0.7, 0.95)
cat("\nUsing random subset (", round(test_sample_ratio * 100, 1), "%) of test data\n", sep="")

# Create random test subset
random_test_data <- sample_data(test_data, sample_ratio = test_sample_ratio)

# Evaluate the model on all datasets
train_eval <- evaluate_reg_model(svr_model, train_data, "Training", noise_level = 0)
val_eval <- evaluate_reg_model(svr_model, val_data, "Validation", noise_level = 0)
test_eval <- evaluate_reg_model(svr_model, random_test_data, "Test (Sampled)", noise_level = 0)

# Summary of regression model performance
cat("\n=== Summary of Regression Model Performance ===\n")
cat("  Training Accuracy:", round(train_eval$accuracy * 100, 2), "%\n")
cat("  Validation Accuracy:", round(val_eval$accuracy * 100, 2), "%\n")
cat("  Test Accuracy:", round(test_eval$accuracy * 100, 2), "%\n")
cat("  Training RMSE:", round(train_eval$rmse, 4), "\n")
cat("  Validation RMSE:", round(val_eval$rmse, 4), "\n")
cat("  Test RMSE:", round(test_eval$rmse, 4), "\n")

# Create visualization of predictions vs actual values
plot_data <- data.frame(
  Actual = test_data$Survived,
  Predicted = predict(svr_model, test_data)
)

ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "SVR: Actual vs Predicted Survival",
       x = "Actual Survival Value",
       y = "Predicted Survival Value") +
  theme_minimal()

# Create a data frame with the run results
results_df <- data.frame(
  Timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  RunID = format(Sys.time(), "%Y%m%d%H%M%S"),
  ModelType = "SVR",
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
model_type <- "svr"
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
saveRDS(svr_model, "titanic_svr_model.rds")