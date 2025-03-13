
# Install required packages if not already installed
if (!require("rpart")) install.packages("rpart")
if (!require("rpart.plot")) install.packages("rpart.plot")
if (!require("caret")) install.packages("caret")
if (!require("dplyr")) install.packages("dplyr")
if (!require("e1071")) install.packages("e1071")
if (!require("ROSE")) install.packages("ROSE")  # For data synthesis

# Load required libraries
library(rpart)
library(rpart.plot)
library(caret)
library(dplyr)
library(e1071)
library(ROSE)  # For data synthesis

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
train_data_original <- titanic_data[train_indices, ]
val_data <- titanic_data[val_indices, ]
test_data <- titanic_data[test_indices, ]

# Verify the split sizes
cat("\nOriginal Split sizes:\n")
cat("Training set:", nrow(train_data_original), "observations (", round(nrow(train_data_original)/n * 100, 2), "%)\n")
cat("Validation set:", nrow(val_data), "observations (", round(nrow(val_data)/n * 100, 2), "%)\n")
cat("Test set:", nrow(test_data), "observations (", round(nrow(test_data)/n * 100, 2), "%)\n")

# Check class balance in the training set
cat("\nClass Balance in Original Training Data:\n")
table(train_data_original$Survived)

# Add basic feature engineering (but keep it simple compared to tuned version)
# Create a family size feature
train_data_original$FamilySize <- train_data_original$SibSp + train_data_original$PrCh
val_data$FamilySize <- val_data$SibSp + val_data$PrCh
test_data$FamilySize <- test_data$SibSp + test_data$PrCh

# Create a binary feature for traveling alone
train_data_original$IsAlone <- ifelse(train_data_original$FamilySize == 0, 1, 0)
val_data$IsAlone <- ifelse(val_data$FamilySize == 0, 1, 0)
test_data$IsAlone <- ifelse(test_data$FamilySize == 0, 1, 0)

# =============================================
# DATA SYNTHESIS APPROACHES
# =============================================

# 1. ROSE (Random Over-Sampling Examples)
# This creates synthetic examples using smoothed bootstrapping
set.seed(42)
rose_data <- ROSE(Survived ~ ., data = train_data_original, N = nrow(train_data_original) * 2)$data
cat("\nROSE Synthetic Data Size:", nrow(rose_data), "observations\n")
cat("Class Balance in ROSE Synthetic Data:\n")
table(rose_data$Survived)

# 2. Over-sampling the minority class
# Identify the minority class
survived_count <- sum(train_data_original$Survived == 1)
not_survived_count <- sum(train_data_original$Survived == 0)
minority_class <- ifelse(survived_count < not_survived_count, 1, 0)

# Extract minority and majority class samples
minority_samples <- train_data_original[train_data_original$Survived == minority_class, ]
majority_samples <- train_data_original[train_data_original$Survived != minority_class, ]

# Over-sample the minority class with replacement
set.seed(42)
oversampled_minority <- minority_samples[sample(1:nrow(minority_samples), nrow(majority_samples), replace = TRUE), ]

# Combine with majority class
oversampled_data <- rbind(majority_samples, oversampled_minority)
oversampled_data <- oversampled_data[sample(1:nrow(oversampled_data)), ]  # Shuffle

cat("\nOver-sampling Synthetic Data Size:", nrow(oversampled_data), "observations\n")
cat("Class Balance in Over-sampling Synthetic Data:\n")
table(oversampled_data$Survived)

# 3. Add small noise to create more samples
# Function to add noise to numeric columns
add_noise <- function(data, noise_level = 0.05) {
  result <- data
  numeric_cols <- sapply(data, is.numeric)
  
  for (col in names(data)[numeric_cols]) {
    if (col != "Survived") {  # Don't add noise to the target variable
      col_std <- sd(data[[col]], na.rm = TRUE)
      noise <- rnorm(nrow(data), mean = 0, sd = noise_level * col_std)
      result[[col]] <- data[[col]] + noise
    }
  }
  
  return(result)
}

# Create noisy samples
set.seed(42)
noisy_samples <- add_noise(train_data_original)
combined_noise_data <- rbind(train_data_original, noisy_samples)

cat("\nNoise-Added Synthetic Data Size:", nrow(combined_noise_data), "observations\n")
cat("Class Balance in Noise-Added Synthetic Data:\n")
table(combined_noise_data$Survived)

# =============================================
# TRAIN MODELS WITH DIFFERENT DATASETS
# =============================================

# Function to train and evaluate a model
train_and_evaluate <- function(train_data, val_data, test_data, dataset_name) {
  # Train the decision tree regression model
  dt_reg_model <- rpart(Survived ~ ., 
                        data = train_data, 
                        method = "anova",  # Use anova for regression
                        control = rpart.control(minsplit = 20, 
                                                minbucket = 7, 
                                                maxdepth = 30, 
                                                cp = 0.01))
  
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
  
  # Evaluate on validation data
  val_results <- evaluate_reg_model(dt_reg_model, val_data, "Validation")
  
  # Evaluate on test data
  test_results <- evaluate_reg_model(dt_reg_model, test_data, "Test")
  
  # Return results and model
  return(list(
    model = dt_reg_model,
    validation = val_results,
    test = test_results,
    dataset_name = dataset_name
  ))
}

# Train models with different synthetic datasets
cat("\n============ MODELS WITH DIFFERENT DATASETS ============\n")

cat("\n1. ORIGINAL DATA MODEL\n")
original_results <- train_and_evaluate(
  train_data_original, val_data, test_data, "Original Data"
)

cat("\n2. ROSE SYNTHETIC DATA MODEL\n")
rose_results <- train_and_evaluate(
  rose_data, val_data, test_data, "ROSE Synthetic"
)

cat("\n3. OVERSAMPLING DATA MODEL\n")
oversample_results <- train_and_evaluate(
  oversampled_data, val_data, test_data, "Oversampling"
)

cat("\n4. NOISE-ADDED DATA MODEL\n")
noise_results <- train_and_evaluate(
  combined_noise_data, val_data, test_data, "Noise-Added"
)

# =============================================
# COMPARE RESULTS AND SAVE TO CSV
# =============================================

# Create a data frame with the run results for all models
results_df <- data.frame(
  Timestamp = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  RunID = format(Sys.time(), "%Y%m%d%H%M%S"),
  ModelType = c("dt_regression", "dt_regression", "dt_regression", "dt_regression"),
  DatasetType = c("Original", "ROSE_Synthetic", "Oversampling", "Noise_Added"),
  TrainingSize = c(
    nrow(train_data_original),
    nrow(rose_data),
    nrow(oversampled_data),
    nrow(combined_noise_data)
  ),
  ValidationAccuracy = c(
    original_results$validation$accuracy,
    rose_results$validation$accuracy,
    oversample_results$validation$accuracy,
    noise_results$validation$accuracy
  ),
  TestAccuracy = c(
    original_results$test$accuracy,
    rose_results$test$accuracy,
    oversample_results$test$accuracy,
    noise_results$test$accuracy
  ),
  ValidationRMSE = c(
    original_results$validation$rmse,
    rose_results$validation$rmse,
    oversample_results$validation$rmse,
    noise_results$validation$rmse
  ),
  TestRMSE = c(
    original_results$test$rmse,
    rose_results$test$rmse,
    oversample_results$test$rmse,
    noise_results$test$rmse
  ),
  ValidationF1 = c(
    original_results$validation$f1_score,
    rose_results$validation$f1_score,
    oversample_results$validation$f1_score,
    noise_results$validation$f1_score
  ),
  TestF1 = c(
    original_results$test$f1_score,
    rose_results$test$f1_score,
    oversample_results$test$f1_score,
    noise_results$test$f1_score
  ),
  TrainingR2 = c(
    NA, # We don't calculate this for training
    NA,
    NA,
    NA
  ),
  ValidationR2 = c(
    original_results$validation$r_squared,
    rose_results$validation$r_squared,
    oversample_results$validation$r_squared,
    noise_results$validation$r_squared
  ),
  TestR2 = c(
    original_results$test$r_squared,
    rose_results$test$r_squared,
    oversample_results$test$r_squared,
    noise_results$test$r_squared
  )
)

# Find the best model based on validation accuracy
best_idx <- which.max(results_df$ValidationAccuracy)
best_dataset_type <- results_df$DatasetType[best_idx]
best_model <- switch(best_dataset_type,
                     "Original" = original_results$model,
                     "ROSE_Synthetic" = rose_results$model,
                     "Oversampling" = oversample_results$model,
                     "Noise_Added" = noise_results$model)

cat("\n============ BEST MODEL ============\n")
cat("Best Dataset Type:", best_dataset_type, "\n")
cat("Best Validation Accuracy:", round(results_df$ValidationAccuracy[best_idx] * 100, 2), "%\n")
cat("Best Test Accuracy:", round(results_df$TestAccuracy[best_idx] * 100, 2), "%\n")

# Plot the best decision tree model
pdf("best_decision_tree_plot.pdf", width = 12, height = 8)
rpart.plot(best_model, extra = 101, box.palette = "RdBu", shadow.col = "gray")
dev.off()

# Function to safely write CSV data
safe_write_csv <- function(df, filepath, append = FALSE) {
  tryCatch({
    if (!append || !file.exists(filepath)) {
      # If the file doesn't exist or not appending, write with headers
      write.csv(df, filepath, row.names = FALSE)
      cat("\nSuccessfully created new results file:", filepath, "\n")
      return(TRUE)
    } else {
      # If the file exists and we're appending, append without headers
      write.table(df, filepath, append = TRUE, sep = ",", 
                  row.names = FALSE, col.names = FALSE)
      cat("\nSuccessfully appended results to existing file:", filepath, "\n")
      return(TRUE)
    }
  }, error = function(e) {
    cat("\nError writing to", filepath, ":", e$message, "\n")
    return(FALSE)
  })
}

# Define the output file name
results_file <- "titanic_dt_regression_performance.csv"

# Try to write results
write_success <- safe_write_csv(results_df, results_file, file.exists(results_file))

# If failed, try temp directory
if (!write_success) {
  temp_file <- file.path(tempdir(), results_file)
  write_success <- safe_write_csv(results_df, temp_file, file.exists(temp_file))
  if (write_success) {
    cat("Results saved to temporary location:", temp_file, "\n")
  }
}

# Print the data that was saved
cat("\nResults Summary:\n")
print(results_df)

# Save the best model
saveRDS(best_model, paste0("titanic_decision_tree_", gsub("_", "", tolower(best_dataset_type)), ".rds"))