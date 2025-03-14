# Install required packages if not already installed
if (!require("rpart")) install.packages("rpart")
if (!require("rpart.plot")) install.packages("rpart.plot")
if (!require("caret")) install.packages("caret")
if (!require("dplyr")) install.packages("dplyr")

# Load required libraries
library(rpart)
library(rpart.plot)
library(caret)
library(dplyr)

# Function to read and preprocess data
preprocess_data <- function(file_path) {
  # Read the Titanic dataset
  titanic_data <- read.csv(file_path)
  
  # Display basic information about the dataset
  cat("Dataset dimensions:", dim(titanic_data), "\n")
  cat("\nFirst few rows of the dataset:\n")
  print(head(titanic_data))
  
  # Convert 'Sex' to a factor variable
  titanic_data$Sex <- as.factor(titanic_data$Sex)
  
  # Convert 'Pclass' to a factor (since it's categorical)
  titanic_data$Pclass <- as.factor(titanic_data$Pclass)
  
  # For regression, keep Survived as numeric (0 or 1)
  if (is.factor(titanic_data$Survived)) {
    titanic_data$Survived <- as.numeric(as.character(titanic_data$Survived))
  }
  
  return(titanic_data)
}

# Function to split data into train, validation, and test sets
split_data <- function(data, train_ratio = 0.8, val_ratio = 0.1) {
  set.seed(42) # For reproducibility
  
  # Calculate the sample sizes
  n <- nrow(data)
  train_size <- floor(train_ratio * n)
  val_size <- floor(val_ratio * n)
  test_size <- n - train_size - val_size
  
  # Create indices for the different sets
  indices <- sample(1:n, n)
  train_indices <- indices[1:train_size]
  val_indices <- indices[(train_size + 1):(train_size + val_size)]
  test_indices <- indices[(train_size + val_size + 1):n]
  
  # Split the data
  train_data <- data[train_indices, ]
  val_data <- data[val_indices, ]
  test_data <- data[test_indices, ]
  
  # Verify the split sizes
  cat("\nSplit sizes:\n")
  cat("Training set:", nrow(train_data), "observations (", round(nrow(train_data)/n * 100, 2), "%)\n")
  cat("Validation set:", nrow(val_data), "observations (", round(nrow(val_data)/n * 100, 2), "%)\n")
  cat("Test set:", nrow(test_data), "observations (", round(nrow(test_data)/n * 100, 2), "%)\n")
  
  return(list(train = train_data, val = val_data, test = test_data))
}

# Function to train decision trees with different complexity parameters
train_decision_trees <- function(train_data, val_data, cp_values = seq(0.001, 0.05, by = 0.001)) {
  results <- data.frame(
    cp = numeric(),
    train_rmse = numeric(),
    val_rmse = numeric(),
    train_accuracy = numeric(),
    val_accuracy = numeric()
  )
  
  models <- list()
  
  for (cp in cp_values) {
    # Train model with current complexity parameter
    model <- rpart(Survived ~ ., 
                   data = train_data, 
                   method = "anova",  # Use anova for regression
                   control = rpart.control(cp = cp))
    
    # Store the model
    models[[as.character(cp)]] <- model
    
    # Make predictions
    train_pred <- predict(model, train_data)
    val_pred <- predict(model, val_data)
    
    # Binary predictions for accuracy
    train_bin_pred <- ifelse(train_pred >= 0.5, 1, 0)
    val_bin_pred <- ifelse(val_pred >= 0.5, 1, 0)
    
    # Calculate metrics
    train_rmse <- sqrt(mean((train_pred - train_data$Survived)^2))
    val_rmse <- sqrt(mean((val_pred - val_data$Survived)^2))
    
    train_accuracy <- mean(train_bin_pred == train_data$Survived)
    val_accuracy <- mean(val_bin_pred == val_data$Survived)
    
    # Store results
    results <- rbind(results, data.frame(
      cp = cp,
      train_rmse = train_rmse,
      val_rmse = val_rmse,
      train_accuracy = train_accuracy,
      val_accuracy = val_accuracy
    ))
  }
  
  return(list(results = results, models = models))
}

# Function to find optimal CP without plotting
find_optimal_cp <- function(results) {
  # Find optimal CP value
  optimal_row <- results[which.min(results$val_rmse), ]
  cat("\nOptimal CP value:", optimal_row$cp, "\n")
  cat("Validation RMSE:", optimal_row$val_rmse, "\n")
  cat("Validation Accuracy:", optimal_row$val_accuracy, "\n")
  
  return(optimal_row$cp)
}

# Function to visualize the decision tree
visualize_tree <- function(model, file_name = "decision_tree_plot.png") {
  # Save the plot to a file
  png(file_name, width = 1600, height = 1200, res = 150)
  
  # Create a more detailed and visually appealing tree visualization
  rpart.plot(model, 
             type = 4,              # Use type 4 for more detailed node information
             extra = 101,           # Show standard info for regression models
             box.palette = "GnBu",  # Blue color palette (compatible with regression)
             shadow.col = "gray80", # Subtle shadow
             nn = TRUE,             # Show node numbers
             fallen.leaves = TRUE,  # Align all leaves at bottom
             branch = 0.5,          # Branch shape parameter
             round = 0,             # Round corners of the boxes
             leaf.round = 9,        # Round corners for leaf nodes
             space = 0.2,           # More space between nodes
             branch.lwd = 1.5,      # Line width for branches
             split.cex = 1.2,       # Size of split text
             split.prefix = "➤ ",   # Split text prefix
             split.suffix = "",     # Split text suffix
             under = TRUE,          # Place split text under the box
             under.cex = 1,         # Size of text under the box
             compress = TRUE,       # Compress the tree
             family = "sans",       # Font family
             main = "Decision Tree for Titanic Survival Prediction")
  
  dev.off()
  
  # Also create an SVG version for better quality
  svg(gsub("\\.png$", ".svg", file_name), width = 12, height = 9)
  rpart.plot(model, 
             type = 4,
             extra = 101,          # Changed from 106 to 101 for regression models
             box.palette = "GnBu",  # Using standard color palette for regression
             shadow.col = "gray80",
             nn = TRUE,
             fallen.leaves = TRUE,
             branch = 0.5,
             round = 0,
             leaf.round = 9,
             space = 0.2,
             branch.lwd = 1.5,
             split.cex = 1.2,
             split.prefix = "➤ ",
             split.suffix = "",
             under = TRUE,
             under.cex = 1,
             compress = TRUE,
             family = "sans",
             main = "Decision Tree for Titanic Survival Prediction")
  dev.off()
  
  cat("\nDecision tree visualizations saved as", file_name, "and", gsub("\\.png$", ".svg", file_name), "\n")
}

# Analyze feature importance without plotting
analyze_features <- function(model) {
  # Variable importance
  var_importance <- model$variable.importance
  
  if (length(var_importance) > 0) {
    var_importance_df <- data.frame(
      Feature = names(var_importance),
      Importance = var_importance
    )
    var_importance_df <- var_importance_df[order(-var_importance_df$Importance), ]
    
    cat("\nVariable Importance:\n")
    print(var_importance_df)
    
    return(var_importance_df)
  } else {
    cat("\nNo variable importance available for this model\n")
    return(NULL)
  }
}

# Function to evaluate model performance
evaluate_model <- function(model, data, dataset_name) {
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
  cat("\n=== ", dataset_name, " Set Evaluation ===\n")
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

# Function to track changes in validation performance over time
track_validation_performance <- function(results_file) {
  if (file.exists(results_file)) {
    # Read historical results
    historical_results <- read.csv(results_file)
    
    if (nrow(historical_results) > 1) {
      # Sort by timestamp
      historical_results$Timestamp <- as.POSIXct(historical_results$Timestamp)
      historical_results <- historical_results[order(historical_results$Timestamp), ]
      
      # Calculate changes in validation metrics
      historical_results$ValidationAccuracyChange <- c(NA, diff(historical_results$ValidationAccuracy) * 100)
      historical_results$ValidationRMSEChange <- c(NA, diff(historical_results$ValidationRMSE))
      
      # Display trends
      cat("\n=== Validation Performance Trends ===\n")
      cat("Total model runs tracked:", nrow(historical_results), "\n\n")
      
      # Create a simplified trend view for the last few runs
      n_display <- min(5, nrow(historical_results))
      recent_runs <- tail(historical_results, n_display)
      
      # Format and display the validation accuracy trend
      cat("Recent Validation Accuracy Trend:\n")
      for (i in 1:nrow(recent_runs)) {
        run_info <- recent_runs[i, ]
        accuracy_pct <- round(run_info$ValidationAccuracy * 100, 2)
        
        # Add trend indicator for all except the first displayed run
        if (i > 1 || nrow(historical_results) > n_display) {
          accuracy_change <- run_info$ValidationAccuracyChange
          if (is.na(accuracy_change)) {
            trend <- "   "
          } else if (accuracy_change > 0) {
            trend <- sprintf(" ↑ +%.2f%%", accuracy_change)
          } else if (accuracy_change < 0) {
            trend <- sprintf(" ↓ %.2f%%", accuracy_change)
          } else {
            trend <- " ↔ 0.00%"
          }
        } else {
          trend <- " (baseline)"
        }
        
        cat(sprintf("Run %s: %.2f%%%s\n", 
                    format(run_info$Timestamp, "%Y-%m-%d %H:%M:%S"),
                    accuracy_pct, 
                    trend))
      }
      
      # Format and display the validation RMSE trend
      cat("\nRecent Validation RMSE Trend:\n")
      for (i in 1:nrow(recent_runs)) {
        run_info <- recent_runs[i, ]
        
        # Add trend indicator for all except the first displayed run
        if (i > 1 || nrow(historical_results) > n_display) {
          rmse_change <- run_info$ValidationRMSEChange
          if (is.na(rmse_change)) {
            trend <- "   "
          } else if (rmse_change < 0) {
            trend <- sprintf(" ↑ %.4f", rmse_change)  # Lower RMSE is better, so arrow up
          } else if (rmse_change > 0) {
            trend <- sprintf(" ↓ +%.4f", rmse_change)  # Higher RMSE is worse, so arrow down
          } else {
            trend <- " ↔ 0.0000"
          }
        } else {
          trend <- " (baseline)"
        }
        
        cat(sprintf("Run %s: %.4f%s\n", 
                    format(run_info$Timestamp, "%Y-%m-%d %H:%M:%S"),
                    run_info$ValidationRMSE, 
                    trend))
      }
      
      # Overall trend assessment
      first_accuracy <- historical_results$ValidationAccuracy[1]
      last_accuracy <- tail(historical_results$ValidationAccuracy, 1)
      overall_change <- (last_accuracy - first_accuracy) * 100
      
      cat("\nOverall trend from first to most recent run:\n")
      if (overall_change > 0) {
        cat(sprintf("Validation accuracy IMPROVED by %.2f percentage points\n", overall_change))
      } else if (overall_change < 0) {
        cat(sprintf("Validation accuracy DEGRADED by %.2f percentage points\n", abs(overall_change)))
      } else {
        cat("Validation accuracy remained UNCHANGED\n")
      }
      
      return(recent_runs)
    } else {
      cat("\nOnly one run found in history. Need more runs to track performance changes.\n")
      return(NULL)
    }
  } else {
    cat("\nNo historical data found. This is the first run.\n")
    return(NULL)
  }
}

# Main function to run the entire workflow
run_titanic_analysis <- function(file_path = "Titanic_Cleaned.csv") {
  cat("=== Starting Titanic Survival Prediction with Decision Tree Regression ===\n\n")
  
  # Preprocess data
  titanic_data <- preprocess_data(file_path)
  
  # Split data
  data_splits <- split_data(titanic_data)
  
  # Iterate through different complexity parameters to find the best model
  cat("\n=== Training Decision Trees with Different Complexity Parameters ===\n")
  cp_results <- train_decision_trees(data_splits$train, data_splits$val)
  
  # Find the optimal CP without plotting
  cat("\n=== Finding Optimal Complexity Parameter ===\n")
  optimal_cp <- find_optimal_cp(cp_results$results)
  
  # Train the final model with the optimal CP value
  cat("\n=== Training Final Model with Optimal CP ===\n")
  final_model <- rpart(Survived ~ ., 
                       data = data_splits$train, 
                       method = "anova",
                       control = rpart.control(cp = optimal_cp))
  
  # Print model summary
  cat("\nFinal Decision Tree Model Summary:\n")
  print(final_model)
  
  # Visualize the decision tree
  visualize_tree(final_model)
  
  # Analyze feature importance
  feature_importance <- analyze_features(final_model)
  
  # Evaluate the model on all datasets
  cat("\n=== Evaluating Model Performance ===\n")
  train_eval <- evaluate_model(final_model, data_splits$train, "Training")
  val_eval <- evaluate_model(final_model, data_splits$val, "Validation")
  test_eval <- evaluate_model(final_model, data_splits$test, "Test")
  
  # Summary of model performance
  cat("\n=== Summary of Decision Tree Regression Performance ===\n")
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
    ModelType = "Decision Tree Regression",
    OptimalCP = optimal_cp,
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
    TestF1 = test_eval$f1_score
  )
  
  # Define the output file name
  results_file <- "titanic_dt_regression_performance.csv"
  
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
  
  # Track validation performance over time
  cat("\n=== Tracking Validation Performance Over Time ===\n")
  performance_trend <- track_validation_performance(results_file)
  
  # Save the model
  model_file <- "titanic_decision_tree_reg_model.rds"
  saveRDS(final_model, model_file)
  cat("\nSaved model to file:", model_file, "\n")
  
  cat("\n=== Analysis Complete ===\n")
  
  return(list(
    model = final_model,
    results = results_df,
    feature_importance = feature_importance,
    data_splits = data_splits,
    evaluations = list(
      train = train_eval,
      val = val_eval,
      test = test_eval
    )
  ))
}

# Run the full analysis
results <- run_titanic_analysis("Titanic_Cleaned.csv")