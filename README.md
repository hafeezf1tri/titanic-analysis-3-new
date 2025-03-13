# Titanic Survival Analysis

## Project Overview
This project analyzes the famous Titanic dataset to predict passenger survival using a decision tree regression model. The analysis includes data preprocessing, model training, and evaluation to understand which factors influenced survival rates during the Titanic disaster.

## Dataset Description
The dataset contains information about 714 Titanic passengers with the following variables:

- **Survived**: Survival status (0 = No, 1 = Yes) - This is our target variable
- **Pclass**: Passenger class (1 = First, 2 = Second, 3 = Third)
- **Sex**: Gender of the passenger (male/female)
- **Age**: Age of the passenger
- **SibSp**: Number of siblings/spouses aboard
- **PrCh**: Number of parents/children aboard
- **Fare**: Fare paid for the ticket

## Project Structure
```
├── Titanic_Cleaned.csv                      # Cleaned dataset
├── decision_tree_regression_plot.pdf        # Visualization of the decision tree
├── titanic_decision_tree_reg_model.rds      # Saved model for future use
├── titanic_dt_regression_performance.csv    # Performance metrics log
└── Dataset Preprocessing.txt                # Data preprocessing documentation
```

## Data Preprocessing
The following preprocessing steps were applied to prepare the dataset:
1. Handling missing values
2. Removing duplicates
3. Converting data types
4. Detecting and removing outliers
5. Min-Max normalization
6. Splitting the dataset into train (80%), validation (10%), and test (10%) sets

## Model Development
A decision tree regression model was developed using the R programming language with the following libraries:
- rpart
- rpart.plot
- caret
- dplyr
- e1071

### Model Parameters
- Method: anova (for regression)
- Minimum split: 20
- Minimum bucket: 7
- Maximum depth: 30
- Complexity parameter: 0.01

## Results and Evaluation
The model performance is evaluated using multiple metrics:

### Classification Metrics (using 0.5 threshold)
- Accuracy
- Sensitivity/Recall
- Specificity
- Precision
- F1 Score
- Confusion Matrix

### Regression Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared

### Feature Importance
The model identifies the most important features that influence survival prediction.

## How to Run the Analysis
1. Install R and the required packages:
```R
install.packages(c("rpart", "rpart.plot", "caret", "dplyr", "e1071"))
```

2. Place the Titanic_Cleaned.csv file in your working directory

3. Run the R script:
```R
source("titanic_survival_analysis.R")
```

## Future Work
- Experiment with different machine learning algorithms (Random Forest, SVM, etc.)
- Engineer additional features from the existing data
- Perform hyperparameter tuning to improve model performance
- Visualize survival patterns based on different features

## References
- Original Titanic dataset: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
