
packages <- c("dplyr")
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

# Load the necessary library
library(dplyr)

#Step 2: Load Dataset
titanic_data <- read.csv("Titanic_processed.csv")

# Remove unnecessary column (Passenger No.)
titanic_data$No. <- NULL  

#Step 3: Function to Replace Outliers with Median Values
replace_outliers_with_median <- function(column) {
  Q1 <- quantile(column, 0.25, na.rm = TRUE)  # First quartile (25th percentile)
  Q3 <- quantile(column, 0.75, na.rm = TRUE)  # Third quartile (75th percentile)
  IQR_value <- Q3 - Q1  # Interquartile Range
  
  # Define lower and upper bounds
  lower_bound <- Q1 - 1.5 * IQR_value
  upper_bound <- Q3 + 1.5 * IQR_value
  
  # Replace outliers with median
  column[column < lower_bound | column > upper_bound] <- median(column, na.rm = TRUE)
  return(column)
}

#Step 4: Apply the Function to Numeric Columns with Outliers
numeric_columns <- c("Age", "SibSp", "PrCh", "Fare")

for (col in numeric_columns) {
  titanic_data[[col]] <- replace_outliers_with_median(titanic_data[[col]])
}

#Step 5: Save the Cleaned Dataset
write.csv(titanic_data, "Titanic_Cleaned.csv", row.names = FALSE)

# Step 6: Confirm Changes
summary(titanic_data)  # Check if outliers are handled
