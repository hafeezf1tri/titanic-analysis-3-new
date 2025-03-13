
packages <- c("ggplot2", "dplyr")
new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
if(length(new_packages)) install.packages(new_packages)

# Load the necessary libraries
library(ggplot2)
library(dplyr)

# 📌 Step 2: Load the Cleaned Dataset
titanic_cleaned <- read.csv("Titanic_Cleaned.csv")

# 📌 Step 3: Define Numeric Columns to Check for Outliers
numeric_columns <- c("Age", "SibSp", "PrCh", "Fare")

# 📌 Step 4: Create Boxplots for Each Numeric Feature
# Use ggplot2 to visualize outliers in each numeric column
par(mfrow=c(2,2))  # Arrange plots in 2x2 grid
for (col in numeric_columns) {
  boxplot(titanic_cleaned[[col]], main=paste("Boxplot of", col), col="gold")
}

# 📌 Step 5: Reset Plot Layout
par(mfrow=c(1,1))
