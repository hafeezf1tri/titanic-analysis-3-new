# ============================================
# Titanic Dataset: Plot Independent Variables vs. Survived (with Value Ranges)
# ============================================

# Install required packages if not already installed
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(reshape2)) install.packages("reshape2")
if (!require(dplyr)) install.packages("dplyr")

# Load libraries
library(ggplot2)
library(reshape2)
library(dplyr)

# ============================================
# 1. Load the Dataset
# ============================================
# Ensure that Titanic_processed.csv is in your working directory
titanic_data <- read.csv("Titanic_processed.csv")

# ============================================
# 2. Data Preprocessing
# ============================================
# Remove 'No.' column if it exists
if ("No." %in% colnames(titanic_data)) {
  titanic_data <- titanic_data %>% select(-No.)
}

# Convert categorical variables to factors
titanic_data$Pclass <- as.factor(titanic_data$Pclass)
titanic_data$Sex <- as.factor(titanic_data$Sex)
titanic_data$Survived <- as.factor(titanic_data$Survived)

# ============================================
# 3. Apply Value Ranges (Filtering)
# ============================================
# Limit Age from 1 to 80
titanic_data <- subset(titanic_data, Age >= 1 & Age <= 80)

# Limit Fare from 1 to 600
titanic_data <- subset(titanic_data, Fare >= 1 & Fare <= 600)

# ============================================
# 4. Reshape Data for Plotting
# ============================================
# Reshape dataset into long format using melt()
titanic_melted <- melt(titanic_data, id.vars = "Survived", 
                       measure.vars = c("Pclass", "Sex", "Age", "SibSp", "PrCh", "Fare"))

# ============================================
# 5. Plot Independent Variables vs. Survived
# ============================================
# Facet plot to compare all independent variables in one figure
plot <- ggplot(titanic_melted, aes(x = value, y = as.factor(Survived))) +
  geom_jitter(alpha = 0.6, color = "blue", width = 0.3, height = 0.2) + # Scatter plot with jitter
  facet_wrap(~ variable, scales = "free_x") +                          # Separate plots for each variable
  labs(title = "Independent Variables vs. Survived (Age: 1-80, Fare: 1-600)", 
       x = "Independent Variables", 
       y = "Survived") +
  theme_minimal()

# Display the plot
print(plot)

# ============================================
# 6. Save Plot as Image
# ============================================
# Save the plot as a PNG image in the working directory
ggsave("Titanic_Combined_ScatterPlot_Range_Age1-80_Fare1-600.png", 
       plot = plot, width = 12, height = 8, dpi = 300)
