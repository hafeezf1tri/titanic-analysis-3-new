# Load Titanic dataset (replace 'Titanic.csv' with your actual file path)
df <- read.csv("Titanic_processed.csv")

# Filter out zero values for Age and Fare
df_filtered <- df %>%
  filter(Age > 0, Fare > 0)  # Exclude zero or negative values

# Find the lowest and highest age
lowest_age <- min(df_filtered$Age, na.rm = TRUE)   # Minimum age, ignoring NA values
highest_age <- max(df_filtered$Age, na.rm = TRUE)  # Maximum age, ignoring NA values

# Find the lowest and highest fare
lowest_fare <- min(df_filtered$Fare, na.rm = TRUE)  # Minimum fare, ignoring NA values
highest_fare <- max(df_filtered$Fare, na.rm = TRUE) # Maximum fare, ignoring NA values

# Print results
cat("Lowest Age (excluding zero):", lowest_age, "\n")
cat("Highest Age:", highest_age, "\n")
cat("Lowest Fare (excluding zero):", lowest_fare, "\n")
cat("Highest Fare:", highest_fare, "\n")
