# Load necessary library
library(dplyr)

# Load Titanic dataset (replace 'Titanic.csv' with your actual file path)
df <- read.csv("Titanic.csv")

# Round Age to the nearest whole number
df <- df %>%
  mutate(Age = round(Age, 0))  # Round 0.5 and above to the next whole number

# Floor Fare to remove decimal points
df <- df %>%
  mutate(Fare = floor(Fare))

# Save the processed dataset
write.csv(df, "Titanic_processed.csv", row.names = FALSE)

# Display first few rows of the dataset
head(df)
