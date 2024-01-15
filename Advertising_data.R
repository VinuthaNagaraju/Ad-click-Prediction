#importing libraries
require(ggplot2)
library(dplyr)
library(corrplot)
library(gridExtra)
library(caret)
library(pROC)
library(xgboost)
library(PRROC)
install.packages("pROC")
library(pROC)

#Loading the data
data <- read.csv("Ad_click_data.csv")

#Exploratory data analysis
View(data)
str(data)

#display rows and columns
dim(data)

# Summary statistics for each variable
summary(data)

#identifying numerical and categorical columns in the dataset
numerical_columns <- data[, sapply(data, is.numeric)]
head(numerical_columns)
categorical_columns <- data[, sapply(data, function(x) is.factor(x) | is.character(x))]
head(categorical_columns)

#data preprocessing
# Check if there are any missing values in the entire data frame
if (any(is.na(data))) {
  print("There are missing values in the data frame.")
} else {
  print("No missing values found.")
}
#count total missing values in each column
sapply(data, function(x) sum(is.na(x)))

# Find duplicate records based on all columns
duplicates <- data[duplicated(data) | duplicated(data, fromLast = TRUE), ]
# Display duplicate records
print(duplicates)

#removing ID column from the data and making target variable "Clicked" as factor which is in int
data <- data %>% select(-ID)
data$Clicked <- as.factor(data$Clicked)

# Plot histograms for all numeric columns to understand distribution
numerical_columns1 <- data[, sapply(data, is.numeric)]
par(mfrow = c(ceiling(sqrt(ncol(numerical_columns1))), ceiling(sqrt(ncol(numerical_columns1)))))
for (col in colnames(numerical_columns1)) {
  hist(numerical_columns1[[col]], main = paste("Histogram of", col), xlab = col, col = "skyblue", border = "black")
}

# Create a count plot for the 'Male' 
ggplot(data, aes(y = Male, fill = Male)) +
  geom_bar() +
  labs(title = "Distribution of Gender", y = "Male") +
  scale_fill_manual(values = c("No" = "skyblue", "Yes" = "salmon")) +  # Specify colors
  theme_minimal()

# count plot for the 'Clicked' variable 
ggplot(data, aes(y = Clicked, fill = factor(Clicked))) +
  geom_bar() +
  labs(title = "Distribution of Clicked on Ad", y = "Clicked") +
  scale_fill_manual(values = c("0" = "lightgreen", "1" = "tomato")) +  # Specify colors
  theme_minimal()

#analysing on how the user clicks based on the time they spend on site and the internet usage 
#  plot for 'Daily Time Spent on Site'
plot1 <- ggplot(data, aes(x = Time_Spent, fill = Clicked)) +
  geom_histogram(binwidth = 20, color = 'white', position = 'identity', alpha = 0.7) +
  labs(title = 'Daily Time Spent on Site', x = 'Daily Time Spent on Site', y = 'Count') +
  scale_fill_manual(values = c("0" = "#3498db", "1" = "#e74c3c")) +
  theme_minimal()
# plot for 'Daily Internet Usage'
plot2 <- ggplot(data, aes(x = Internet_Usage, fill = Clicked)) +
  geom_histogram(binwidth = 20, color = 'white', position = 'identity', alpha = 0.7) +
  labs(title = 'Daily Internet Usage', x = 'Daily Internet Usage', y = 'Count') +
  scale_fill_manual(values = c("0" = "#3498db", "1" = "#e74c3c")) +
  theme_minimal()
grid.arrange(plot1, plot2, ncol = 2)

#analysing on how age influences the ad-click
ggplot(data, aes(x = Age, fill = Clicked)) +
  geom_histogram(binwidth = 5, color = 'white', position = 'identity', alpha = 0.7) +
  labs(title = 'Age v/s ad-clicked', x = 'Age', y = 'Count') +
  scale_fill_manual(values = c("0" = "#3498db", "1" = "#e74c3c"))+
theme_minimal()

#Click through rate analysis
#calculating CTR for different AD-topic we have and visualizing the top 10 
ctr_by_product <- data %>%
  group_by(Ad_Topic) %>%
  summarise(CTR = mean(as.numeric(Clicked == 1), na.rm = TRUE)) %>%
  top_n(10, wt = CTR) %>%
  arrange(desc(CTR)) 

# Print the top 10 CTR products
print(ctr_by_product)

#bar plot for the top 10 CTR products
ggplot(ctr_by_product, aes(x = Ad_Topic, y = CTR, fill = Ad_Topic)) +
  geom_bar(stat = "identity") +
  labs(title = "Top 10 Click-Through Rate by Ad-Topics", x = "Product", y = "Click-Through Rate") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#Identifying outliers using boxplots
numerical_columns1 <- data[, sapply(data, is.numeric)]
# Set up the layout
par(mfrow = c(3, 3), mar = c(4, 4, 2, 1))

# Create boxplots for each numerical column
for (col in names(numerical_columns)) {
  boxplot(numerical_columns[[col]], main = paste("Boxplot of", col), col = "skyblue", border = "black", horizontal = TRUE)
}

# Observed outliers in the age column ,applying IQR method  
Q1 <- quantile(data$Age, 0.25)
Q3 <- quantile(data$Age, 0.75)
IQR_value <- Q3 - Q1
#Define a threshold for outliers 
threshold <- 1.5 * IQR_value
outliers_low <- data$Age < Q1 - threshold
outliers_high <- data$Age > Q3 + threshold

# Replace outliers with Winsorizing technique which caps at a specified percentile
data$Age[outliers_low] <- Q1 - threshold  
data$Age[outliers_high] <- Q3 + threshold 
# Create a boxplot using ggplot to visualize "Age" variable after treating outliers
ggplot(data, aes(y = Age)) +
  geom_boxplot() +
  labs(title = "Boxplot of Age", y = "Age")
#we can see that there are no outliers 

#year has a constant value ,so removing it from the dataframe 
data <- data %>% select(-Year)

# plotting correlation matrix to check multi collinearity
cor_matrix <- cor(data[, c('Time_Spent', 'Age', 'Avg_Income', 'Internet_Usage')])
# Display the correlation matrix
print(cor_matrix)


#Encoding categorical columns 
# Nominal encoding for Ad_Topic
data$Ad_Topic <- as.factor(data$Ad_Topic)

# Nominal encoding for Country_Name 
data$Country_Name <- as.factor(data$Country_Name)
unique_countries_count <- length(unique(data$Country_Name))
print(unique_countries_count)
#we have data from all the countries

# Nominal encoding for Time_Period, Weekday, and Month
data$Time_Period <- as.factor(data$Time_Period)
data$Weekday <- as.factor(data$Weekday)
data$Month <- as.factor(data$Month)
data$Male <- as.factor(data$Male)

summary(data)


#Modeling
# Setting  seed  value for reproducibility
set.seed(1000)

#Splitting data into train(80 %) and test(20 % data)
split_indices <- createDataPartition(data$Ad_Topic, p = 0.8, list = FALSE)
train_data <- data[split_indices, ]
test_data <- data[-split_indices, ]

#applying logistic regression model
logistic_model <- glm(Clicked ~ Time_Spent + Age + Avg_Income + Internet_Usage +Male + Time_Period +Country_Name + Weekday + Month + Ad_Topic,
                      data = train_data, family = binomial)
summary(logistic_model)
data <- subset(data, select = -Country_Name)

# Setting  seed  value for reproducibility
set.seed(1000)

#Splitting data into train(80 %) and test(20 % data)
split_indices1 <- createDataPartition(data$Ad_Topic, p = 0.8, list = FALSE)
train_data1 <- data[split_indices, ]
test_data1 <- data[-split_indices, ]

#applying logistic regression model
logistic_model1 <- glm(Clicked ~ Time_Spent + Age + Avg_Income + Internet_Usage +Male + Time_Period + Weekday + Month + Ad_Topic,
                      data = train_data1, family = binomial)
summary(logistic_model1)
#Evaluation
# Make predictions on the test set
predictions_LR <- predict(logistic_model1, newdata = test_data1, type = "response")

# Convert predicted probabilities to binary predictions (0 or 1)
predictions_LR <- ifelse(predictions_LR > 0.5, 1, 0)

# logistic regression accuracy
accuracy_LR <- sum(predictions_LR == test_data1$Clicked) / length(test_data1$Clicked)
cat("\nLogistic regression accuracy:", accuracy_LR)

# Converting Clicked to a factor with levels "0" and "1" in both predictions and test_data
predictions_LR <- factor(predictions_LR, levels = c("0", "1"))
test_data1$Clicked <- factor(test_data1$Clicked, levels = c("0", "1"))
predictions_LR <- factor(predictions_LR, levels = levels(test_data1$Clicked))
# Creating a confusion matrix
cf_matrix <- confusionMatrix(predictions_LR, test_data1$Clicked)

# Print classification report for the model
print(cf_matrix$byClass)

# Plot confusion matrix
par(mfrow = c(1, 1))
plot(cf_matrix$table, col = "blue", main = "Confusion Matrix", 
     cex.axis = 1.2, cex.main = 1.4, cex.lab = 1.2)

# printing the TN,FP,FN,TP for the model
TN <- cf_matrix$table[1, 1]
FP <- cf_matrix$table[1, 2]
FN <- cf_matrix$table[2, 1]
TP <- cf_matrix$table[2, 2]
cat("\nTrue Negatives (TN):", TN, "\n")
cat("False Positives (FP):", FP, "\n")
cat("False Negatives (FN):", FN, "\n")
cat("True Positives (TP):", TP, "\n")

# Plot ROC curve
par(mfrow = c(1, 1))
roc_curve <- roc(test_data1$Clicked, as.numeric(predictions_LR))
plot(roc_curve, main = "ROC Curve", col = "blue", lwd = 2)
auc_value <- auc(roc_curve)
text(x = 0.5, y = 1.0, labels = paste("AUC =", round(auc_value, 2)),
     col = "blue", cex = 1.2, pos = 3)





#CTR predictions for each ad-topic
#unique ad topics from the test data
unique_ad_topics <- levels(test_data1$Ad_Topic)

#  data frame to store CTR for each ad topic
ctr_results <- data.frame(Ad_Topic = character(), Predicted_CTR = numeric(), stringsAsFactors = FALSE)

# Iterate over unique ad topics
for (ad_topic in unique_ad_topics) {
    subset_data <- subset(test_data1, Ad_Topic == ad_topic)
  subset_features <- subset_data[, c('Time_Spent', 'Age', 'Avg_Income', 'Internet_Usage', 'Male', 'Time_Period', 'Weekday', 'Month', 'Ad_Topic')]
    predictions_LR_subset <- predict(logistic_model1, newdata = subset_features, type = "response")
  
  # Convert predicted probabilities to binary predictions (0 or 1)
  predictions_LR_subset <- ifelse(predictions_LR_subset > 0.5, 1, 0)
  
  # Calculate Predicted CTR for the current ad topic
  predicted_ctr_topic <- sum(predictions_LR_subset == 1) / length(predictions_LR_subset)
  
  # Print Predicted CTR for the current ad topic
  cat(paste("\nPredicted CTR for Ad Topic", ad_topic, ":", predicted_ctr_topic))
  
  # Store results in the data frame
  ctr_results <- rbind(ctr_results, data.frame(Ad_Topic = ad_topic, Predicted_CTR = predicted_ctr_topic, stringsAsFactors = FALSE))
}

# Sort the results by Predicted CTR in descending order
ctr_results <- ctr_results[order(-ctr_results$Predicted_CTR), ]

#top 10 ads with the highest Predicted CTRs
cat("\n\nTop 10 Ads with the Highest Predicted CTRs:")
print(head(ctr_results, 10))



