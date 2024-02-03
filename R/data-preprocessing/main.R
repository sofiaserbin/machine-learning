# data preprocessing
# import a library for splitting data sets into test and training sets
install.packages("caTools")
library(caTools)
# importing the dataset
dataset <- read.csv("C:/Users/Acer/Python/machine-learning/machine-learning/data-preprocessing/Data.csv")
# indices start from 1, not from 0
# taking care of missing data
dataset$Age <- ifelse(is.na(dataset$Age), ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)), dataset$Age)
dataset$Salary <- ifelse(is.na(dataset$Salary), ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)), dataset$Salary)
# encoding categorical data
# c is vector
dataset$Country <- factor(dataset$Country, levels = c("France", "Spain", "Germany"), labels = c(1, 2, 3))
dataset$Purchased <- factor(dataset$Purchased, levels = c("Yes", "No"), labels = c(1, 0))
# splitting the data set into training and test sets
set.seed(123)
# the percentage here is for the training set
split <- sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)
# feature scaling
# not applied for factors
training_set[, 2:3] <- scale(training_set[, 2:3])
test_set[, 2:3] <- scale(test_set[, 2:3])
print(training_set)
print(test_set)