#--------------- set up --------------------------------------
#install required packages
install.packages("tidyverse")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("GGally")
install.packages("corrplot")
install.packages("caret")
install.packages("MASS")
install.packages("pROC")
library(tidyverse)
library(dplyr)
library(ggplot2)
library(GGally)
library(corrplot)
library(caret)
library(MASS)
library(pROC)


#load the dataset
data <- read_csv("medical_insurance.csv")
View(data)

#-------------------------------------------------------------
####EDA: initial data exploration 
summary(data)
str(data)

#convert non-ordinal data into factor variables 
data$sex <- as.factor(data$sex)
data$smoker <- as.factor(data$smoker)
data$region <- as.factor(data$region)

#clean up dataset 
colSums(is.na(data)) #no NA values observed

#Visualisation
#correlogram
ggpairs(data)

#zoomed-in plots
#how medical charges vary between smoker and non-smoker 
ggplot(data, aes(x=factor(smoker), y=charges)) +
  geom_boxplot() + ggtitle("Charges Distribution: smoker vs non-smoker")

#--------------------------------------------------------------------
####test-train split 
set.seed(123)
trainIndex <- createDataPartition(data$charges, p=0.8, list=FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

#--------------------------------------------------------------------
#determine variable selection to best fit the model
#subset selection
install.packages("leaps")
library(leaps)
regfit.full <- regsubsets(charges ~ ., data = train_data, nvmax = 12)
summary(regfit.full)

reg.summary <- summary(regfit.full)
names(reg.summary)
t(t(sprintf("%0.2f%%", reg.summary$rsq * 100)))

#line plots (performance metrics) for model selection
par(mfrow=c(2,2))
plot(reg.summary$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")

#Plot C_p 
plot(reg.summary$cp, xlab = "Number of variables", ylab = "C_p", type = "l")
points(which.min(reg.summary$cp), reg.summary$cp[which.min(reg.summary$cp)], col = "red", cex = 2, pch = 20)

#Plot BIC
plot(reg.summary$bic, xlab = "Number of variables", ylab = "BIC", type = "l")
points(which.min(reg.summary$bic), reg.summary$bic[which.min(reg.summary$bic)], col = "red", cex = 2, pch = 20)

#Plot R^2
plot(reg.summary$adjr2, xlab = "Number of variables", ylab = "Adjusted R^2", type = "l")
points(which.max(reg.summary$adjr2), reg.summary$adjr2[which.max(reg.summary$adjr2)], col = "red", cex = 2, pch = 20)

coef(regfit.full, 6)

#-----------------------------------------------------------------------
####Model formation
##(1) linear model
model1 <- lm(charges ~ age + bmi + region + smoker + children, data=train_data)
summary(model1)

# Predictions and MSE Calculation
pred1 <- predict(model1, newdata=test_data)
mse1 <- mean((test_data$charges - pred1)^2)
mse1 #39570140 
rmse1 <- sqrt(mse1) #use rmse for interpretability 
rmse1 #model is off by approx $6290 (6290.48) per person

#range to aid intepretation of rmse
range(test_data$charges)
pred1 <- predict(model1, newdata=test_data)
residuals1 <- test_data$charges - pred1
range(residuals1)

##(2) Generalised Additive Model (GAM) 
library(mgcv)
model_gam <- gam(charges ~ smoker + region + as.factor(children) + s(age) + s(bmi),
                 data = train_data)
summary(model_gam)

#plot GAM
par(mfrow = c(2, 3))
plot(model_gam, page = 1)
# Plot the contributions.
par( mfrow = c(2,3) )
plot( model_gam,  se = TRUE, col = "blue" )
# Define the y-axis label
y.lab <- "Charges (dollars)"
# Compare with the following plots.
plot( train_data$age, train_data$charges, pch = 16, col = 2, 
      ylab = y.lab, xlab = "Age (years)" )
plot( train_data$bmi, train_data$charges, pch = 16, col = 2, 
      ylab = y.lab, xlab = "BMI" )
plot( train_data$children, train_data$charges, pch = 16, col = 2, 
      ylab = y.lab, xlab = "number of children" )
plot( train_data$region, train_data$charges, pch = 16, col = 2, 
      ylab = y.lab, xlab = "region" )
plot( train_data$smoker, train_data$charges, pch = 16, col = 2, 
      ylab = y.lab, xlab = "smoker" )

# Predictions and MSE Calculation
pred_gam <- predict(model_gam, newdata=test_data)
mse_gam <- mean((test_data$charges - pred_gam)^2)
mse_gam #38917346
rmse_gam <- sqrt(mse_gam) #use rmse for interpretability 
rmse_gam #6238.377


##(3) GAM 2
#convert bmi into a categorical variable for interpretability of GAM plot
train_data$bmi_category <- cut(train_data$bmi, breaks = c(0, 18.5, 25, 30, Inf),
                               labels = c("underweight", "normal", "overweight", "obese"))
test_data$bmi_category <- cut(test_data$bmi, breaks = c(0, 18.5, 25, 30, Inf),
                              labels = c("underweight", "normal", "overweight", "obese"))

#Fit GAM 2
model_gam2 <- gam(charges ~ smoker + region + as.factor(children) + s(age) + bmi_category + smoker*bmi_category,
                  data = train_data)
summary(model_gam2)

#plot GAM
par(mfrow = c(2, 3))
plot(model_gam2, page = 1)
# Plot the contributions.
par( mfrow = c(2,3) )
plot( model_gam2,  se = TRUE, col = "blue" )
# Define the y-axis label
y.lab <- "Charges (dollars)"
# Compare with the following plots.
plot( train_data$age, train_data$charges, pch = 16, col = 2, 
      ylab = y.lab, xlab = "Age (years)" )
plot( train_data$bmi_category, train_data$charges, pch = 16, col = 2, 
      ylab = y.lab, xlab = "BMI" )
plot( train_data$children, train_data$charges, pch = 16, col = 2, 
      ylab = y.lab, xlab = "number of children" )
plot( train_data$region, train_data$charges, pch = 16, col = 2, 
      ylab = y.lab, xlab = "region" )
plot( train_data$smoker, train_data$charges, pch = 16, col = 2, 
      ylab = y.lab, xlab = "smoker" )

# Predictions and MSE Calculation
pred_gam2 <- predict(model_gam2, newdata=test_data)
mse_gam2 <- mean((test_data$charges - pred_gam2)^2)
mse_gam2 #22574364
rmse_gam2 <- sqrt(mse_gam2) #use rmse for interpretability 
rmse_gam2 #4751.249


##(4) regression tree
if (!require(randomForest)) install.packages("randomForest", dependencies=TRUE)
if (!require(gbm)) install.packages("gbm", dependencies=TRUE)
if (!require(tree)) install.packages("tree", dependencies=TRUE)
if (!require(ggplot2)) install.packages("ggplot2", dependencies=TRUE)
if (!require(caret)) install.packages("caret", dependencies=TRUE)
# Load libraries
library(randomForest)
library(gbm)
library(tree)
library(ggplot2)
library(caret)

# Fit a regression Tree
tree_model <- tree(charges ~ age + children + region + bmi + smoker, data = train_data)
summary(tree_model)

# Plot the tree
plot(tree_model)
text(tree_model, pretty = 0, cex = 0.7)

# Predict on test data
tree_pred <- predict(tree_model, newdata = test_data)
# Compute Mean Squared Error (MSE)
tree_mse <- mean((test_data$charges - tree_pred)^2)
tree_mse #28267864
#regression tree RMSE
tree_rmse <- sqrt(tree_mse) #use rmse for interpretability 
tree_rmse #3928.961

#Generate adjusted R-squared
predicted <- predict(tree_model, train_data)
actual <- train_data$charges

sst <- sum((actual - mean(actual))^2)  # Total Sum of Squares
ssr <- sum((actual - predicted)^2)     # Sum of Squared Residuals
r_squared <- 1 - (ssr / sst)

n <- nrow(train_data)  
p <- 3     
adjusted_r_squared <- 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

cat("Adjusted R^2:", adjusted_r_squared, "\n")


##(5) random forest 
install.packages("randomForest")
library(randomForest)
# Fit a Random Forest model
rf_model <- randomForest(charges ~ age + smoker + children + bmi + region, data = train_data, mtry = 3, ntree = 500, importance = TRUE)
summary(rf_model)

# Predict on test data
rf_pred <- predict(rf_model, newdata = test_data)

# Compute MSE and RMSE
rf_mse <- mean((test_data$charges - rf_pred)^2)
rf_mse #24028256
rf_rmse <- sqrt(rf_mse) #use rmse for interpretability 
rf_rmse #4901.863

#Generate adjusted R-squared
rf_pred_train <- predict(rf_model, newdata = train_data)
actual_train <- train_data$charges

sst <- sum((actual_train - mean(actual_train))^2)

ssr <- sum((actual_train - rf_pred_train)^2)

r_squared <- 1 - (ssr / sst)

n <- nrow(train_data)  
p <- 5                
adjusted_r_squared <- 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

cat("Adjusted R^2:", adjusted_r_squared, "\n")

# Plot variable importance
varImpPlot(rf_model)

#----------------------------------------------------------------------
####Model selection
#model comparison: predicted vs actual charges
# Assuming you have predictions in a data frame like this:
install.packages("tidyr")
library(tidyr)
model_predictions <- data.frame(
  actual = test_data$charges,
  linear_regression = predict(model1, newdata = test_data),
  gam = predict(model_gam2, newdata = test_data),
  random_forest = predict(rf_model, newdata = test_data),
  decision_tree = predict(tree_model, newdata = test_data)
) %>%
  pivot_longer(cols = -actual, names_to = "model", values_to = "predicted")

ggplot(model_predictions, aes(x = actual, y = predicted, color = model)) +
  geom_point(alpha = 0.5) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black") +
  labs(title = "Model Comparison: Predicted vs. Actual Charges",
       x = "Actual Charges",
       y = "Predicted Charges",
       color = "Model") +
  theme_minimal()

# Create a table comparing MSE and RMSE
results <- rbind(
  model1 = c(MSE= mse1, RMSE = rmse1),
  GAM = c(MSE = mse_gam, RMSE = rmse_gam),
  GAM2 = c(MSE= mse_gam2, RMSE = rmse_gam2),
  Regression_Tree = c(MSE = tree_mse, RMSE = tree_rmse),
  Random_Forest = c(MSE = rf_mse, RMSE = rf_rmse)
)

# Print the table
print(results)

#-----------------------------------------------------------------------
####Model evaluation
#10 fold k-cross-validation
set.seed(123)
train_control <- trainControl(method="cv", number=10, savePredictions=TRUE)
#reiterate bmi_category and as.factor(children) for the entire dataset
data$bmi_category <- cut(data$bmi, breaks = c(0, 18.5, 25, 30, Inf),
                         labels = c("underweight", "normal", "overweight", "obese"))
data$children <- as.factor(data$children)
# cross-validation for final GAM (2) model - simplified for caret
tune_grid <- expand.grid(select = c(TRUE, FALSE), method = "REML")
cv_gam2 <- train(
  charges ~ smoker + region + children + age + bmi_category + smoker * bmi_category,
  data = data,
  method = "gam",
  trControl = train_control,
  tuneGrid = tune_grid,
  family = gaussian()
)
print(cv_gam2)