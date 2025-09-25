# Predictive-Modeling-of-Medical-Insurance-Charges

## Table of contents 
- [Data Set](#data-set)
- [Introduction](#introduction)
- [Exploratory data analysis (EDA) and pre-processing](#exploratory-data-analysis-(EDA)-and-pre-processing)
- [Model fitting](#model-fitting)
  - [Model 1 - Linear Model](#model-1-linear-model)
  - [Model 2 - GAM](#model-2-gam)
  - [Model 3 - Improved GAM](#model-3-improved-gam)
  - [Model 4 - Regression Tree](#model-4-regression-tree)
  - [Model 5 - Random Forest](#model-5-random-forest)
 - [Model Selection](#model-selection)
 - [Model evaluation](#model-evaluation)
 - [Suggestions for improvement](#suggestions-for-improvement)

## Data Set
The dataset ‘medical_insurance.csv’ consists of 1338 observations alongside 7 variables. The dependent variable ‘charges’ holds a quantitative nature, being recognized as a continuous variable, allowing us to advance using a regression model. 
Description of variables:
- age: Age of the individual (in years).
- sex: Gender of the individual (male/female).
- bmi: Body Mass Index, calculated as weight in kg divided by the square of height in meters.
- children: Number of children/dependents covered by the insurance plan.
- smoker: Whether the individual is a smoker (yes/no).
- region: The geographical region of the individual (e.g., southeast, northwest).
- charges: The medical insurance cost is billed to the individual (in dollars).

# Introduction
This model aims to predict medical insurance charges, taking into account potential factors that may influence the varying costs of medical care. The predictive power derived from the models will prove crucial to medical insurance providers in characterizing optimal resource allocation and developing effective pricing strategies.

# Exploratory data analysis (EDA) and pre-processing
To explore the dataset, I will create a correlogram - examining the prevalent trends and relationships amongst the variables, leading the primary exploration of data wrangling.

```{r}
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
```

We can deduce that the numerical predictors, ‘age’ (0.299) and ‘bmi’ (0.198) show greater correlation with the outcome variable ‘charges’ . This is indicative of the complementary increase in the volume of medical care along its charges that accompany growing age and bmi. Amongst categorical variables, ‘smoker’ predicts individuals who smoke to be more susceptible to higher medical chargers. The boxplot supports this correlation, where smokers displays a higher mean and range of charges compared to non-smokers, making clear the effect of smoking history on medical insurance prices. The correlogram also display faint signs of collinearity between ‘ages’ and ‘bmi’, holding a relatively low correlational strength of 0.109 supported by a 0.05 significance. The weak relationship limits its ability to disturb the combinatorial significance of the predictors, expunging the need to address it at this stage.

# Model fitting 
To initiate, I will split the observations into a train-test ratio of 80/20 to strengthen the predictive power of the model I want to design. Doing so will allow the model to gain higher generalizability to unforeseen data, employing the portion reserved for testing as a real-world proxy.

```{r}
#test-train split 
set.seed(123)
trainIndex <- createDataPartition(data$charges, p=0.8, list=FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]
```

Prior to forming my models, I will employ the subset selection technique on the training dataset, identifying the predictors that will gather the highest predictive ability towards medical insurance ‘charges’. The dataset follows a low-dimensional setting and is not subjected to a long array of complex predictors, authorizing my choice of approach for variable selection.

```{r}
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
```

The corresponding C_p and BIC select the model with 4 predictors to be the best fit model for this dataset. BIC selects the model with 4 predictors to show the lowest BIC score, prioritizing goodness of fit and simplicity. This is because the BIC statistic tends to behave in a stricter manner, attaching heavier penalty on models with several variables and higher complexity. C_p on the other hand, assumes linearity, which restricts the scope of the models I can form to best predict ‘charges’. For this reason, I will follow the suggestion made by R^2, choosing to obtain higher explanatory power over parsimony, selecting the 6 predictors evident in table 1 to be my key variables.

## Model 1 - Linear Model
```{r}
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
```

Now fitting my models, my primary design will be a linear model, combining the predictive power of the 6 variables previously chosen. The model produces a MSE test score of 29930845 suggesting a linear model may not be a good fit for the data. This poses me to question the assumed linearity within the dataset, and to consider alternative  approaches like the General Additive Model (GAM).

## Model 2 - GAM
```{r}
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
```

The GAM can address covert complexities and non-linear predictive inconsistencies through the inclusion of splines. To form my model, I clubbed the continuous variables to run under smoothing splines, automatically assigning knots to limit the possibility of inaccurate input. Additionally, I converted ‘children’ to a factor variable to capture any non-linear trend of marginal increase in charges for children, enhancing flexibility, interpretability, and accuracy. The GAM plot with a MSE of 38917346, show slight improvements from the linear model, confirming the presence of non-linearity.

The ‘bmi’ in the GAM plot display greater complexity, somewhat mimicking an upside-down u-shape that limits interpretability. For this reason, I will transform ‘bmi’ into a categorical variable using the standard clinical classifications established by WHO. Additionally, I will adopt feature engineering to include an interaction term for bmi_category*smoker, targeting the unexplained effect that may stem from the large coefficient observed from ‘smoker’. This term can account for overlooked interactions like higher medical charges that appear amongst obese individuals who smoke, risking cardiovascular disease, respiratory problems, and other comorbidities compared to obese non-smokers or normal-weight smokers.

## Model 3 - Improved GAM
```{r}
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
```

The improved GAM generates a MSE of 22574364, showing significant improvement from my first model illustrating better goodness of fit.

My next model is a regression tree that prioritize simplicity and interpretability. It is primarily concerned with the variables ‘smoker’, ‘age’, and ‘bmi’ that hold leading coefficients as evident in Figure 6; overlooking other variables that may hold similar significance.

## Model 4 - Regression Tree
```{r}
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
```

The regression tree already presents itself to be simple enough, inserting pruning into the equation may not prove necessary. The MSE of this model amounts to 28267864 which is still higher than our previous GAM model. This motivates me to consider an alternative approach that may offer a better goodness of fit.

My final design will be the Random Forest model, suitable for datasets with multiple predictors, welcoming complexity. It will address the shortcomings rooted in the regression tree model, targeting oversimplification and capturing potential interactions between the variables at hand to improve predictive power.

## Model 5 - Random Forest 
```{r}
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
```

The Random Forest plot ranks the significant variables in terms of its importance to the model’s predictive accuracy, extending the hierarchy established by the regression tree and generating its test MSE to be 23755135.

# Model Selection
```{r}
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
```

# Model evaluation
```{r}
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
```

For the final selection of my model, I will rely on MSE to evaluate how well each model performs on predicting insurance charges. GAM2 seems to perform best, holding the lowest MSE of 22574364, and RMSE of 4751.249. Simultaneously, GAM2 displays low residuals, maintaining predictive accuracy consistently, even along higher charges where the other models cannot keep up. Although Random Forest hold the highest Adjusted R-squared, because of the model’s flexible nature (ensemble of trees), it may reflect signs of overfitting, averting our focus away from R-squared and towards MSE, RMSE and cross-validation.

My final model – GAM2 remains significant, explaining 87.4% of the variations in charges maintaining the goodness of fit of the model. Additionally, I executed k-fold cross validation (cv) (10 folds) to determine how well it performs on independent data. I choose this approach over LOOCV to evade the higher variance that accompanies LOOCV’s test errors. Table 5 (green) show the cv’s preference for the FALSE model, basing it on the lower RMSE – 4422.170. The GAM fitted to the training dataset is slightly more flexible in comparison and can capture additional noise specific to the training dataset. This may not generalize well to the testing data set and subsequently display signs of overfitting as seen with the cv displaying lower RMSE and R-squared in comparison to the GAM based on the training dataset.


# Suggestions for improvement
The models were designed with priority for explanatory power over parsimony. I assumed preference for greater accuracy over interpretability to critically account for all possible attributes that can impact medical charges and vary on an individual level. Although higher variability is captured, my models may overexert on flexibility and consequently succumb to overfitting. In that case, the incorporation of lasso or ridge regularization may help correct for this overfitting. The challenges toward the interpretability of the statistics may further present as an obstacle in correctly establishing pricing strategies and resource allocation.

While GAM2 performs relatively well, addressing the signs of overfitting will amplify the model’s predictive ability. It will prove beneficial to incorporate cross-validation at the stage prior to fitting the model, identifying the optimal number of spline terms that maximize accuracy.

Overall, the models may not be performing as well as it should in predicting charges across varying levels since it is based on a skewed dataset. ‘Charges’ reflects this in Figure 1 as the data displays presence of skewness towards the right that limit sufficient data to feed predictions for higher charges. The GAM and Random Forest approaches this non-linearity effectively but may benefit from a log transformation attachment to the outcome variable, ‘charges’ to help stabilize the variance and improve each model’s fit.
