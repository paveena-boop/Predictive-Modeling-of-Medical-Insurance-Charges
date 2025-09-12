# Predictive-Modeling-of-Medical-Insurance-Charges

# Table of contents 
- [Introduction](#introduction)
- [Exploratory data analysis (EDA) and pre-processing](#exploratory-data-analysis-(EDA)-and-pre-processing)

# Introduction
This model aims to predict medical insurance charges, taking into account potential factors that may influence the varying costs of medical care. The predictive power derived from the models will prove crucial to medical insurance providers in characterizing optimal resource allocation and developing effective pricing strategies.

# Exploratory data analysis (EDA) and pre-processing
The dataset ‘medical_insurance.csv’ consists of 1338 observations alongside 7 variables that are descriptively listed in the Appendix 1.1. The dependent variable ‘charges’ holds a quantitative nature, being recognized as a continuous variable, allowing us to advance using a regression model. To explore the dataset, I will create a correlogram - examining the prevalent trends and relationships amongst the variables, leading the primary exploration of data wrangling.

## 📊 Interactive Analysis Report

[**View Complete HTML Report**](medical_insurance.html) ← Click here for the full interactive analysis!

## 🔍 Quick Preview

### Code Example: Data Exploration
```r
# From medical_insurance.Rmd
summary(insurance)
cat("Dataset dimensions:", dim(insurance))
```

### Key Findings
- **Strongest predictors**: Age (r=0.299) and BMI (r=0.198)
- **Smokers pay 3-4x more** than non-smokers
- **Best model**: GAM2 with RMSE of $4,751.25

### Sample Visualization
![Age vs Charges Relationship](plots/age_vs_charges.png)

## 📁 Files
- [`medical_insurance.Rmd`](medical_insurance.Rmd) - Source code
- [`medical_insurance.html`](medical_insurance.html) - Interactive report ← **NEW**
- [`medical_insurance.csv`](medical_insurance.csv) - Dataset

## 🛠️ How to Reproduce
```r
# Install required packages
install.packages(c("tidyverse", "ggplot2", "caret", "mgcv", "randomForest"))

# Render the report
rmarkdown::render("medical_insurance.Rmd")
```
