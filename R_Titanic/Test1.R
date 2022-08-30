rm(list=ls())
library(caret)
setwd('c:/CLEMSON/CITI/Workshop/4.Machine Learning/Kaggle/Titanic/')
training <- read.csv('data/train.csv',header = TRUE)
testing  <- read.csv('data/test.csv',header = TRUE) 
obs      <- read.csv('data/gender_submission.csv',header = TRUE) 

#Apply logistic regression
ModFit_log <- train(as.factor(Survived) ~ Pclass+Sex+SibSp+Parch+Fare,
                    data=training,method="rf")
Predictand <- predict(ModFit_log,testing)
confusionMatrix(Predictand,factor(obs$Survived[2:418]))
write.csv(Predictand,'Predictand_logistic.csv',row.names = FALSE)
