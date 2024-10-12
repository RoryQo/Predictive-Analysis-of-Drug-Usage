## ----setup, include=FALSE------------------------------------------
knitr::opts_chunk$set(echo = TRUE, cache=T, message=F)


## ----message=FALSE,warning=FALSE-----------------------------------
library(NHANES)
library(tidyverse)
library(rpart)
library(rpart.plot) 
library(randomForest)



## ------------------------------------------------------------------

# Select columns from NHANES data
# remove na values
data2<- NHANES %>%
  select(Gender,Age,Race1,Education,MaritalStatus,HHIncomeMid,Poverty,HomeOwn,Weight,Height,
         BMI,Pulse,BPSysAve,BPDiaAve,Diabetes,HealthGen,DaysPhysHlthBad,DaysMentHlthBad,
         Depressed,SleepHrsNight,SleepTrouble,AlcoholDay,Smoke100,Marijuana,HardDrugs) %>% 
  drop_na()


set.seed(100)     # For reproducibility

# Select 80% of data to train model with
train2 <- data2 %>% sample_frac(size = 0.8, fac=HardDrugs)
# Select the rest to test the model with
test2 <- data2 %>% setdiff(train2)


## ----warning=FALSE,message=FALSE-----------------------------------
library(rpart)
library(rpart.plot)

# Specify target (y) and features (x) for model
form_full<- as.formula(HardDrugs~Gender+Age+Race1+Education+MaritalStatus+HHIncomeMid+Poverty+HomeOwn+Weight+Height+BMI+Pulse+BPSysAve+BPDiaAve+Diabetes+HealthGen+DaysPhysHlthBad+DaysMentHlthBad+Depressed+SleepHrsNight+SleepTrouble+AlcoholDay+Smoke100+Marijuana)

# Create tree using formula above and training data
mod_tree <- rpart(form_full,data=train2)


## ------------------------------------------------------------------
library(randomForest)

# Create random forest with formula and training set with 1000 trees
mod_rf<- randomForest(form_full,train2,ntree=1000)
mod_rf


## ------------------------------------------------------------------

# Create function to make a confusion matrix 
# create new columns in data with predictions from model
# select actual target values (y) and predicted y values
# Table them
confusion_matrix <- function(data,y,mod){
  confusion_matrix <- data %>% 
  mutate(pred = predict(mod, newdata = data, type = "class"),
         y=y) %>%
  select(y,pred) %>% table()
}

# 1- sum of the diagonals of the confusion matrix leaves those that were misclassified
# Divide them by sum of total confusion matrix
# Return this value
misclass <- function(confusion){
  misclass <- 1- sum(diag(confusion))/sum(confusion)
  return(misclass)
}


## ------------------------------------------------------------------

# Create confusion matrix with function above from mod_rf
# Hard Drugs is target variable (y)
confusion.rf<-confusion_matrix(test2,test2$HardDrugs,mod_rf)


# Create misclassification rate with function from above
misclass(confusion.rf) 



## ------------------------------------------------------------------
library(e1071) 

# Create naives bayes model, train with training data and target Hard drugs
NBMod<- naiveBayes(HardDrugs~ ., data=train2) 

# Use model to predict y values from test data
NBY<-predict(NBMod, newdata=test2)

# Create confusion matrix for Naive bayes model
NBConfusion<-confusion_matrix(test2,test2$HardDrugs,NBMod) 
NBConfusion

# Create misclassification rate for naive bayes model
misclass(NBConfusion)


## ------------------------------------------------------------------
library(glmnet)


# calculate usage rate for hard drugs
table(train2$HardDrugs)/length(train2$HardDrugs)

# transform data set to matrix suitable for next model
predictors <- model.matrix(form_full, data = train2) 

# Create cross fold validation for predictors and training set y values
cv.fit <- cv.glmnet(predictors, train2$HardDrugs, family = "binomial", type = "class") 

# Save optimal lambda to use in model from cv.fit
# lambda is penalty strength
lambda_opt=cv.fit$lambda.1se 

# Create regularized regression (l2) model 
mod_lr2 <- glmnet(predictors, train2$HardDrugs, family = "binomial", lambda = lambda_opt)


# Save predicted y from lr model above
y_lr = predict(mod_lr2, newx = model.matrix(form_full, data = test2), type = "class")

# Create confusion matrix
confusion_lr = table(test2$HardDrugs, y_lr) 
confusion_lr

# Specificity rate
# True positive rate
tpr_lr = confusion_lr[2,2]/sum(confusion_lr[2,]); tpr_lr

# Sensitivity rate
# True negative rate
tnr_lr = confusion_lr[1,1]/sum(confusion_lr[1,]); tnr_lr


## ------------------------------------------------------------------
library(ROCR)

# Create function to return false positive and true positive rate as a data frame
roc_data <- function(test,y_test,model,type){
  prob = model %>% 
    predict(newdata=test, type=type) %>% 
    as.data.frame()
  pred_prob = prediction(prob[,2], y_test)
  perf = performance(pred_prob, 'tpr', 'fpr')
  perf_df = data.frame(perf@x.values, perf@y.values)
  names(perf_df)=c('fpr','tpr')
  return(perf_df)
}

# Create function to return point data 
# Returns false and true positive rate
point_data <- function(test,y_test,model,type){
  y_pred = predict(model, newdata=test,type=type)
  confusion_matrix = table(y_test, y_pred)
  tpr = confusion_matrix['Yes','Yes']/sum(confusion_matrix['Yes',])
  fpr = confusion_matrix['No','Yes']/sum(confusion_matrix['No',])
  return(c(fpr,tpr))
}

# Use functions for random forest model data
perf_df_rf = roc_data(test2, test2$HardDrugs, mod_rf, "prob") 

point_rf = point_data(test2, test2$HardDrugs, mod_rf, "class")


# Predict y values with naives bayes model as a dataframe
# calculate performance for naive bayes model
prob_nb <- NBMod %>% predict(newdata=test2,type="raw") %>% as.data.frame() 
pred_nb <- predict(NBMod, newdata = test2)
pred_nb_prob <- prediction(prob_nb[,2],test2$HardDrugs) 
perf_nb <- performance(pred_nb_prob,'tpr','fpr') 
perf_df_nb <- data.frame(perf_nb@x.values, perf_nb@y.values)

# Create confusion matrix of naive bayes model
names(perf_df_nb) <- c("fpr", "tpr")
confusion_nb <- table(pred_nb,test2$HardDrugs)


# Predict y values with tree model as a dataframe
# calculate performance tree model
prob_tree <- mod_tree %>% predict(newdata=test2,type="matrix") %>% as.data.frame() 
pred_tree <- predict(mod_tree, newdata = test2) 

pred_tree_prob <- prediction(prob_tree[,2],test2$HardDrugs) 

perf_tree <- performance(pred_tree_prob,'tpr','fpr') 

perf_df_tree <- data.frame(perf_tree@x.values, perf_tree@y.values) 
names(perf_df_tree) <- c("fpr", "tpr")


# Plot graph
ggplot(data =perf_df_nb, aes(x=fpr, y=tpr))+ 
  geom_line(data= perf_df_rf, color="purple",lwd=1)+         geom_line(color="orange", lwd= 1)+ geom_line(data=perf_df_tree, color= "blue", lwd=1)+ geom_point(x=point_rf[1],y=point_rf[2],size=3,col="red")+ geom_point(x=0.17, y= 0.475,size=3, col="red")+ labs(x='False Positive Rate', y='True Positive Rate')


## ------------------------------------------------------------------
library(caret) 

# Train control with 5 fold cross validation twice
control <- trainControl(method="repeatedcv", number=5, repeats=2, search="grid") 
set.seed(100) 

# Find optimal model
tunegrid <- expand.grid(.mtry=seq(2,20,2)) 
rf_gridsearch <- train(HardDrugs~., data=train2, method="rf",
metric="Accuracy", tuneGrid=tunegrid, trControl=control) 


print(rf_gridsearch)


## ------------------------------------------------------------------

# Create optimum random forest model
mod_rf_opt<- randomForest(form_full,train2,ntree=1000, mtry= 10) 

# Create confusion matrix for optimum RF model
confusion.rf.opt<-confusion_matrix(test2,test2$HardDrugs,mod_rf_opt) 
# Calculate misclass rate for optimum rf model
Miscalassification<-misclass(confusion.rf.opt) 
Miscalassification


## ------------------------------------------------------------------
# Create RF model
mod_rf_6<- randomForest(form_full,train2,ntree=1000, mtry= 6) 

# Creat Confusion matrix
confusion.rf.6<-confusion_matrix(test2,test2$HardDrugs,mod_rf_6) 

# Create misclass rate
misclass(confusion.rf.6)

