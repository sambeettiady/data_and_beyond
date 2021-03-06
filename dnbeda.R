#Remove previous objects
rm(list = ls())

#Set Working Directory
setwd(dir = '/home/sambeet/data/Data&Beyond/Data&Beyond/train_WG8MUe5/')

#Load libraries
library(dplyr)
library(ggplot2)
library(readr)
library(tidyr)
library(caret)
library(ROCR)
library(DMwR)
library(xgboost)
library(caretEnsemble)

#Function to create variables
create_new_vars = function(data_frame){
    data_frame = data_frame %>% 
        mutate(nbr_hist_tot_offer_extns = nbr_hist_supp_offer_extns + nbr_hist_elite_offer_extns + 
                   nbr_hist_credit_offer_extns,
               nbr_hist_tot_offer_acc = nbr_hist_supp_offer_acc + nbr_hist_elite_offer_acc + 
                   nbr_hist_credit_offer_acc,
               nbr_hist_supp_offer_acc_pct = ifelse(nbr_hist_supp_offer_extns == 0,0,
                                                    ifelse(nbr_hist_supp_offer_acc < nbr_hist_supp_offer_extns,
                                                           100*nbr_hist_supp_offer_acc/nbr_hist_supp_offer_extns,100)),
               nbr_hist_credit_offer_acc_pct = ifelse(nbr_hist_credit_offer_extns == 0,0,
                                                      ifelse(nbr_hist_credit_offer_acc < nbr_hist_credit_offer_extns,
                                                             100*nbr_hist_credit_offer_acc/nbr_hist_credit_offer_extns,100)),
               nbr_hist_elite_offer_acc_pct = ifelse(nbr_hist_elite_offer_extns == 0,0,
                                                     ifelse(nbr_hist_elite_offer_acc < nbr_hist_elite_offer_extns,
                                                            100*nbr_hist_elite_offer_acc/nbr_hist_elite_offer_extns,100)),
               nbr_hist_tot_offer_acc_pct = ifelse(nbr_hist_tot_offer_extns == 0,0,
                                                   ifelse(nbr_hist_tot_offer_acc < nbr_hist_tot_offer_extns,
                                                          100*nbr_hist_tot_offer_acc/nbr_hist_tot_offer_extns,100)),
               spend_capacity_per_mem = cust_spending_capacity/(1 + family_size),
               score_affinity_spend_per_mem = score_affinity_spend/(1 + family_size),
               score_business_spend_per_mem = score_affinity_business_spend/(1 + family_size),
               spend_capacity_na = ifelse(cust_spending_capacity == 0,1,0),
               income_na = ifelse(income == 0,1,0),
               family_size_more_than_2 = ifelse(family_size >= 3,1,0),
               spend_capacity_log1p = log1p(cust_spending_capacity),
               income_log1p = log1p(income),
               spend_ec_pct = ifelse(spend_total != 0,100*spend_ec/spend_total,0),
               spend_retail_pct = ifelse(spend_total != 0,100*spend_retail/spend_total,0),
               spend_household_pct = ifelse(spend_total != 0,100*spend_household/spend_total,0),
               spend_hotel_pct = ifelse(spend_total != 0,100*spend_hotel/spend_total,0),
               spend_car_pct = ifelse(spend_total != 0,100*spend_car/spend_total,0),
               total_memberships = nbr_clb_mem + nbr_air_miles_mem,
               nbr_yrs_acct_more_than_200 = ifelse(nbr_mths_acct >= 200,1,0),
               max_spent_ind_apparel_bool = ifelse(max_spent_ind == 'Apparel',1,0),
               income_per_mem = income/(1+family_size),
               income_log1p_per_mem = income_log1p/(1+family_size),
               spend_log1p_per_mem = spend_capacity_log1p/(1+family_size),
               cnt_payments_per_mem = cnt_payments_12m/(1+family_size),
               spend_total_log1p = log1p(spend_total))
    return(data_frame)
}

#Read data
data_dictionary = read_csv(file = 'data_dictionary.csv')
train_data = read_csv('train.csv')

#Create new variables
train_data = create_new_vars(train_data)
train_data$y = ifelse(test = train_data$card_accepted == 'Not Accepted',0,1)
train_data$accepted = factor(ifelse(train_data$y == 1,'acc','nacc'))
train_data$card_accepted[train_data$card_accepted == 'Not Accepted'] = 'None'
train_data$y_mult = factor(ifelse(train_data$card_accepted == 'None',0,
                                     ifelse(train_data$card_accepted == 'Supp',1,
                                            ifelse(train_data$card_accepted == 'Credit',2,3))))

#Split data into train and test
set.seed(34)
intrain = createDataPartition(train_data$y,times = 1,p = 0.5,list = F)
training_data = train_data[intrain,]
remaining_data = train_data[-intrain,]
set.seed(12)
intest = createDataPartition(remaining_data$y,times = 1,p = 0.8,list = F)
testing_data = remaining_data[intest,]
validation_data = remaining_data[-intest,]

#Stepwise backward algo for logistic regression
fullmodel = glm(formula = y~family_size+cust_spending_capacity+nbr_tot_cards+nbr_mths_acct+
                    mem_fee_24m+score_affinity_spend+internal_influencer_score+
                    elite_card_ind+cnt_payments_12m+nbr_clb_mem+nbr_air_miles_mem+
                    spend_ec_pct+spend_hotel_pct+spend_household_pct+spend_car_pct+spend_retail_pct+
                    nbr_hist_supp_offer_extns+nbr_hist_supp_offer_acc+
                    spend_log1p_per_mem+spend_capacity_na+income_na+
                    nbr_hist_tot_offer_acc_pct+spend_total_log1p+max_spent_ind_apparel_bool,
                data = train_data,family = 'binomial')
backwards = step(object = fullmodel)
summary(backwards)

#Define Cross validation method
set.seed(1234)
seeds = vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]] = sample.int(1000,1)
seeds[[31]] = sample.int(1000,1)
print(seeds)
logistic_ctrl <- trainControl(method="repeatedcv", repeats=3,seeds=seeds,allowParallel = F,number = 10)

#Train logistic model
log_reg_model <- train(factor(y) ~ nbr_tot_cards + nbr_mths_acct + mem_fee_24m + 
                           score_affinity_spend + internal_influencer_score + elite_card_ind + 
                           cnt_payments_12m + nbr_clb_mem + spend_car_pct + spend_retail_pct + 
                           nbr_hist_supp_offer_extns + nbr_hist_supp_offer_acc + spend_log1p_per_mem + 
                           spend_capacity_na + nbr_hist_tot_offer_acc_pct + spend_total_log1p + 
                           max_spent_ind_apparel_bool,data=training_data, 
                       trControl=logistic_ctrl, method='glm',family = 'binomial')

#Summarize results
print(log_reg_model$finalModel)
training_data$pred_prob = as.numeric(predict(object = log_reg_model$finalModel,newdata = training_data,type = 'response'))
validation_data$pred_prob = as.numeric(predict(object = log_reg_model$finalModel,newdata = validation_data,type = 'response'))
testing_data$pred_prob = as.numeric(predict(object = log_reg_model$finalModel,newdata = testing_data,type = 'response'))
train_data$pred_prob = as.numeric(predict(object = log_reg_model$finalModel,newdata = train_data,type = 'response'))

#Test predictions on testing data for optimal threshold
predictions = validation_data$pred_prob
labels = validation_data$y

for(i in 10:70){
    prob_threshold = i/100
    pred_labels = ifelse(predictions >= prob_threshold,1,0)
    table_temp = table(labels,pred_labels)
    print('Threshold:')
    print(prob_threshold)
    print('Price:')
    print((table_temp[3] + table_temp[4])*1500)
    print('Return:')
    print((table_temp[4]*9500) + (100000 - ((table_temp[3] + table_temp[4])*1500))*1.1)
    print('%Accepted:')
    print(table_temp[4]/(table_temp[3] + table_temp[4]))
}
#Optimal Threshold from train data = 0.26-0.28

#ROC Analysis
predictions = as.vector(validation_data$pred_prob)
pred = prediction(predictions,validation_data$y)

perf_AUC = performance(pred,"auc") #Calculate the AUC value
AUC = perf_AUC@y.values[[1]]

perf_ROC = performance(pred,"tpr","fpr") #plot the actual ROC curve
plot(perf_ROC, main="ROC plot")
text(0.5,0.5,paste("AUC = ",format(AUC, digits=5, scientific=FALSE)))
abline(a = 0,b = 1)

#Create training and testing data from Card Accepted Category
#training_data_accepted = train_data[train_data$card_accepted != 'None',]

#table(train_data_accepted$elite_card_ind,train_data_accepted$card_accepted)
#table(train_data_accepted$family_size,train_data_accepted$card_accepted)
#boxplot(train_data_accepted$spend_log1p_per_mem~train_data_accepted$card_accepted,outline = F)
#boxplot(train_data_accepted$mem_fee_24m~train_data_accepted$card_accepted,outline = F)
#boxplot(train_data_accepted$score_affinity_spend_per_mem~train_data_accepted$card_accepted,outline = F)
#boxplot(train_data_accepted$score_affinity_business_spend~train_data_accepted$card_accepted,outline = F)
#boxplot(train_data_accepted$spend_car_pct~train_data_accepted$card_accepted,outline = F)
#boxplot(train_data_accepted$spend_total_log1p~train_data_accepted$card_accepted,outline = F)
#boxplot(train_data_accepted$nbr_hist_supp_offer_extns~train_data_accepted$card_accepted,outline = F)
#boxplot(train_data_accepted$nbr_mths_acct~train_data_accepted$card_accepted,outline = F)

#normalize = function(x){return ((x - min(x)) / (max(x) - min(x)))}

#normalise_variables = function(data_frame){
#    knn_data = as.data.frame(lapply(
#        data_frame[c('family_size','spend_log1p_per_mem','elite_card_ind',
#                              'mem_fee_24m','score_affinity_spend_per_mem',
#                              'score_affinity_business_spend','spend_car_pct','nbr_mths_acct',
#                              'nbr_hist_supp_offer_extns','spend_total_log1p')],normalize))
#    return(knn_data)
#}

#knn_train_data_accepted = normalise_variables(training_data_accepted)
#knn_test_data_accepted = normalise_variables(training_data_accepted)

#XgBoost in Caret
set.seed(352)
seeds = vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]] = sample.int(10000,1)
seeds[[31]] = sample.int(1000,1)
print(seeds)
xgb_ctrl = trainControl(method = "repeatedcv",number = 10,repeats = 3,seeds = seeds,allowParallel = F,search = 'grid',
                        summaryFunction = multiClassSummary,classProbs = T,verboseIter = T,sampling = 'down')
set.seed(377)
tunegrid = expand.grid(nrounds = 1000,max_depth = 10,
                       eta = 0.001,
                       gamma = 2,
                       colsample_bytree = 0.5,
                       min_child_weight = 10,
                       subsample = 0.5)

xgb_gridsearch = train(form = card_accepted~family_size+spend_capacity_log1p+nbr_tot_cards+
                          nbr_mths_acct+mem_fee_24m+score_affinity_spend+income_log1p+
                          internal_influencer_score+elite_card_ind+score_affinity_business_spend+
                          cnt_payments_12m+spend_capacity_per_mem+
                          nbr_clb_mem+nbr_air_miles_mem+spend_ec_pct+spend_hotel_pct+
                          spend_household_pct+spend_car_pct+spend_retail_pct+
                          nbr_hist_supp_offer_extns+nbr_hist_supp_offer_acc+
                          spend_log1p_per_mem+spend_capacity_na+income_na+
                          nbr_hist_tot_offer_acc_pct+spend_total_log1p+pred_prob+
                          max_spent_ind_apparel_bool,data=training_data, method="xgbTree",
                      metric='AUC', trControl = xgb_ctrl,tuneGrid=tunegrid,max_delta_step=10,
                      objective = "multi:softprob",num_class = 4,tree_method='exact',eval_metric='auc')
print(xgb_gridsearch)
varImp(xgb_gridsearch,scale = T)
#print(xgb_gridsearch$finalModel)
#plot(xgb_gridsearch)
training_data$xgb_pred = predict(object = xgb_gridsearch,newdata = training_data)
print(confusionMatrix(training_data$xgb_pred,training_data$card_accepted))

validation_data$xgb_pred = predict(object = xgb_gridsearch,newdata = validation_data)
#testing_data$labels = 'None'
#testing_data$labels[testing_data$rf_pred$Supp >= 0.3] = 'Credit'
#testing_data$labels[testing_data$rf_pred$Credit >= 0.3] = 'Supp'
#testing_data$labels[testing_data$rf_pred$Elite >= 0.35] = 'Elite'
print(confusionMatrix(validation_data$xgb_pred,reference = validation_data$card_accepted))
#testing_data$rf_pred = as.character(predict(object = xgb_gridsearch,newdata = testing_data))
pred_probs = validation_data$pred_prob
actual_labels = validation_data$card_accepted
prob_threshold = 0
while(prob_threshold <= 1){
    pred_labels = as.character(validation_data$xgb_pred)
    pred_labels[pred_probs <= prob_threshold] = 'None'
    z = table(pred_labels)
    budget = z[['Credit']]*1800 + z[['Supp']]*1200 + z[['Elite']]*1500
    print(c(prob_threshold,budget))
    if(budget <= 400000){
        table_temp = table(actual_labels,pred_labels)
        print(table_temp[1]*9500 + table_temp[6]*11000 + table_temp[16]*8500 + (400000 - budget)*1.1)
        break
    }
    prob_threshold = prob_threshold + 0.001
}

#knn_accepted <- train(card_accepted~family_size+spend_log1p_per_mem+elite_card_ind+
#                     mem_fee_24m+score_affinity_spend_per_mem+score_affinity_business_spend+
#                     spend_car_pct+nbr_mths_acct+nbr_hist_supp_offer_extns+spend_total_log1p,
#                 data = training_data_accepted, method = "knn",
#                 trControl=knn_ctrl,
#                 preProcess = c('range'),
#                 tuneLength = 5)
#print(knn_accepted)
#plot(knn_accepted)
#Random Forest model with repeated cross-validation
library(doMC)
registerDoMC(cores = 7)
set.seed(3322)
seeds = vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]] = sample.int(1000,1)
seeds[[31]] = sample.int(1000,1)
print(seeds)
rf_ctrl = trainControl(method = "repeatedcv",number = 10,repeats = 3,seeds = seeds,verboseIter = T,
                       allowParallel = T,search = 'grid',summaryFunction = multiClassSummary,classProbs = T)
set.seed(3735)
tunegrid = expand.grid(.mtry = c(2))

rf_gridsearch = train(form = card_accepted~family_size+spend_capacity_log1p+nbr_tot_cards+
                          nbr_mths_acct+mem_fee_24m+score_affinity_spend+income_log1p+
                          internal_influencer_score+elite_card_ind+score_affinity_business_spend+
                          cnt_payments_12m+spend_capacity_per_mem+
                          nbr_clb_mem+nbr_air_miles_mem+spend_ec_pct+spend_hotel_pct+
                          spend_household_pct+spend_car_pct+spend_retail_pct+
                          nbr_hist_supp_offer_extns+nbr_hist_supp_offer_acc+
                          spend_log1p_per_mem+spend_capacity_na+income_na+
                          nbr_hist_tot_offer_acc_pct+spend_total_log1p+
                          max_spent_ind_apparel_bool+pred_prob,data=training_data, method="rf", 
                      metric='Kappa', trControl=rf_ctrl,ntree = 1000,tuneGrid=tunegrid,sampsize = c(270,270,450,270))
print(rf_gridsearch)
print(rf_gridsearch$finalModel)
#plot(rf_gridsearch)
print(confusionMatrix(testing_data$rf_pred,testing_data$card_accepted))
validation_data$rf_pred = predict(object = rf_gridsearch$finalModel,newdata = validation_data)
print(confusionMatrix(validation_data$rf_pred,validation_data$card_accepted))
validation_data$rf_pred = as.character(predict(object = rf_gridsearch$finalModel,newdata = validation_data))
pred_probs = validation_data$pred_prob
actual_labels = validation_data$card_accepted
prob_threshold = 0
while(prob_threshold <= 1){
    pred_labels = as.character(validation_data$rf_pred)
    pred_labels[pred_probs <= prob_threshold] = 'None'
    z = table(pred_labels)
    budget = z[['Credit']]*1800 + z[['Supp']]*1200 + z[['Elite']]*1500
    print(c(prob_threshold,budget))
    if(budget <= 400000){
        table_temp = table(actual_labels,pred_labels)
        print(table_temp[1]*9500 + table_temp[6]*11000 + table_temp[16]*8500 + (400000 - budget)*1.1)
        break
    }
    prob_threshold = prob_threshold + 0.001
}
varImp(object = rf_gridsearch)

#Ensemble of Random Forest, logistic and xgboost
training_data$rf_prob_credit = predict(rf_gridsearch,newdata = training_data,type = 'prob')[[1]]
training_data$rf_prob_elite = predict(rf_gridsearch,newdata = training_data,type = 'prob')[[2]]
training_data$rf_prob_supp = predict(rf_gridsearch,newdata = training_data,type = 'prob')[[4]]
training_data$xg_prob_credit = predict(xgb_gridsearch,newdata = training_data,type = 'prob')[1][[1]]
training_data$xg_prob_elite = predict(xgb_gridsearch,newdata = training_data,type = 'prob')[2][[1]]
training_data$xg_prob_supp = predict(xgb_gridsearch,newdata = training_data,type = 'prob')[4][[1]]
training_data$log_prob = predict(log_reg_model$finalModel,newdata = training_data,type = 'response')
training_data$rf_pred = predict(rf_gridsearch,newdata = training_data)
training_data$xgb_pred = predict(xgb_gridsearch,newdata = training_data)

testing_data$rf_prob_credit = predict(rf_gridsearch,newdata = testing_data,type = 'prob')[[1]]
testing_data$rf_prob_elite = predict(rf_gridsearch,newdata = testing_data,type = 'prob')[[2]]
testing_data$rf_prob_supp = predict(rf_gridsearch,newdata = testing_data,type = 'prob')[[4]]
testing_data$xg_prob_credit = predict(xgb_gridsearch,newdata = testing_data,type = 'prob')[1][[1]]
testing_data$xg_prob_elite = predict(xgb_gridsearch,newdata = testing_data,type = 'prob')[2][[1]]
testing_data$xg_prob_supp = predict(xgb_gridsearch,newdata = testing_data,type = 'prob')[4][[1]]
testing_data$log_prob = predict(log_reg_model$finalModel,newdata = testing_data,type = 'response')
testing_data$rf_pred = predict(rf_gridsearch,newdata = testing_data)
testing_data$xgb_pred = predict(xgb_gridsearch,newdata = testing_data)

validation_data$rf_prob_credit = predict(rf_gridsearch,newdata = validation_data,type = 'prob')[[1]]
validation_data$rf_prob_elite = predict(rf_gridsearch,newdata = validation_data,type = 'prob')[[2]]
validation_data$rf_prob_supp = predict(rf_gridsearch,newdata = validation_data,type = 'prob')[[4]]
validation_data$xg_prob_credit = predict(xgb_gridsearch,newdata = validation_data,type = 'prob')[1][[1]]
validation_data$xg_prob_elite = predict(xgb_gridsearch,newdata = validation_data,type = 'prob')[2][[1]]
validation_data$xg_prob_supp = predict(xgb_gridsearch,newdata = validation_data,type = 'prob')[4][[1]]
validation_data$log_prob = predict(log_reg_model$finalModel,newdata = validation_data,type = 'response')
validation_data$rf_pred = predict(rf_gridsearch,newdata = validation_data)
validation_data$xgb_pred = predict(xgb_gridsearch,newdata = validation_data)

results <- resamples(list(mod1 = xgb_gridsearch, mod2 = rf_gridsearch,mod3 = ensemble)) 
modelCor(results) 

set.seed(25)
seeds = vector(mode = "list", length = 2)
for(i in 1:1) seeds[[i]] = sample.int(1000,1)
seeds[[2]] = sample.int(1000,1)
print(seeds)
ensemble_ctrl = trainControl(method = "oob",allowParallel = T,search = 'grid',classProbs = T,verboseIter = T)
set.seed(315)
tunegrid = expand.grid(.mtry = c(1))
ensemble = train(form = card_accepted~rf_prob_credit+rf_prob_elite+rf_prob_supp+xg_prob_credit+xg_prob_elite+
                     xg_prob_supp+log_prob+rf_pred+xgb_pred,data=testing_data, method="rf", 
                      metric='Kappa', trControl=ensemble_ctrl,ntree = 5000,tuneGrid=tunegrid,
                 sampsize = c(250,250,340,250))
print(ensemble$finalModel)
#varImp(ensemble)
testing_data$ensemble_pred = as.character(predict(object = ensemble,newdata = testing_data))
print(confusionMatrix(reference = testing_data$card_accepted,testing_data$ensemble_pred))
validation_data$ensemble_pred = as.character(predict(object = ensemble,newdata = validation_data))
print(confusionMatrix(reference = validation_data$card_accepted,validation_data$ensemble_pred))

pred_probs = validation_data$pred_prob
actual_labels = validation_data$card_accepted
prob_threshold = 0
while(prob_threshold <= 1){
    pred_labels = as.character(validation_data$ensemble_pred)
    pred_labels[pred_probs <= prob_threshold] = 'None'
    z = table(pred_labels)
    budget = z[['Credit']]*1800 + z[['Supp']]*1200 + z[['Elite']]*1500
#    print(c(prob_threshold,budget))
    if(budget <= 400000){
        table_temp = table(actual_labels,pred_labels)
        print(table_temp[1]*9500 + table_temp[6]*11000 + table_temp[16]*8500 + (400000 - budget)*1.1)
        break
    }
    prob_threshold = prob_threshold + 0.001
}

validation_data$en_prob_credit = predict(ensemble,newdata = validation_data,type = 'prob')[1][[1]]
validation_data$en_prob_elite = predict(ensemble,newdata = validation_data,type = 'prob')[2][[1]]
validation_data$en_prob_supp = predict(ensemble,newdata = validation_data,type = 'prob')[4][[1]]

# validation_data$hueristic_pred = 'None'
# validation_data$hueristic_pred[validation_data$en_prob_supp >= 0.73] = 'Supp'
# validation_data$hueristic_pred[validation_data$en_prob_elite >= 0.87] = 'Elite'
# validation_data$hueristic_pred[validation_data$en_prob_credit >= 0.57] = 'Credit'
# table(actual =validation_data$card_accepted,predicted = validation_data$ensemble_pred)

pred_prob_supp = validation_data$en_prob_supp
pred_prob_elite = validation_data$en_prob_elite
pred_prob_credit = validation_data$en_prob_credit
actual_labels = validation_data$card_accepted
df = data.frame(prob_threshold = rep(NA,101),PPV_supp = rep(NA,101),PPV_elite = rep(NA,101),
                PPV_credit = rep(NA,101))
for(i in 0:100){
    pred_labels = validation_data$ensemble_pred
    prob_threshold = i/100
    df$prob_threshold[i+1] = prob_threshold
    pred_labels[pred_prob_elite <= prob_threshold & pred_labels == "Elite"] = 'None'
    pred_labels[pred_prob_supp <= prob_threshold & pred_labels == "Supp"] = 'None'
    pred_labels[pred_prob_credit <= prob_threshold & pred_labels == "Credit"] = 'None'
    z = table(pred_labels)
    df$elite_targeted[i+1] = ifelse("Elite" %in% names(z),(z[['Elite']]),0)
    df$credit_targeted[i+1] = ifelse("Credit" %in% names(z),(z[['Credit']]),0)
    df$supp_targeted[i+1] = ifelse("Supp" %in% names(z),(z[['Supp']]),0)
    df$PPV_elite[i+1] = ifelse("Elite" %in% names(z),sum((pred_labels == 'Elite') & (actual_labels == 'Elite'))/(z[['Elite']]),NA)
    df$PPV_supp[i+1] = ifelse("Supp" %in% names(z),sum((pred_labels == 'Supp') & (actual_labels == 'Supp'))/(z[['Supp']]),NA)
    df$PPV_credit[i+1] = ifelse("Credit" %in% names(z),sum((pred_labels == 'Credit') & (actual_labels == 'Credit'))/(z[['Credit']]),NA)
}
df$elite_roi = ((df$PPV_elite*11000)-1500)/1500
df$supp_roi = ((df$PPV_supp*8000)-1200)/1200
df$credit_roi = ((df$PPV_credit*9500)-1800)/1800
df$budget = df$elite_targeted*1500 + df$credit_targeted*1800 + df$supp_targeted*1200
    
g = ggplot(data = df,mapping = aes(x = prob_threshold)) + geom_line(mapping = aes(y = PPV_elite),col = 'red') + 
    geom_line(mapping = aes(y = PPV_supp),col = 'blue') + geom_line(mapping = aes(y = PPV_credit),col = 'green');g
# g = ggplot(data = df,mapping = aes(x = prob_threshold)) + geom_line(mapping = aes(y = elite_roi),col = 'red') + 
#     geom_line(mapping = aes(y = supp_roi),col = 'blue') + geom_line(mapping = aes(y = credit_roi),col = 'green');g

credit_roi = ((0.1908*9500)-1800)/1800;print(credit_roi)
elite_roi = ((0.1707*11000)-1500)/1500;print(elite_roi)
supp_roi = ((0.1957*8000)-1200)/1200;print(supp_roi)
sum_wt = credit_roi + elite_roi + supp_roi
credit_pct = credit_roi/sum_wt;print(credit_pct)
supp_pct = supp_roi/sum_wt;print(supp_pct)
elite_pct = elite_roi/sum_wt;print(elite_pct)

#Function for creating variables and predictions
predict_final = function(data_frame,logistic_model=log_reg_model$finalModel,model){
    data_frame = create_new_vars(data_frame)
    data_frame$pred_prob = as.numeric(predict(object = logistic_model,newdata = data_frame,type = 'response'))
    data_frame$card_offered = as.character(predict(object = model,newdata = data_frame))
    pred_probs = data_frame$pred_prob
    prob_threshold = 0
    while(prob_threshold <= 1){
        labels = data_frame$card_offered
        labels[pred_probs <= prob_threshold] = 'None'
        z = table(labels)
        budget = z[['Credit']]*1800 + z[['Supp']]*1200 + z[['Elite']]*1500
        print(c(prob_threshold,budget))
        if(budget <= 1000000){break}
        prob_threshold = prob_threshold + 0.0001
    }
    data_frame$card_offered[data_frame$pred_prob <= prob_threshold] = 'None'
    data_frame = data_frame %>% select(cust_key,card_offered)
}

predict_final_2 = function(data_frame,logistic_model=log_reg_model$finalModel,rf_model=rf_gridsearch$finalModel,xgb_model = xgb_gridsearch,ensemble_model = ensemble){
    data_frame = create_new_vars(data_frame)
    data_frame$pred_prob = as.numeric(predict(object = logistic_model,newdata = data_frame,type = 'response'))
    data_frame$rf_prob_credit = predict(rf_model,newdata = data_frame,type = 'prob')[[1]]
    data_frame$rf_prob_elite = predict(rf_model,newdata = data_frame,type = 'prob')[[2]]
    data_frame$rf_prob_supp = predict(rf_model,newdata = data_frame,type = 'prob')[[4]]
    data_frame$xg_prob_credit = predict(xgb_model,newdata = data_frame,type = 'prob')[1][[1]]
    data_frame$xg_prob_elite = predict(xgb_model,newdata = data_frame,type = 'prob')[2][[1]]
    data_frame$xg_prob_supp = predict(xgb_model,newdata = data_frame,type = 'prob')[4][[1]]
    data_frame$log_prob = predict(logistic_model,newdata = data_frame,type = 'response')
    data_frame$rf_pred = predict(rf_model,newdata = data_frame)
    data_frame$xgb_pred = predict(xgb_model,newdata = data_frame)
    data_frame$card_offered = as.character(predict(object = ensemble_model,newdata = data_frame))
    pred_probs = data_frame$pred_prob
    prob_threshold = 0
    while(prob_threshold <= 1){
        pred_labels = as.character(data_frame$card_offered)
        pred_labels[pred_probs <= prob_threshold] = 'None'
        z = table(pred_labels)
        budget = z[['Credit']]*1800 + z[['Supp']]*1200 + z[['Elite']]*1500
        print(c(prob_threshold,budget))
        if(budget <= 1000000){break}
        prob_threshold = prob_threshold + 0.001
    }
    data_frame$card_offered[data_frame$pred_prob <= prob_threshold] = 'None'
    data_frame = data_frame %>% select(cust_key,card_offered)
}

#Predict on Test1 and Test2 datasets
test1 = read_csv(file = '../test_lRt40Da/test1.csv')
test2 = read_csv(file = '../test_lRt40Da/test2.csv')

#Create variables, predict and create final submission files
#test1 = predict_final(data_frame = test1,model = xgb_gridsearch)
#test2 = predict_final(data_frame = test2,model = xgb_gridsearch)
test1 = predict_final_2(data_frame = test1)
test2 = predict_final_2(data_frame = test2)

#Save output files to disk
write.csv(test1,'../sub1.csv',row.names = F)
write.csv(test2,'../sub2.csv',row.names = F)
