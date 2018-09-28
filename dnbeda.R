#Remove previous objects
rm(list = ls())

#Set Working Directory
setwd(dir = 'Desktop/Data&Beyond/train_WG8MUe5/')

#Load libraries
library(dplyr)
library(ggplot2)
library(readr)
library(tidyr)
library(class)
library(caret)
library(ROCR)

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

#Split data into train and test
set.seed(34)
intrain = createDataPartition(train_data$y,times = 1,p = 0.75,list = F)
training_data = train_data[intrain,]
testing_data = train_data[-intrain,]

#Apparel industry, business_score/family_size, income/family_size, dummy vars,
#nbr_months_active, score_affinity_spend/per_mem, nmbr_hist_supp_extns,

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
for(i in 1:30) seeds[[i]] = sample.int(1000,10)
seeds[[31]] = sample.int(1000,1)
print(seeds)
logistic_ctrl <- trainControl(method="repeatedcv", repeats=3,seeds=seeds)

#Train logistic model
log_reg_model <- train(as.factor(y)~family_size+cust_spending_capacity+nbr_tot_cards+nbr_mths_acct+
                   mem_fee_24m+score_affinity_spend+internal_influencer_score+
                   elite_card_ind+cnt_payments_12m+nbr_clb_mem+nbr_air_miles_mem+
                   spend_ec_pct+spend_hotel_pct+spend_household_pct+spend_car_pct+spend_retail_pct+
                   nbr_hist_supp_offer_extns+nbr_hist_supp_offer_acc+
                   spend_log1p_per_mem+spend_capacity_na+income_na+
                   nbr_hist_tot_offer_acc_pct+spend_total_log1p+max_spent_ind_apparel_bool, 
               data=training_data, trControl=logistic_ctrl, method='glm',family = 'binomial')

#Summarize results
print(log_reg_model)
training_data$pred_prob = as.numeric(predict(object = log_reg_model,newdata = training_data,type = 'prob')[2][[1]])
testing_data$pred_prob = as.numeric(predict(object = log_reg_model,newdata = testing_data,type = 'prob')[2][[1]])

#Test predictions on testing data for optimal threshold
predictions = testing_data$pred_prob
labels = testing_data$y

for(i in 10:70){
    prob_threshold = i/100
    pred_labels = ifelse(predictions >= prob_threshold,1,0)
    table_temp = table(labels,pred_labels)
    print('Threshold:')
    print(prob_threshold)
    print('Price:')
    print((table_temp[3] + table_temp[4])*1500)
    print('Return:')
    print((table_temp[4]*9500) + (1000000 - ((table_temp[3] + table_temp[4])*1500))*1.1)
    print('%Accepted:')
    print(table_temp[4]/(table_temp[3] + table_temp[4]))
}
#Optimal Threshold from train data = 0.26-0.28

#ROC Analysis
predictions = as.vector(testing_data$pred_prob)
pred = prediction(predictions,testing_data$y)

perf_AUC = performance(pred,"auc") #Calculate the AUC value
AUC = perf_AUC@y.values[[1]]

perf_ROC = performance(pred,"tpr","fpr") #plot the actual ROC curve
plot(perf_ROC, main="ROC plot")
text(0.5,0.5,paste("AUC = ",format(AUC, digits=5, scientific=FALSE)))
abline(a = 0,b = 1)

#Create training and testing data from Card Accepted Category
training_data_accepted = training_data[training_data$card_accepted != 'Not Accepted',]
testing_data_accepted = testing_data[testing_data$card_accepted != 'Not Accepted',]

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

#Random Forest model with repeated cross-validation
set.seed(332)
seeds = vector(mode = "list", length = 11)
for(i in 1:10) seeds[[i]] = sample.int(1000,10)
seeds[[11]] = sample.int(1000,1)
print(seeds)
rf_ctrl = trainControl(method = "repeatedcv", number = 10, repeats = 1,seeds = seeds,
                       search="grid")
set.seed(37)
tunegrid = expand.grid(.mtry=c(6:15))
for(i in c(5,10,25,50,75)){
print(i)
rf_gridsearch = train(card_accepted~family_size+cust_spending_capacity+nbr_tot_cards+
                          nbr_mths_acct+mem_fee_24m+score_affinity_spend+
                          internal_influencer_score+elite_card_ind+cnt_payments_12m+
                          nbr_clb_mem+nbr_air_miles_mem+spend_ec_pct+spend_hotel_pct+
                          spend_household_pct+spend_car_pct+spend_retail_pct+
                          nbr_hist_supp_offer_extns+nbr_hist_supp_offer_acc+
                          spend_log1p_per_mem+spend_capacity_na+income_na+
                          nbr_hist_tot_offer_acc_pct+spend_total_log1p+
                          max_spent_ind_apparel_bool,data=training_data, method="rf", 
                      metric='logLoss', tuneGrid=tunegrid, trControl=rf_ctrl,ntree=i)
print(rf_gridsearch)
plot(rf_gridsearch)
testing_data$rf_pred = predict(object = rf_gridsearch,newdata = testing_data)
confusionMatrix(testing_data$rf_pred,testing_data$card_accepted)

testing_data$rf_pred = as.character(predict(object = rf_gridsearch,newdata = testing_data))
pred_probs = testing_data$pred_prob
actual_labels = testing_data$card_accepted
prob_threshold = 0
while(prob_threshold <= 1){
    pred_labels = as.character(testing_data$rf_pred)
    pred_labels[pred_probs <= prob_threshold] = 'Not Accepted'
    z = table(pred_labels)
    budget = z[['Credit']]*1800 + z[['Supp']]*1200 + z[['Elite']]*1500
#    print(c(prob_threshold,budget))
    if(budget <= 1000000){
        table_temp = table(actual_labels,pred_labels)
        print(table_temp[1]*9500 + table_temp[6]*11000 + table_temp[16]*8500 + (1000000 - budget)*1.1)
        break
    }
    prob_threshold = prob_threshold + 0.001
}
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

#Test classification model on testing data
testing_data$rf_pred = predict(object = rf_gridsearch,newdata = testing_data)
confusionMatrix(testing_data$rf_pred,testing_data$card_accepted)

testing_data$rf_pred = as.character(predict(object = rf_gridsearch,newdata = testing_data))
pred_probs = testing_data$pred_prob
actual_labels = testing_data$card_accepted
prob_threshold = 0
while(prob_threshold <= 1){
    pred_labels = as.character(testing_data$rf_pred)
    pred_labels[pred_probs <= prob_threshold] = 'Not Accepted'
    z = table(pred_labels)
    budget = z[['Credit']]*1800 + z[['Supp']]*1200 + z[['Elite']]*1500
    print(c(prob_threshold,budget))
    if(budget <= 1000000){
        table_temp = table(actual_labels,pred_labels)
        print(table_temp[1]*9500 + table_temp[6]*11000 + table_temp[16]*8500 + (1000000 - budget)*1.1)
        break
    }
    prob_threshold = prob_threshold + 0.001
}

#Predict on Test1 and Test2 datasets
test1 = read_csv(file = '../test_lRt40Da/test1.csv')
test2 = read_csv(file = '../test_lRt40Da/test2.csv')

#Function for creating variables and predictions
predict_final = function(data_frame,logistic_model,rf_model){
    data_frame = create_new_vars(data_frame)
    data_frame$pred_y = as.numeric(predict(object = logistic_model,newdata = data_frame,type = 'prob')[2][[1]])
    data_frame$card_offered = as.character(predict(object = rf_model,newdata = testing_data))
    pred_probs = data_frame$pred_y
    prob_threshold = 0
    while(prob_threshold <= 1){
        labels = data_frame$card_offered
        labels[pred_probs <= prob_threshold] = 'None'
        z = table(labels)
        budget = z[['Credit']]*1800 + z[['Supp']]*1200 + z[['Elite']]*1500
        print(c(prob_threshold,budget))
        if(budget <= 1000000){break}
        prob_threshold = prob_threshold + 0.001
    }
    data_frame$card_offered[data_frame$pred_y <= prob_threshold] = 'None'
    data_frame = data_frame %>% select(cust_key,card_offered)
}

#Create variables, predict and create final submission files
test1 = predict_final(data_frame = test1,logistic_model = log_reg_model,rf_model = rf_gridsearch)
test2 = predict_final(data_frame = test2,logistic_model = log_reg_model,rf_model = rf_gridsearch)
#test1$card_offered[test1$card_offered == 'Not Accepted'] = 'None'
#test2$card_offered[test2$card_offered == 'Not Accepted'] = 'None'

#Save output files to disk
write.csv(test1,'../sub1.csv',row.names = F)
write.csv(test2,'../sub2.csv',row.names = F)
