#Remove previous objects
rm(list = ls())

#Set Working Directory
setwd(dir = '/home/sambeet/data/Data&Beyond/Data&Beyond/train_WG8MUe5/')

#Load libraries
library(dplyr)
library(ggplot2)
library(readr)
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
train_data$accepted = factor(ifelse(train_data$y == 1,'acc','nacc'))
train_data$card_accepted[train_data$card_accepted == 'Not Accepted'] = 'None'
train_data$y_mult = factor(ifelse(train_data$card_accepted == 'None',0,
                                  ifelse(train_data$card_accepted == 'Supp',1,
                                         ifelse(train_data$card_accepted == 'Credit',2,3))))

#Split data into train and test
set.seed(34)
intrain = createDataPartition(train_data$y,times = 1,p = 0.75,list = F)
training_data = train_data[intrain,]
testing_data = train_data[-intrain,]

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
                           max_spent_ind_apparel_bool,data=train_data, 
                       trControl=logistic_ctrl, method='glm',family = 'binomial')

#Summarize results
print(log_reg_model$finalModel)
training_data$pred_prob = as.numeric(predict(object = log_reg_model$finalModel,newdata = training_data,type = 'response'))
testing_data$pred_prob = as.numeric(predict(object = log_reg_model$finalModel,newdata = testing_data,type = 'response'))
train_data$pred_prob = as.numeric(predict(object = log_reg_model$finalModel,newdata = train_data,type = 'response'))

#ROC Analysis
predictions = as.vector(testing_data$pred_prob)
pred = prediction(predictions,testing_data$y)

perf_AUC = performance(pred,"auc") #Calculate the AUC value
AUC = perf_AUC@y.values[[1]]

perf_ROC = performance(pred,"tpr","fpr") #plot the actual ROC curve
plot(perf_ROC, main="ROC plot")
text(0.5,0.5,paste("AUC = ",format(AUC, digits=5, scientific=FALSE)))
abline(a = 0,b = 1)

#Random Forest model with repeated cross-validation
library(doMC)
registerDoMC(cores = 7)
set.seed(3322)
seeds = vector(mode = "list", length = 31)
for(i in 1:30) seeds[[i]] = sample.int(1000,9)
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
                          max_spent_ind_apparel_bool+pred_prob,data=train_data, method="rf", 
                      metric='Kappa', trControl=rf_ctrl,ntree = 1000,tuneGrid=tunegrid,sampsize = c(600,600,1000,600))
print(rf_gridsearch)
print(rf_gridsearch$finalModel)
varImp(object = rf_gridsearch)

#Function for creating variables and predictions
predict_final = function(data_frame,logistic_model=log_reg_model$finalModel,rf_model=rf_gridsearch$finalModel){
    data_frame = create_new_vars(data_frame)
    data_frame$pred_prob = as.numeric(predict(object = logistic_model,newdata = data_frame,type = 'response'))
    data_frame$rf_prob_credit = predict(rf_model,newdata = data_frame,type = 'prob')[,1]
    data_frame$rf_prob_elite = predict(rf_model,newdata = data_frame,type = 'prob')[,2]
    data_frame$rf_prob_supp = predict(rf_model,newdata = data_frame,type = 'prob')[,4]
    data_frame$rf_pred = predict(rf_model,newdata = data_frame)
    data_frame$card_offered = as.character(predict(object = rf_model,newdata = data_frame))
    pred_labels = as.character(data_frame$card_offered)
    budget_left = 1000000 - (sum(pred_labels == 'Elite')*1500 + sum(pred_labels == 'Credit')*1800 + sum(pred_labels == 'Supp')*1200)
    if(budget_left >= 1200){
        n = length(integer(budget_left/1200))
        x = n
        prob_threshold_values = top_n(data.frame(prob_values = data_frame$rf_prob_supp[pred_labels == 'None']),n = n)
        while(nrow(prob_threshold_values) > n){
            x = x - 1
            prob_threshold_values = top_n(data.frame(prob_values = data_frame$rf_prob_supp[pred_labels == 'None']),n = x)
        }
        pred_labels[data_frame$rf_prob_supp %in% prob_threshold_values$prob_values & pred_labels == 'None'] = 'Supp'
    }
    data_frame$card_offered = pred_labels
    print(paste('Budget:',(sum(pred_labels == 'Elite')*1500 + sum(pred_labels == 'Credit')*1800 + sum(pred_labels == 'Supp')*1200)))
    data_frame = data_frame %>% select(cust_key,card_offered)
}

#Predict on Test1 and Test2 datasets
test1 = read_csv(file = '../test_lRt40Da/test1.csv')
test2 = read_csv(file = '../test_lRt40Da/test2.csv')

#Create variables, predict and create final submission files
test1 = predict_final(data_frame = test1)
test2 = predict_final(data_frame = test2)

#Save output files to disk
write.csv(test1,'../sub1.csv',row.names = F)
write.csv(test2,'../sub2.csv',row.names = F)
