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
library(corrplot)

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

z = cor(training_data[,sapply(X = training_data,FUN = function(x){is.numeric(x)})])
write.csv(z,'corr_matrix.csv')
#corrplot.mixed(corr = z,tl.pos = 'lt')

spend_missing_data = train_data[train_data$cust_spending_capacity == 0,]
spend_lm_train = training_data[training_data$cust_spending_capacity != 0,]
spend_lm_test = testing_data[testing_data$cust_spending_capacity != 0,]
#Linear Model to predict spend capacity
spend_lm = lm(data = spend_lm_train,formula = cust_spending_capacity~family_size+nbr_tot_cards+mem_fee_24m+score_affinity_spend+spend_total+cnt_payments_12m)
summary(spend_lm)
spend_lm_test$predicted_spend = predict(object = spend_lm,newdata = spend_lm_test)
sqrt(mean(((spend_lm_test$cust_spending_capacity - spend_lm_test$predicted_spend)^2)))
