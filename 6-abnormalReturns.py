# -*- coding: utf-8 -*-
"""
SCRIPT # 6
Created on Sun Aug  2 12:19:09 2020

@author: omid
"""
import pandas as pd
import numpy as np
from datetime import datetime
from khayyam import *
import statsmodels.api as sm
from scipy.stats import ttest_ind
# seed the random number generator

allFunds = pd.read_pickle("./allFunds.pkl")
bookvalues = pd.read_pickle("./bookvalues.pkl")
risk_premium = pd.read_pickle("./risk_premium.pkl")
No = 30 # No. of funds

# allFunds & risk_premium with Gregorian date index:
dates_convert = pd.DataFrame()
dates_convert["date_j"] = allFunds.index.astype(str).str.split().str.get(0) # date column
dates_convert["date_j"] = dates_convert["date_j"].apply(lambda x : (JalaliDatetime.strptime(x, '%Y-%m-%d')))
dates_convert["date_g"] = dates_convert["date_j"].apply(lambda x : x.todate())
##
funds = allFunds.set_index(dates_convert["date_g"])

dates_convert = pd.DataFrame()
dates_convert["date_j"] = risk_premium.index.astype(str).str.split().str.get(0) # date column
dates_convert["date_j"] = dates_convert["date_j"].apply(lambda x : (JalaliDatetime.strptime(x, '%Y-%m-%d')))
dates_convert["date_g"] = dates_convert["date_j"].apply(lambda x : x.todate())
##
factors = risk_premium.set_index(dates_convert["date_g"])

# Total study window: 1392/1/1 to 1399/1/1
start_date_j = JalaliDate(1392,4,1)
start_date_g = start_date_j.todate() # '2010-06-21'
#Q = 4*(bookvalues["year"].iloc[-1] - bookvalues["year"].iloc[0]) - 1 # total quarters
Q = 4*(bookvalues["year"].iloc[-1] - bookvalues["year"].iloc[0] - 3) - 1 # total quarters


########################################## 1.1 Stock selection
alphas = pd.DataFrame()
performance_deciles_Ranking_quarter_alpha = pd.DataFrame()
performance_deciles_Post_ranking_alpha = pd.DataFrame()
dfs = []
for t in (pd.date_range(start_date_g, periods=Q, freq='Q') - pd.DateOffset(days = 8)) : # -8 to offset between j & g quarter start dates
    g = pd.DataFrame()
    # PREVIOUS 3 months:
    funds_evaluation_period = funds[(funds.index < t) & (funds.index >= (t - pd.DateOffset(months=3)))]
    factors_evaluation_period = factors[(factors.index < t) & (factors.index >= (t - pd.DateOffset(months=3)))]
    
    # Post-ranking quarter:
    funds_Post_ranking = funds[(funds.index >= t) & (funds.index < (t + pd.DateOffset(months=3)))]
    factors_Post_ranking = factors[(factors.index >= t) & (factors.index < (t + pd.DateOffset(months=3)))]
    
    funds_names = funds_evaluation_period.columns
    for f in funds_names :
        ### Regression on the PREVIOUS 3 months:
        Y = pd.DataFrame()
        Y["fund_excess_return"] = funds_evaluation_period[f] - factors_evaluation_period["Rf"]
        factors_evaluation_period.loc[:, "Market-Rf"] = factors_evaluation_period.loc[:,"TEPIX"].values - factors_evaluation_period.loc[:,"Rf"].values
        X = factors_evaluation_period[["Market-Rf", "SMB", "HML", "MOM"]]
        X = sm.add_constant(X)
        fitted_model = sm.OLS(Y["fund_excess_return"].astype(float), X.astype(float), missing='drop').fit()
        
        ### Regression on the Post-ranking quarter:
        Y2 = pd.DataFrame()
        Y2["fund_excess_return"] = funds_Post_ranking[f] - factors_Post_ranking["Rf"]
        factors_Post_ranking.loc[:, "Market-Rf"] = factors_Post_ranking.loc[:,"TEPIX"].values - factors_Post_ranking.loc[:,"Rf"].values
        X2 = factors_Post_ranking[["Market-Rf", "SMB", "HML", "MOM"]]
        X2 = sm.add_constant(X2)
        fitted_model2 = sm.OLS(Y2["fund_excess_return"].astype(float), X2.astype(float), missing='drop').fit()
        
        g = g.append({'Fund': f,
                    'Date': t,
                    'alpha': fitted_model.params[0],
                    '5% significance': fitted_model.pvalues[0]<0.025,
                    'Post alpha': fitted_model2.params[0],
                    'Post 5% significance': fitted_model2.pvalues[0]<0.025}, ignore_index=True)
        
    # Sorting and recording the decile number:
    sorted_g = g.sort_values(by = 'alpha', ascending=False, ignore_index=True)
    i = 0
    d = 1
    groupSize = No//10
    while i < No :
        sorted_g.loc[i, 'decile'] = d
        i += 1
        if (i % groupSize) == 0 :
            d += 1
    dfs.append(sorted_g)
    # Creating report table:
    decile_mean_alphas = sorted_g.groupby('decile')['alpha'].mean()
    decile_mean_Post_alphas = sorted_g.groupby('decile')['Post alpha'].mean()
    
    performance_deciles_Ranking_quarter_alpha = performance_deciles_Ranking_quarter_alpha.join(
        decile_mean_alphas, how='outer', lsuffix='_left', rsuffix='_right')
    performance_deciles_Post_ranking_alpha = performance_deciles_Post_ranking_alpha.join(
        decile_mean_Post_alphas, how='outer', lsuffix='_left', rsuffix='_right')

alphas = pd.concat(dfs, ignore_index=True)

# For the report:
# Ranking quarter:
performance_deciles_Ranking_quarter_alpha = performance_deciles_Ranking_quarter_alpha.mean(axis=1)
performance_deciles_Ranking_quarter_alpha = 100* performance_deciles_Ranking_quarter_alpha

# Post Ranking quarter:
zero = np.zeros(performance_deciles_Post_ranking_alpha.shape[1])
pval = ttest_ind(performance_deciles_Post_ranking_alpha.T, zero)[1]
performance_deciles_Post_ranking_alpha = performance_deciles_Post_ranking_alpha.mean(axis=1)
performance_deciles_Post_ranking_alpha = 100* performance_deciles_Post_ranking_alpha
performance_deciles_Post_ranking_alpha = performance_deciles_Post_ranking_alpha.to_frame('Mean')
performance_deciles_Post_ranking_alpha.loc[:, "P-val"] = pval

for i in performance_deciles_Post_ranking_alpha.index :
    if performance_deciles_Post_ranking_alpha.loc[i, "P-val"] <= 1 : #nothing
        performance_deciles_Post_ranking_alpha.loc[i, "significance"] = ' '
    if performance_deciles_Post_ranking_alpha.loc[i, "P-val"] <= 0.05 : #10%
        performance_deciles_Post_ranking_alpha.loc[i, "significance"] = '*'
    if performance_deciles_Post_ranking_alpha.loc[i, "P-val"] <= 0.025 : #5%
        performance_deciles_Post_ranking_alpha.loc[i, "significance"] = '    '
    if performance_deciles_Post_ranking_alpha.loc[i, "P-val"] <= 0.005 : #1%
        performance_deciles_Post_ranking_alpha.loc[i, "significance"] = '***'




########################################## 1.4 Total returns
R = pd.DataFrame()
performance_deciles_Ranking_quarter_R = pd.DataFrame()
performance_deciles_Post_ranking_R = pd.DataFrame()
dfs = []
for t in (pd.date_range(start_date_g, periods=Q, freq='Q') - pd.DateOffset(days = 8)) : # -8 to offset between j & g quarter start dates
    g = pd.DataFrame()
    # PREVIOUS 3 months:
    funds_evaluation_period = funds[(funds.index < t) & (funds.index >= (t - pd.DateOffset(months=3)))]
    
    # Post-ranking quarter:
    funds_Post_ranking = funds[(funds.index >= t) & (funds.index < (t + pd.DateOffset(months=3)))]
    
    funds_names = funds_evaluation_period.columns
    for f in funds_names :
        g = g.append({'Fund': f,
                    'Date': t,
                    'R': funds_evaluation_period[f].mean(),
                    'Post R': funds_Post_ranking[f].mean()}, ignore_index=True)
        
    # Sorting and recording the decile number:
    sorted_g = g.sort_values(by = 'R', ascending=False, ignore_index=True)
    i = 0
    d = 1
    groupSize = No//10
    while i < No :
        sorted_g.loc[i, 'decile'] = d
        i += 1
        if (i % groupSize) == 0 :
            d += 1
    dfs.append(sorted_g)
    # Creating report table:
    decile_mean_R = sorted_g.groupby('decile')['R'].mean()
    decile_mean_Post_R = sorted_g.groupby('decile')['Post R'].mean()
    
    performance_deciles_Ranking_quarter_R = performance_deciles_Ranking_quarter_R.join(
        decile_mean_R, how='outer', lsuffix='_left', rsuffix='_right')
    performance_deciles_Post_ranking_R = performance_deciles_Post_ranking_R.join(
        decile_mean_Post_R, how='outer', lsuffix='_left', rsuffix='_right')

R = pd.concat(dfs, ignore_index=True)

# For the report:
# Ranking quarter:
performance_deciles_Ranking_quarter_R = performance_deciles_Ranking_quarter_R.mean(axis=1)
performance_deciles_Ranking_quarter_R = 100* performance_deciles_Ranking_quarter_R

# Post Ranking quarter:
performance_deciles_Post_ranking_R = performance_deciles_Post_ranking_R.mean(axis=1)
performance_deciles_Post_ranking_R = 100* performance_deciles_Post_ranking_R



########################################## 1.2 Market timing
# Method 1: Treynor and Mazuy (TM)
TM = pd.DataFrame()
performance_deciles_Ranking_quarter_TM = pd.DataFrame()
performance_deciles_Post_ranking_TM = pd.DataFrame()
dfs = []
for t in (pd.date_range(start_date_g, periods=Q, freq='Q') - pd.DateOffset(days = 8)) :
    g = pd.DataFrame()
    # PREVIOUS 3 months:
    funds_evaluation_period = funds[(funds.index < t) & (funds.index >= (t - pd.DateOffset(months=3)))]
    factors_evaluation_period = factors[(factors.index < t) & (factors.index >= (t - pd.DateOffset(months=3)))]
    
    # Post-ranking quarter:
    funds_Post_ranking = funds[(funds.index >= t) & (funds.index < (t + pd.DateOffset(months=3)))]
    factors_Post_ranking = factors[(factors.index >= t) & (factors.index < (t + pd.DateOffset(months=3)))]
    
    N = factors_evaluation_period.shape[0]
    funds_names = funds_evaluation_period.columns
    for f in funds_names :
        # Regression on the PREVIOUS 3 months:
        Y = pd.DataFrame()
        Y["fund_excess_return"] = funds_evaluation_period[f] - factors_evaluation_period["Rf"]
        factors_evaluation_period.loc[:, "Market-Rf"] = factors_evaluation_period.loc[:,"TEPIX"].values - factors_evaluation_period.loc[:,"Rf"].values
        factors_evaluation_period["(Market-Rf)^2"] = factors_evaluation_period["Market-Rf"]**2
        X = factors_evaluation_period[["Market-Rf", "(Market-Rf)^2", "SMB", "HML", "MOM"]]
        X = sm.add_constant(X)
        fitted_model = sm.OLS(Y["fund_excess_return"].astype(float), X.astype(float), missing='drop').fit()
        r = fitted_model.params[0] + fitted_model.params[2] * (factors_evaluation_period["(Market-Rf)^2"].sum()) * (1/N)
        
        # Regression on the Post-ranking quarter:
        Y2 = pd.DataFrame()
        Y2["fund_excess_return"] = funds_Post_ranking[f] - factors_Post_ranking["Rf"]
        factors_Post_ranking.loc[:, "Market-Rf"] = factors_Post_ranking.loc[:,"TEPIX"].values - factors_Post_ranking.loc[:,"Rf"].values
        factors_Post_ranking["(Market-Rf)^2"] = factors_Post_ranking["Market-Rf"]**2
        X2 = factors_Post_ranking[["Market-Rf", "(Market-Rf)^2", "SMB", "HML", "MOM"]]
        X2 = sm.add_constant(X2)
        fitted_model2 = sm.OLS(Y2["fund_excess_return"].astype(float), X2.astype(float), missing='drop').fit()
        r2 = fitted_model2.params[0] + fitted_model2.params[2] * (factors_Post_ranking["(Market-Rf)^2"].sum()) * (1/N)
        
        g = g.append({'Fund': f,
                    'Date': t,
                    'alpha': fitted_model.params[0],
                    'lambda': fitted_model.params[2],
                    '5% significance of lambda': fitted_model.pvalues[2]<0.025,
                    'r': r,
                    'Post alpha': fitted_model2.params[0],
                    'Post lambda': fitted_model2.params[2],
                    'Post 5% significance of lambda': fitted_model2.pvalues[2]<0.025,
                    'Post r': r2,}, ignore_index=True)
        
    # Sorting and recording the decile number:
    sorted_g = g.sort_values(by = 'r', ascending=False, ignore_index=True)
    i = 0
    d = 1
    groupSize = No//10
    while i < No :
        sorted_g.loc[i, 'decile'] = d
        i += 1
        if (i % groupSize) == 0 :
            d += 1
    dfs.append(sorted_g)
    # Creating report table:
    decile_means = sorted_g.groupby('decile')['r'].mean()
    decile_means_post = sorted_g.groupby('decile')['Post r'].mean()
    
    performance_deciles_Ranking_quarter_TM = performance_deciles_Ranking_quarter_TM.join(
        decile_means, how='outer', lsuffix='_left', rsuffix='_right')
    performance_deciles_Post_ranking_TM = performance_deciles_Post_ranking_TM.join(
        decile_means_post, how='outer', lsuffix='_left', rsuffix='_right')

TM = pd.concat(dfs, ignore_index=True)

# For the report:
# Ranking quarter:
performance_deciles_Ranking_quarter_TM = performance_deciles_Ranking_quarter_TM.mean(axis=1)
performance_deciles_Ranking_quarter_TM = 100 * performance_deciles_Ranking_quarter_TM

# Post Ranking quarter:
zero = np.zeros(performance_deciles_Post_ranking_TM.shape[1])
pval = ttest_ind(performance_deciles_Post_ranking_TM.T, zero)[1]
performance_deciles_Post_ranking_TM = performance_deciles_Post_ranking_TM.mean(axis=1)
performance_deciles_Post_ranking_TM = 100* performance_deciles_Post_ranking_TM
performance_deciles_Post_ranking_TM = performance_deciles_Post_ranking_TM.to_frame('Mean')
performance_deciles_Post_ranking_TM.loc[:, "P-val"] = pval

for i in performance_deciles_Post_ranking_TM.index :
    if performance_deciles_Post_ranking_TM.loc[i, "P-val"] <= 1 : #nothing
        performance_deciles_Post_ranking_TM.loc[i, "significance"] = '*'
    if performance_deciles_Post_ranking_TM.loc[i, "P-val"] <= 0.05 : #10%
        performance_deciles_Post_ranking_TM.loc[i, "significance"] = '*'
    if performance_deciles_Post_ranking_TM.loc[i, "P-val"] <= 0.025 : #5%
        performance_deciles_Post_ranking_TM.loc[i, "significance"] = '**'
    if performance_deciles_Post_ranking_TM.loc[i, "P-val"] <= 0.005 : #1%
        performance_deciles_Post_ranking_TM.loc[i, "significance"] = '***'




### Method 2: Henriksson and Merton (HM)

HM = pd.DataFrame()
performance_deciles_Ranking_quarter_HM = pd.DataFrame()
performance_deciles_Post_ranking_HM = pd.DataFrame()
dfs = []
for t in (pd.date_range(start_date_g, periods=Q, freq='Q') - pd.DateOffset(days = 8)) :
    g = pd.DataFrame()
    # PREVIOUS 3 months:
    funds_evaluation_period = funds[(funds.index < t) & (funds.index >= (t - pd.DateOffset(months=3)))]
    factors_evaluation_period = factors[(factors.index < t) & (factors.index >= (t - pd.DateOffset(months=3)))]
    
    # Post-ranking quarter:
    funds_Post_ranking = funds[(funds.index >= t) & (funds.index < (t + pd.DateOffset(months=3)))]
    factors_Post_ranking = factors[(factors.index >= t) & (factors.index < (t + pd.DateOffset(months=3)))]
    
    N = factors_evaluation_period.shape[0]
    funds_names = funds_evaluation_period.columns
    for f in funds_names :
        # Regression on the previous 3 months:
        Y = pd.DataFrame()
        Y["fund_excess_return"] = funds_evaluation_period[f] - factors_evaluation_period["Rf"]
        factors_evaluation_period.loc[:, "Market-Rf"] = factors_evaluation_period.loc[:,"TEPIX"].values - factors_evaluation_period.loc[:,"Rf"].values
        factors_evaluation_period["(Market-Rf)>0"] = factors_evaluation_period["Market-Rf"] * pd.Series(factors_evaluation_period["Market-Rf"]>0).astype(int)
        X = factors_evaluation_period[["Market-Rf", "(Market-Rf)>0", "SMB", "HML", "MOM"]]
        X = sm.add_constant(X)
        fitted_model = sm.OLS(Y["fund_excess_return"].astype(float), X.astype(float), missing='drop').fit()
        r = fitted_model.params[0] + fitted_model.params[2] * (factors_evaluation_period["(Market-Rf)>0"].sum()) * (1/N)
        
        # Regression on the Post-ranking quarter:
        Y2 = pd.DataFrame()
        Y2["fund_excess_return"] = funds_Post_ranking[f] - factors_Post_ranking["Rf"]
        factors_Post_ranking.loc[:, "Market-Rf"] = factors_Post_ranking.loc[:,"TEPIX"].values - factors_Post_ranking.loc[:,"Rf"].values
        factors_Post_ranking["(Market-Rf)>0"] = factors_Post_ranking["Market-Rf"] * pd.Series(factors_Post_ranking["Market-Rf"]>0).astype(int)
        X2 = factors_Post_ranking[["Market-Rf", "(Market-Rf)>0", "SMB", "HML", "MOM"]]
        X2 = sm.add_constant(X2)
        fitted_model2 = sm.OLS(Y2["fund_excess_return"].astype(float), X2.astype(float), missing='drop').fit()
        r2 = fitted_model2.params[0] + fitted_model2.params[2] * (factors_Post_ranking["(Market-Rf)>0"].sum()) * (1/N)
        
        g = g.append({'Fund': f,
                    'Date': t,
                    'alpha': fitted_model.params[0],
                    'lambda': fitted_model.params[2],
                    '5% significance of lambda': fitted_model.pvalues[2]<0.025,
                    'r': r,
                    'Post alpha': fitted_model2.params[0],
                    'Post lambda': fitted_model2.params[2],
                    'Post 5% significance of lambda': fitted_model2.pvalues[2]<0.025,
                    'Post r': r2,}, ignore_index=True)
        
    # Sorting and recording the decile number:
    sorted_g = g.sort_values(by = 'r', ascending=False, ignore_index=True)
    i = 0
    d = 1
    groupSize = No//10
    while i < No :
        sorted_g.loc[i, 'decile'] = d
        i += 1
        if (i % groupSize) == 0 :
            d += 1
    dfs.append(sorted_g)
    # Creating report table:
    decile_means = sorted_g.groupby('decile')['r'].mean()
    decile_means_post = sorted_g.groupby('decile')['Post r'].mean()
    
    performance_deciles_Ranking_quarter_HM = performance_deciles_Ranking_quarter_HM.join(
        decile_means, how='outer', lsuffix='_left', rsuffix='_right')
    performance_deciles_Post_ranking_HM = performance_deciles_Post_ranking_HM.join(
        decile_means_post, how='outer', lsuffix='_left', rsuffix='_right')
    
HM = pd.concat(dfs, ignore_index=True)

# For the report:
# Ranking quarter:
performance_deciles_Ranking_quarter_HM = performance_deciles_Ranking_quarter_HM.mean(axis=1)
performance_deciles_Ranking_quarter_HM = 100 * performance_deciles_Ranking_quarter_HM

# Post Ranking quarter:
zero = np.zeros(performance_deciles_Post_ranking_HM.shape[1])
pval = ttest_ind(performance_deciles_Post_ranking_HM.T, zero)[1]
performance_deciles_Post_ranking_HM = performance_deciles_Post_ranking_HM.mean(axis=1)
performance_deciles_Post_ranking_HM = 100* performance_deciles_Post_ranking_HM
performance_deciles_Post_ranking_HM = performance_deciles_Post_ranking_HM.to_frame('Mean')
performance_deciles_Post_ranking_HM.loc[:, "P-val"] = pval

for i in performance_deciles_Post_ranking_HM.index :
    if performance_deciles_Post_ranking_HM.loc[i, "P-val"] <= 1 : #nothing
        performance_deciles_Post_ranking_HM.loc[i, "significance"] = '*'
    if performance_deciles_Post_ranking_HM.loc[i, "P-val"] <= 0.05 : #10%
        performance_deciles_Post_ranking_HM.loc[i, "significance"] = '*'
    if performance_deciles_Post_ranking_HM.loc[i, "P-val"] <= 0.025 : #5%
        performance_deciles_Post_ranking_HM.loc[i, "significance"] = '**'
    if performance_deciles_Post_ranking_HM.loc[i, "P-val"] <= 0.005 : #1%
        performance_deciles_Post_ranking_HM.loc[i, "significance"] = '***'



########################################## 1.3 Mixed Ranking

### Method 1: (TM)
Mixed_TM = TM.copy()
del Mixed_TM['alpha']
del Mixed_TM['lambda']
del Mixed_TM['decile']

# Replacing r of the previuos & next season with alpha (stock selection) where timing lambda of the previuos season is not significant:
for i in range(len(Mixed_TM.index)) :
    if Mixed_TM.loc[i, '5% significance of lambda'] == 0 :
        t = Mixed_TM.loc[i, 'Date']
        f = Mixed_TM.loc[i, 'Fund']
        Mixed_TM.loc[i, 'r'] = alphas.loc[(alphas['Date'] == t) & (alphas['Fund'] == f), 'alpha'].iloc[0]
        Mixed_TM.loc[i, 'Post r'] = alphas.loc[(alphas['Date'] == t) & (alphas['Fund'] == f), 'Post alpha'].iloc[0]

# Assigning deciles in each quarter:
performance_deciles_Ranking_quarter_mixed_TM = pd.DataFrame()
performance_deciles_Post_ranking_mixed_TM = pd.DataFrame()
for t in Mixed_TM['Date'].unique() :
    g = Mixed_TM[Mixed_TM['Date'] == t]
    sorted_g = g.sort_values(by = 'r', ascending=False, ignore_index=True)
    i = 0
    d = 1
    groupSize = No//10
    while i < No :
        sorted_g.loc[i, 'decile'] = d
        i += 1
        if (i % groupSize) == 0 :
            d += 1

    # Creating report table:
    decile_means = sorted_g.groupby('decile')['r'].mean()
    decile_means_post = sorted_g.groupby('decile')['Post r'].mean()
    
    performance_deciles_Ranking_quarter_mixed_TM = performance_deciles_Ranking_quarter_mixed_TM.join(
        decile_means, how='outer', lsuffix='_left', rsuffix='_right')
    performance_deciles_Post_ranking_mixed_TM = performance_deciles_Post_ranking_mixed_TM.join(
        decile_means_post, how='outer', lsuffix='_left', rsuffix='_right')

# For the report:
# Ranking quarter:
performance_deciles_Ranking_quarter_mixed_TM = performance_deciles_Ranking_quarter_mixed_TM.mean(axis=1)
performance_deciles_Ranking_quarter_mixed_TM = 100 * performance_deciles_Ranking_quarter_mixed_TM

# Post Ranking quarter:
zero = np.zeros(performance_deciles_Post_ranking_mixed_TM.shape[1])
pval = ttest_ind(performance_deciles_Post_ranking_mixed_TM.T, zero)[1]
performance_deciles_Post_ranking_mixed_TM = performance_deciles_Post_ranking_mixed_TM.mean(axis=1)
performance_deciles_Post_ranking_mixed_TM = 100* performance_deciles_Post_ranking_mixed_TM
performance_deciles_Post_ranking_mixed_TM = performance_deciles_Post_ranking_mixed_TM.to_frame('Mean')
performance_deciles_Post_ranking_mixed_TM.loc[:, "P-val"] = pval

for i in performance_deciles_Post_ranking_mixed_TM.index :
    if performance_deciles_Post_ranking_mixed_TM.loc[i, "P-val"] <= 1 : #nothing
        performance_deciles_Post_ranking_mixed_TM.loc[i, "significance"] = '*'
    if performance_deciles_Post_ranking_mixed_TM.loc[i, "P-val"] <= 0.05 : #10%
        performance_deciles_Post_ranking_mixed_TM.loc[i, "significance"] = '*'
    if performance_deciles_Post_ranking_mixed_TM.loc[i, "P-val"] <= 0.025 : #5%
        performance_deciles_Post_ranking_mixed_TM.loc[i, "significance"] = '**'
    if performance_deciles_Post_ranking_mixed_TM.loc[i, "P-val"] <= 0.005 : #1%
        performance_deciles_Post_ranking_mixed_TM.loc[i, "significance"] = '***'




### Method 2: (HM)
Mixed_HM = HM.copy()
del Mixed_HM['alpha']
del Mixed_HM['lambda']
del Mixed_HM['decile']

# Replacing r of the previuos & next season with alpha (stock selection) where timing lambda of the previuos season is not significant:
for i in range(len(Mixed_HM.index)) :
    if Mixed_HM.loc[i, '5% significance of lambda'] == 0 :
        t = Mixed_HM.loc[i, 'Date']
        f = Mixed_HM.loc[i, 'Fund']
        Mixed_HM.loc[i, 'r'] = alphas.loc[(alphas['Date'] == t) & (alphas['Fund'] == f), 'alpha'].iloc[0]
        Mixed_HM.loc[i, 'Post r'] = alphas.loc[(alphas['Date'] == t) & (alphas['Fund'] == f), 'Post alpha'].iloc[0]

# Assigning deciles in each quarter:
performance_deciles_Ranking_quarter_mixed_HM = pd.DataFrame()
performance_deciles_Post_ranking_mixed_HM = pd.DataFrame()
for t in Mixed_HM['Date'].unique() :
    g = Mixed_HM[Mixed_HM['Date'] == t]
    sorted_g = g.sort_values(by = 'r', ascending=False, ignore_index=True)
    i = 0
    d = 1
    groupSize = No//10
    while i < No :
        sorted_g.loc[i, 'decile'] = d
        i += 1
        if (i % groupSize) == 0 :
            d += 1
    
    # Creating report table:
    decile_means = sorted_g.groupby('decile')['r'].mean()
    decile_means_post = sorted_g.groupby('decile')['Post r'].mean()
    
    performance_deciles_Ranking_quarter_mixed_HM = performance_deciles_Ranking_quarter_mixed_HM.join(
        decile_means, how='outer', lsuffix='_left', rsuffix='_right')
    performance_deciles_Post_ranking_mixed_HM = performance_deciles_Post_ranking_mixed_HM.join(
        decile_means_post, how='outer', lsuffix='_left', rsuffix='_right')

# For the report:
# Ranking quarter:
performance_deciles_Ranking_quarter_mixed_HM = performance_deciles_Ranking_quarter_mixed_HM.mean(axis=1)
performance_deciles_Ranking_quarter_mixed_HM = 100 * performance_deciles_Ranking_quarter_mixed_HM

# Post Ranking quarter:
zero = np.zeros(performance_deciles_Post_ranking_mixed_HM.shape[1])
pval = ttest_ind(performance_deciles_Post_ranking_mixed_HM.T, zero)[1]
performance_deciles_Post_ranking_mixed_HM = performance_deciles_Post_ranking_mixed_HM.mean(axis=1)
performance_deciles_Post_ranking_mixed_HM = 100* performance_deciles_Post_ranking_mixed_HM
performance_deciles_Post_ranking_mixed_HM = performance_deciles_Post_ranking_mixed_HM.to_frame('Mean')
performance_deciles_Post_ranking_mixed_HM.loc[:, "P-val"] = pval

for i in performance_deciles_Post_ranking_mixed_HM.index :
    if performance_deciles_Post_ranking_mixed_HM.loc[i, "P-val"] <= 1 : #nothing
        performance_deciles_Post_ranking_mixed_HM.loc[i, "significance"] = '*'
    if performance_deciles_Post_ranking_mixed_HM.loc[i, "P-val"] <= 0.05 : #10%
        performance_deciles_Post_ranking_mixed_HM.loc[i, "significance"] = '*'
    if performance_deciles_Post_ranking_mixed_HM.loc[i, "P-val"] <= 0.025 : #5%
        performance_deciles_Post_ranking_mixed_HM.loc[i, "significance"] = '**'
    if performance_deciles_Post_ranking_mixed_HM.loc[i, "P-val"] <= 0.005 : #1%
        performance_deciles_Post_ranking_mixed_HM.loc[i, "significance"] = '***'




########################################## Table 4: cross-sectional
A_R = []
B_R = []
R2_R = []

A_alpha = []
B_alpha = []
R2_alpha = []

A_HM = []
B_HM = []
R2_HM = []

A_TM = []
B_TM = []
R2_TM = []

A_Mixed_HM = []
B_Mixed_HM = []
R2_Mixed_HM = []

A_Mixed_TM = []
B_Mixed_TM = []
R2_Mixed_TM = []

for t in (pd.date_range(start_date_g, periods=Q, freq='Q') - pd.DateOffset(days = 8)) : # -8 to offset between j & g quarter start dates
    
    R_quarter = R[R['Date'] == t]
    X = R_quarter[["R"]]
    X = sm.add_constant(X)
    Y = R_quarter[["Post R"]]
    fitted_model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
    A_R.append(fitted_model.params[0])
    B_R.append(fitted_model.params[1])
    R2_R.append(fitted_model.rsquared)
    
    alphas_quarter = alphas[alphas['Date'] == t]
    X = alphas_quarter[["alpha"]]
    X = sm.add_constant(X)
    Y = alphas_quarter[["Post alpha"]]
    fitted_model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
    A_alpha.append(fitted_model.params[0])
    B_alpha.append(fitted_model.params[1])
    R2_alpha.append(fitted_model.rsquared)
    
    TM_quarter = TM[TM['Date'] == t]
    X = TM_quarter[["r"]]
    X = sm.add_constant(X)
    Y = TM_quarter[["Post r"]]
    fitted_model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
    A_TM.append(fitted_model.params[0])
    B_TM.append(fitted_model.params[1])
    R2_TM.append(fitted_model.rsquared)
    
    HM_quarter = HM[HM['Date'] == t]
    X = HM_quarter[["r"]]
    X = sm.add_constant(X)
    Y = HM_quarter[["Post r"]]
    fitted_model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
    A_HM.append(fitted_model.params[0])
    B_HM.append(fitted_model.params[1])
    R2_HM.append(fitted_model.rsquared)
    
    Mixed_TM_quarter = Mixed_TM[Mixed_TM['Date'] == t]
    X = Mixed_TM_quarter[["r"]]
    X = sm.add_constant(X)
    Y = Mixed_TM_quarter[["Post r"]]
    fitted_model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
    A_Mixed_TM.append(fitted_model.params[0])
    B_Mixed_TM.append(fitted_model.params[1])
    R2_Mixed_TM.append(fitted_model.rsquared)
    
    Mixed_HM_quarter = Mixed_HM[Mixed_HM['Date'] == t]
    X = Mixed_HM_quarter[["r"]]
    X = sm.add_constant(X)
    Y = Mixed_HM_quarter[["Post r"]]
    fitted_model = sm.OLS(Y.astype(float), X.astype(float), missing='drop').fit()
    A_Mixed_HM.append(fitted_model.params[0])
    B_Mixed_HM.append(fitted_model.params[1])
    R2_Mixed_HM.append(fitted_model.rsquared)
    

# For the report:
# R:
A_R_mean = np.mean(A_R)
B_R_mean = np.mean(B_R)

zero = np.zeros(len(A_R))
A_R_p = ttest_ind(A_R, zero)[1]
B_R_p = ttest_ind(B_R, zero)[1]
R2_R = np.mean(R2_R)

# alpha:
A_alpha_mean = np.mean(A_alpha)
B_alpha_mean = np.mean(B_alpha)

zero = np.zeros(len(A_alpha))
A_alpha_p = ttest_ind(A_alpha, zero)[1]
B_alpha_p = ttest_ind(B_alpha, zero)[1]
R2_alpha = np.mean(R2_alpha)

# TM:
A_TM_mean = np.mean(A_TM)
B_TM_mean = np.mean(B_TM)

zero = np.zeros(len(A_TM))
A_TM_p = ttest_ind(A_TM, zero)[1]
B_TM_p = ttest_ind(B_TM, zero)[1]
R2_TM = np.mean(R2_TM)

# HM:
A_HM_mean = np.mean(A_HM)
B_HM_mean = np.mean(B_HM)

zero = np.zeros(len(A_HM))
A_HM_p = ttest_ind(A_HM, zero)[1]
B_HM_p = ttest_ind(B_HM, zero)[1]
R2_HM = np.mean(R2_HM)

# Mixed_TM:
A_Mixed_TM_mean = np.mean(A_Mixed_TM)
B_Mixed_TM_mean = np.mean(B_Mixed_TM)

zero = np.zeros(len(A_Mixed_TM))
A_Mixed_TM_p = ttest_ind(A_Mixed_TM, zero)[1]
B_Mixed_TM_p = ttest_ind(B_Mixed_TM, zero)[1]
R2_Mixed_TM = np.mean(R2_Mixed_TM)

# Mixed_HM:
A_Mixed_HM_mean = np.mean(A_Mixed_HM)
B_Mixed_HM_mean = np.mean(B_Mixed_HM)

zero = np.zeros(len(A_Mixed_HM))
A_Mixed_HM_p = ttest_ind(A_Mixed_HM, zero)[1]
B_Mixed_HM_p = ttest_ind(B_Mixed_HM, zero)[1]
R2_Mixed_HM = np.mean(R2_Mixed_HM)



########################################## End of calculations

Table1 = pd.DataFrame({"Stock selection (%)": performance_deciles_Ranking_quarter_alpha,
                       "Market timing (%) - TM": performance_deciles_Ranking_quarter_TM,
                       "Market timing (%) - HM": performance_deciles_Ranking_quarter_HM,
                       "Mixed (%) - TM": performance_deciles_Ranking_quarter_mixed_TM,
                       "Mixed (%) - HM": performance_deciles_Ranking_quarter_mixed_HM})

Table2 = pd.DataFrame({("Stock selection (%)","Mean"): performance_deciles_Post_ranking_alpha['Mean'],
                       ("Stock selection (%)","Significance"): performance_deciles_Post_ranking_alpha['significance'],
                       ("Market timing (%) - TM","Mean"): performance_deciles_Post_ranking_TM['Mean'],
                       ("Market timing (%) - TM","Significance"): performance_deciles_Post_ranking_TM['significance'],
                       ("Market timing (%) - HM","Mean"): performance_deciles_Post_ranking_HM['Mean'],
                       ("Market timing (%) - HM","Significance"): performance_deciles_Post_ranking_HM['significance'],
                       ("Mixed (%) - TM","Mean"): performance_deciles_Post_ranking_mixed_TM['Mean'],
                       ("Mixed (%) - TM","Significance"): performance_deciles_Post_ranking_mixed_TM['significance'],
                       ("Mixed (%) - HM","Mean"): performance_deciles_Post_ranking_mixed_HM['Mean'],
                       ("Mixed (%) - HM","Significance"): performance_deciles_Post_ranking_mixed_HM['significance']})

Table3 = pd.DataFrame({("R (%)","Mean"): performance_deciles_Post_ranking_R,
                       ("Stock selection (%)","Mean"): performance_deciles_Post_ranking_alpha['Mean'],
                       ("Market timing (%) - TM","Mean"): performance_deciles_Post_ranking_TM['Mean'],
                       ("Market timing (%) - HM","Mean"): performance_deciles_Post_ranking_HM['Mean'],
                       ("Mixed (%) - TM","Mean"): performance_deciles_Post_ranking_mixed_TM['Mean'],
                       ("Mixed (%) - HM","Mean"): performance_deciles_Post_ranking_mixed_HM['Mean']})

# Table 4 Dataframe:
values = [
            [A_R_mean, A_alpha_mean, A_TM_mean, A_HM_mean, A_Mixed_TM_mean, A_Mixed_HM_mean],
            [A_R_p, A_alpha_p, A_TM_p, A_HM_p, A_Mixed_TM_p, A_Mixed_HM_p],
            [B_R_mean, B_alpha_mean, B_TM_mean, B_HM_mean, B_Mixed_TM_mean, B_Mixed_HM_mean],
            [B_R_p, B_alpha_p, B_TM_p, B_HM_p, B_Mixed_TM_p, B_Mixed_HM_p],
            [R2_R, R2_alpha, R2_TM, R2_HM, R2_Mixed_TM, R2_Mixed_HM]
         ]
Table4 = pd.DataFrame(values, columns=["R", "Stock selection", "Market timing - TM", "Market timing - HM", "Mixed - TM", "Mixed - HM"],
                  index=['A', 'p-value', 'B', 'p-value', 'R-Squared'])


Table1.to_pickle("./Table1.pkl")
Table2.to_pickle("./Table2.pkl")
Table3.to_pickle("./Table3.pkl")
Table4.to_pickle("./Table4.pkl")

#Table1 = pd.read_pickle("./Table1.pkl")
#Table2 = pd.read_pickle("./Table2.pkl")
