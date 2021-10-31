# -*- coding: utf-8 -*-
"""
SCRIPT # 4
Created on Fri Jul 31 19:32:20 2020

@author: omid
"""
import pandas as pd
from datetime import datetime
from khayyam import *

allStocks = pd.read_pickle("./allStocks.pkl")
risk_premium = pd.read_pickle("./risk_premium.pkl")
bookvalues = pd.read_pickle("./bookvalues.pkl")

# Allstocks with Gregorian date index:
dates_convert = pd.DataFrame()
dates_convert["date_j"] = allStocks.index.astype(str).str.split().str.get(0) # date column
dates_convert["date_j"] = dates_convert["date_j"].apply(lambda x : (JalaliDatetime.strptime(x, '%Y-%m-%d')))
dates_convert["date_g"] = dates_convert["date_j"].apply(lambda x : x.todate())
stocks = allStocks.set_index(dates_convert["date_g"])

start_date_j = JalaliDate(1392, 1, 1)
start_date_g = start_date_j.todate() # '2010-03-21'
T = 12 * (bookvalues["year"].iloc[-1] - bookvalues["year"].iloc[0]) # total months
J = 12 # return calculation period start
H = 2 # return calculation period end

for t in (pd.date_range(start_date_g, periods=T, freq='M') - pd.DateOffset(days = 8)) : # 6 is the minimum margin between g & j month start dates to ensure no overlaps
    this_year = JalaliDatetime(t).year
    last_year_BV = bookvalues[bookvalues["year"] == this_year - 1]
    # Keeping only the stocks that we know their last_year_BV.
    valid_stocks = pd.DataFrame()
    cols = stocks.columns 
    for c in cols :
        if c in last_year_BV['ticker'].values :
            valid_stocks[c] = stocks[c]
    #
    stocks_evaluation_period = valid_stocks[(valid_stocks.index <= (t - pd.DateOffset(months = H))) & (valid_stocks.index > (t - pd.DateOffset(months = J)))]
    performances = pd.DataFrame()
    marketCap_sum = 1
    for n in stocks_evaluation_period.columns :
        #if n in last_year_BV['ticker'].values :
        price_slice = stocks_evaluation_period[n]
        price_slice = price_slice.dropna()
        if len(price_slice) > 0 :
            performances.loc[t, n] = (price_slice + 1).product() - 1
            mktCap = last_year_BV.loc[last_year_BV['ticker'] == n, 'marketCap'].iloc[0]
            marketCap_sum += mktCap
            performances.loc[t, n] = performances.loc[t, n] * mktCap
    # Here we have weighted performances of all stocks at the beginning of month t.
    # To find the highest and lowest 30% stocks:
    lowest_30 = int(0.3 * len(performances.columns))
    highest_30 = len(performances.columns) - lowest_30
    sorted_performances = performances.sort_values(by = performances.index[0], axis=1)
    low_return_stocks = sorted_performances.columns[ : lowest_30]
    high_return_stocks = sorted_performances.columns[highest_30 : ]
    
    # Calculating the daily momentom for the NEXT month:
    stocks_slice_t = valid_stocks.loc[t : t + pd.DateOffset(months = 1)]
    momentum_index_t = pd.DataFrame(index = stocks_slice_t.index)    
    sum_mktCap_W = 1
    for c in stocks_slice_t.columns :
        #if c in last_year_BV['ticker'].values :
        mktCap_W = last_year_BV.loc[last_year_BV['ticker'] == c, 'marketCap'].iloc[0]
        sum_mktCap_W += mktCap_W
        stocks_slice_t[c] = stocks_slice_t[c] * mktCap_W
    momentum_index_t["MOM"] = (stocks_slice_t[high_return_stocks].sum(1) - stocks_slice_t[low_return_stocks].sum(1)) / sum_mktCap_W
    
    momentum_index_t["date_g"] = stocks_slice_t.index.astype(str).str.split().str.get(0)
    momentum_index_t["date_g"] = momentum_index_t["date_g"].apply(lambda x : (datetime.strptime(x, '%Y-%m-%d')))
    momentum_index_t["date_j"] = momentum_index_t["date_g"].apply(lambda x : (JalaliDatetime(x)))
    momentum_index_t = momentum_index_t.set_index(momentum_index_t["date_j"])
    del momentum_index_t["date_g"]
    del momentum_index_t["date_j"]
    # Adding to the risk premium table:
    risk_premium.loc[momentum_index_t.index , "MOM"] = momentum_index_t["MOM"]
    
    
####### Pickle:
#risk_premium = risk_premium.dropna()
risk_premium.to_pickle("./risk_premium.pkl")
#momw = risk_premium["MOM"]
