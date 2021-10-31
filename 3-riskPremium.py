# -*- coding: utf-8 -*-
"""
SCRIPT # 3
Created on Fri Jul 31 01:40:28 2020

@author: omid
"""
import numpy as np
import pandas as pd
import glob
from khayyam import *

allStocks = pd.read_pickle("./allStocks.pkl")
bookvalues = pd.read_pickle("./bookvalues.pkl")
############################ compute a return index (series) for each of the 
############################ six groups by weighting the returns by market capitalization
years = [1392, 1393, 1394, 1395, 1396, 1397, 1398]
risk_premium = pd.DataFrame(columns = [1,2,3,4,5,6], index = allStocks.index)
for y in years :
    this_year_values = bookvalues[bookvalues["year"] == y-1]
    for tag in range(1,7) : # tags
        this_group = this_year_values[this_year_values["tag"] == tag]
        # Building the timeline and return index = 0:
        year_days = pd.date_range(JalaliDate(y, 1, 1).todate(), periods=365, freq='d')
        return_index = pd.DataFrame(year_days)
        return_index.index = return_index[0].apply(lambda x : JalaliDatetime(x))
        del return_index[0]
        
        marketCap_sum = 1
        companies = this_group.ticker.unique()
        for c in companies :
            if c in allStocks.columns :
                r = allStocks[c]
                mktCap = this_group.loc[this_group['ticker'] == c, 'marketCap'].iloc[0]
                marketCap_sum += mktCap
                return_index = return_index.join(r, how='inner', lsuffix='_left', rsuffix='_right')
                return_index[c] = return_index[c] * mktCap
        return_index['sum'] = return_index.loc[:,:].sum(1) / marketCap_sum
        risk_premium.loc[return_index.index, tag] = return_index['sum']

risk_premium["HML"] = (risk_premium[1] + risk_premium[4] - risk_premium[3] - risk_premium[6])/2
risk_premium["SMB"] = (risk_premium[4] + risk_premium[5] + risk_premium[6] - risk_premium[3] - risk_premium[2] - risk_premium[1])/3
risk_premium.drop(columns=[1,2,3,4,5,6], inplace=True)


############################# return on the market portfolio
TEPIX = pd.read_csv("Overall Index.csv")
date = TEPIX["<DTYYYYMMDD>"].astype(str).apply(lambda x : (JalaliDatetime.strptime(x, '%Y%m%d')))
price = TEPIX["close"]
TEPIX_df = pd.DataFrame(index = date.values)
TEPIX_df["TEPIX"] = price.values

# Adding to the main DF:
risk_premium.loc[risk_premium.index, "TEPIX"] = TEPIX_df["TEPIX"]
risk_premium["TEPIX"] = risk_premium["TEPIX"].fillna(method='ffill')
risk_premium["TEPIX"] = risk_premium["TEPIX"].pct_change()



################################## Risk free 
allTB = pd.DataFrame(index = risk_premium.index)
# Opening all the files and storing them into one DF:
path3 = r"akhza"
filenames = glob.glob(path3 + "\*.csv")
countor = 0
for filename in filenames :
    countor += 1
    f = pd.read_csv(filename, header = None)
    f = f[~f[0].duplicated()]
    #date = f[0].astype(str).str.slice(stop=4) + "/" + f[0].astype(str).str.slice(start=5,stop=6) + "/" + f["<DTYYYYMMDD>"].astype(str).str.slice(start=6,stop=8)    
    #f[0] = f[0].str.split(pat="/").str.get(0) + "/" + f[0].str.split(pat="/").str.get(1)
    f[0] = f[0].str.split().str.get(0) # date column
    date = f[0].astype(str).apply(lambda x : (JalaliDatetime.strptime(x, '%Y/%m/%d')))
    ret = f[1].apply(lambda x : np.exp(np.log(1+x)/365) - 1) # daily rf return
    #ret = ret.fillna(method='ffill')
    df = pd.DataFrame(index = date.values)
    df[str(countor)] = ret.values
    allTB.loc[allTB.index, str(countor)] = df[str(countor)]

allTB['mean'] = allTB.mean(axis = 1, skipna = True)
allTB['mean'] = allTB['mean'].fillna(method='bfill')

# Adding to the main DF:
risk_premium.loc[risk_premium.index, "Rf"] = allTB['mean']
risk_premium = risk_premium[~risk_premium.isin([np.nan, np.inf, -np.inf]).any(1)]


####### Pickle:
risk_premium.to_pickle("./risk_premium.pkl")