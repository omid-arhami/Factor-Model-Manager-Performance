# -*- coding: utf-8 -*-
"""
SCRIPT # 1
Created on Fri Jul 31 01:42:18 2020

@author: omid
"""
import pandas as pd
import numpy as np
import glob
from khayyam import *

#################################### Reading all the stocks
# Building the timeline:
mother_dates = pd.date_range(JalaliDate(1392, 1, 1).todate(), periods=3653, freq='d')
allStocks = pd.DataFrame(mother_dates)
allStocks.index = allStocks[0].apply(lambda x : JalaliDatetime(x))
del allStocks[0]

# Opening all the files and storing them into one DF:
path2 = r"Adjusted"
filenames = glob.glob(path2 + "\*.csv")
for filename in filenames :
    f = pd.read_csv(filename)
    #date = f["<DTYYYYMMDD>"].astype(str).str.slice(stop=4) + "/" + f["<DTYYYYMMDD>"].astype(str).str.slice(start=4,stop=6) + "/" + f["<DTYYYYMMDD>"].astype(str).str.slice(start=6,stop=8)
    date = f["<DTYYYYMMDD>"].astype(str).apply(lambda x : (JalaliDatetime.strptime(x, '%Y%m%d')))
    price = f["close"]
    ticker = f["<TICKER>"].str.split(pat="-").str.get(0)[0]
    df = pd.DataFrame(index = date.values)
    df[ticker] = price.values
    if df.iloc[0,0] != 'NaN':
        if ticker in allStocks.columns:
            mutual_temp = pd.DataFrame()
            mutual_temp[ticker] = allStocks[ticker]
            mutual_temp = mutual_temp.join(df, how='outer', lsuffix='_left', rsuffix='_right')
            mutual_temp.columns = ['a', 'a']
            allStocks[ticker] = mutual_temp.groupby(level=0, axis=1).max()
            #allStocks.merge(df, how='outer')
        if ticker not in allStocks.columns:
            allStocks = allStocks.join(df, how='outer')

# Removing Holidays:
allStocks.dropna(how='all', inplace=True)
# Removing empty stocks
allStocks.dropna(axis=1, how='all', inplace=True)
allStocks = allStocks.fillna(method='ffill')
# Calculate daily returns:
allStocks = allStocks.pct_change()#*100
allStocks.dropna(how='all', inplace=True)
# Removing inactive stocks:
for c in allStocks.columns:
    if (np.nanmax(allStocks[c]) == 0) & (np.nanmin(allStocks[c]) == 0) :
        del allStocks[c]

####### Pickle:
allStocks.to_pickle("./allStocks.pkl")
###################
