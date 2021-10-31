# -*- coding: utf-8 -*-
"""
SCRIPT # 5
daily returns of 30 mutual funds using NAV.

Created on Sat Aug  1 18:52:11 2020

@author: omid
"""
import pandas as pd
import glob
from khayyam import *

risk_premium = pd.read_pickle("./risk_premium.pkl")

#################################### Reading all the stocks
# Building the timeline:
allFunds = pd.DataFrame(index=risk_premium.index)

# Opening all the files and storing them into one DF:
path3 = r"Funds"
filenames = glob.glob(path3 + "\*.xlsx")
for filename in filenames :
    f = pd.read_excel(filename)
    #date = f["<DTYYYYMMDD>"].astype(str).str.slice(stop=4) + "/" + f["<DTYYYYMMDD>"].astype(str).str.slice(start=4,stop=6) + "/" + f["<DTYYYYMMDD>"].astype(str).str.slice(start=6,stop=8)
    date = f["Date"].astype(str).apply(lambda x : (JalaliDatetime.strptime(x, '%Y/%m/%d')))
    NAVEbtal = f["NAVEbtal"]
    name = f["name"][0]
    df = pd.DataFrame(index = date.values)
    df[name] = NAVEbtal.values
    if df.iloc[0,0] != 'NaN':
        allFunds = allFunds.join(df, how='left', lsuffix='_left', rsuffix='_right')


# Removing empty columns:
allFunds.dropna(axis=1, how='all', inplace=True)
allFunds = allFunds[~allFunds.index.duplicated(keep='first')]
allFunds = allFunds.fillna(method='ffill')
# Calculate daily returns:
allFunds = allFunds.pct_change()#*100
allFunds.dropna(how='all', inplace=True)


####### Pickle:
allFunds.to_pickle("./allFunds.pkl")
