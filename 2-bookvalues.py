# -*- coding: utf-8 -*-
"""
SCRIPT # 2
Created on Fri Jul 31 01:40:28 2020

@author: omid
"""
import numpy as np
import pandas as pd
import glob
from statistics import median

#################################### Reading the yearly BookValue files
# Opening all the value files and storing them into one DF:
path = r'bookvalues'
filenames = glob.glob(path + "\*.xlsx")
dfs = []
for filename in filenames:
    # Reading file as f and saving as g
    f = pd.read_excel(filename)
    g = pd.DataFrame()
    g["ticker"] = f["نماد"]
    g["year"] = f["آخرین معامله"].str.split(pat="/").str.get(0)
    g["year"] = g["year"].astype(int)
    g["month"] = f["آخرین معامله"].str.split(pat="/").str.get(1)
    g["month"] = g["month"].astype(int)
    g["marketCap"] = f["ارزش روز"]
    g["B/M"] = 1 / f["P/BV"]
    # Keep only if last trade was in Esfand of that year:
    g = g[ (g["year"] == max(g["year"])) & (g["month"] == max(g["month"])) ]
    del g["month"]
    g = g[~g.isin([np.nan, np.inf, -np.inf]).any(1)]
    # assigning to groups based on marketCap & P_BV:
    marketCap_median = median(g["marketCap"])
    B_M_70percentile = np.percentile(np.array(g["B/M"]), 70)
    B_M_30percentile = np.percentile(np.array(g["B/M"]), 30)
    for i in g.index :
        if (  g.loc[i, "marketCap"] > marketCap_median) & (g.loc[i, "B/M"] > B_M_70percentile) :
            g.loc[i, "tag"] = 1 #BH
        elif (g.loc[i, "marketCap"] > marketCap_median) & (g.loc[i, "B/M"] <= B_M_70percentile) & (g.loc[i, "B/M"] > B_M_30percentile) :
            g.loc[i, "tag"] = 2 # BM
        elif (g.loc[i, "marketCap"] > marketCap_median) & (g.loc[i, "B/M"] <= B_M_30percentile) :
            g.loc[i, "tag"] = 3 # BL
        elif (g.loc[i, "marketCap"] <= marketCap_median) & (g.loc[i, "B/M"] > B_M_70percentile) :
            g.loc[i, "tag"] = 4 # SH
        elif (g.loc[i, "marketCap"] <= marketCap_median) & (g.loc[i, "B/M"] <= B_M_70percentile) & (g.loc[i, "B/M"] > B_M_30percentile) :
            g.loc[i, "tag"] = 5 # SM
        elif (g.loc[i, "marketCap"] <= marketCap_median) & (g.loc[i, "B/M"] <= B_M_30percentile) :
            g.loc[i, "tag"] = 6 # SL    
    dfs.append(g)
    
# Concatenate all data into one DataFrame
bookvalues = pd.concat(dfs, ignore_index=True)

####### Pickle:
bookvalues.to_pickle("./bookvalues.pkl")