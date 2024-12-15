#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 19:49:14 2023

@author: puneetr
"""

############# Code to calcualte CDR3 phenotypes #################

import os
import pandas as pd
import numpy as np
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnchoredText
import statsmodels.api as sm
from scipy.stats import f
import subprocess
import rpy2.robjects as robjects
from statsmodels.stats.multitest import fdrcorrection

running_for = "cohort1" ### 'CTRL', 'FDR', 'T1D', 'SDR', 'cohort1'
mut_path = "./mutation_sites/"
rep_path = "./hla_rep/"
cdr_freq_path = "./CDR_freq/"
full_meta_df = pd.read_csv("metadata.csv") ## meta data file
meta_df = full_meta_df ### add only for cohort 1

hla_risk_df = pd.read_csv("./hla_risk_score_"+running_for+".csv")

aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']


pd_out_df = pd.DataFrame(columns=["cdr_length","cdr_pos","CDR3_aa","r_sq","coef_est","std_err","t_val","p_val"])

for length_run in range(12,19):
    print(str(length_run))

    df_out = pd.DataFrame(index = meta_df.filename)
    df_out.rename(columns={'index': 'filename'}, inplace=True)
    ## reading CSV
    df_out = df_out.merge(hla_risk_df[['norm_HLA_risk_score','filename']], how="left", on = "filename")

    cdr_df = pd.read_csv(cdr_freq_path+"CDR_freq_"+str(length_run)+".tsv", sep = "\t")
    for j in [x for x in list(cdr_df.pos.unique()) if x not in [104,105,106,117,118]]: ## change
        freq_df = pd.DataFrame(np.nan, columns = aa, index = full_meta_df.filename)
        cdr_temp_df = cdr_df[cdr_df.pos==j]
        for index, row in cdr_temp_df.iterrows():
            freq_df.loc[row['rep'], row['aa']] = row['count']
        freq_df_norm = freq_df.div(freq_df.sum(axis=1), axis=0)
        freq_df_norm = freq_df_norm.loc[meta_df.filename] ## fliter row index based on this
        
        int_df = pd.DataFrame(np.nan, columns = aa, index = meta_df.filename)
        for k in aa:
            int_df[k] = norm.ppf((freq_df_norm[k].rank(method='min', na_option='keep') - 0.5)/np.sum(~np.isnan(freq_df_norm[k])))
        # response = int_df.fillna(0).values
        MMLM_df = df_out.merge(int_df, how= "left", on = "filename" )
        MMLM_df = MMLM_df.fillna(0)
        int_df = int_df.fillna(0)
        cols_aa = list(int_df.loc[:, (int_df!=0).any(axis=0)].columns)
        # MMLM_df.to_csv("/Users/puneetr/Documents/Research/T1D/new_analysis/HLA_CDR3/HLA_CDR_analysis/HLA_site_T1D/temp_df_r.csv", index = False)
        
        #### new python version for LM analysis (only for multiple linear regression and not for MMLM)
        for l in cols_aa:
            y = MMLM_df[l]
            X = df_out['norm_HLA_risk_score']
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            summary = model.summary()
            coef_row = summary.tables[1].data[2]
            pd_out_df.loc[len(pd_out_df)] = [length_run,j,l,model.rsquared,float(coef_row[1]),float(coef_row[2]),float(coef_row[3]),model.pvalues[1]]
        
pd_out_df['corrected_p_val'] = fdrcorrection(pd_out_df.p_val, alpha=0.05, method='indep')[1]

pd_out_df.to_csv("./LR_test_CDR3_phenotype.csv", index = False)

# Estimate	Std. Error	t value	Pr(>|t|)