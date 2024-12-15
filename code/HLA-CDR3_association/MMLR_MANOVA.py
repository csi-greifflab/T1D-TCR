#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 15:08:05 2023

@author: puneetr
"""


import os
import pandas as pd
import numpy as np
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import AnchoredText
# import statsmodels.api as sm
from sklearn import linear_model
from statsmodels.stats.anova import anova_lm
from scipy.stats import f
import subprocess
import rpy2.robjects as robjects
import statsmodels.stats.multitest as sm

mut_path = "./mutation_sites/"
rep_path = "./hla_rep/"
cdr_freq_path = "./CDR_freq/"
full_meta_df = pd.read_csv("metadata.csv")

## only change
meta_df = full_meta_df

hla_pca = pd.read_csv("all_hla_pca.csv")

aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

pd_out_df = pd.DataFrame(columns=["length","hla","hla_site","cdr_pos","pvalue"])
for length_run in range(12,19):
    cdr_df = pd.read_csv(cdr_freq_path+"CDR_freq_"+str(length_run)+".tsv", sep = "\t")
    cdr_df = cdr_df[(cdr_df['pos'] >= 107) & (cdr_df['pos'] <= 116)]
    for hgene in ["A", "B", "C", "DPA1", "DPB1", "DQA1", "DQB1", "DRB1"]:
        temp_df = pd.read_csv(mut_path+hgene+"_mutation_list.txt", sep = " ", names = ["pos_res","HLA"])
        temp_df["pos"] = temp_df.pos_res.apply(lambda x:x.split("_")[0]) 
        temp_df["res"] = temp_df.pos_res.apply(lambda x:x.split("_")[1])
        temp_df['donor'] = temp_df.HLA.apply(lambda x:x.split("_")[2])
        rep_df = pd.read_csv(rep_path+"HLA_matrix_"+hgene+".csv")
        out_df = pd.DataFrame(columns=list(temp_df.pos.unique()), index = rep_df.filename)
        
        temp_df.donor = temp_df.donor.apply(lambda x:x.replace('.1', ''))
        all_var_df = temp_df.groupby(['donor', 'pos', 'res']).pos_res.count().reset_index()
        
        ## getting the list of donor and filenames for mapping
        col_map = meta_df.groupby('donor')['filename'].apply(lambda x: x.iloc[0]).to_dict()
        
        ## extract for each HLA position
        for i in all_var_df.pos.unique():
            temp2 = all_var_df[all_var_df.pos==i]
            # Create a pivot table
            df_out = pd.pivot_table(temp2, values='pos_res', index='res', columns='donor', aggfunc=lambda x: x)
            df_out.fillna(0, inplace=True)

            df_out = df_out.T
            df_out = df_out.drop(df_out.sum().idxmax(), axis=1).reset_index()
            
            ### replace the column name donor with the filename
            df_out['filename'] = df_out.donor.apply(lambda x:col_map[int(x)])
            df_out = df_out.drop('donor', axis = 1)
            df_out.columns = [x+"_allele" if x!="filename" else "filename" for x in df_out.columns]

            #### reading CSV
            df_out = df_out.merge(hla_pca, how="left", on = "filename")

            for j in list(cdr_df.pos.unique()): ## change
                freq_df = pd.DataFrame(np.nan, columns = aa, index = full_meta_df.filename)
                cdr_temp_df = cdr_df[cdr_df.pos==j]
                for index, row in cdr_temp_df.iterrows():
                    freq_df.loc[row['rep'], row['aa']] = row['count']
                freq_df_norm = freq_df.div(freq_df.sum(axis=1), axis=0)
                freq_df_norm = freq_df_norm.loc[meta_df.filename] ## fliter row index based on this
                
                int_df = pd.DataFrame(np.nan, columns = aa, index = meta_df.filename)
                for k in aa:
                    int_df[k] = norm.ppf((freq_df_norm[k].rank(method='min', na_option='keep') - 0.5)/np.sum(~np.isnan(freq_df_norm[k])))

                MMLM_df = df_out.merge(int_df, how= "left", on = "filename" )
                MMLM_df = MMLM_df.fillna(0)
                int_df = int_df.fillna(0)
                cols_aa = list(int_df.loc[:, (int_df!=0).any(axis=0)].columns)
                MMLM_df.to_csv("temp_df_r.csv", index = False)
                
                ###### this part calculates MMLR ##################
                ## Python does not have a library to calculate MMLR so this R script is used to run and get output 
                robjects.r('''MMLM_df <- read.csv("temp_df_r.csv")
                           mod1 <- lm( 
                cbind('''+",".join(cols_aa)+''') ~'''+" + ".join([x for x in list(df_out.columns) if x!="filename"])+''', data = MMLM_df)
                
                mod0 <- lm(cbind('''+",".join(cols_aa)+''') ~ PC1+PC2+PC3, data = MMLM_df)
                
                test <- anova(mod1, mod0)
                pvalue_out = test$"Pr(>F)"[2]
                ''')
                pvalue = robjects.r['pvalue_out'][0]
                pd_out_df.loc[len(pd_out_df)] = [length_run,hgene,i,j,pvalue]
                
## correction of p-value
pd_out_df1 = pd_out_df[pd.notna(pd_out_df['pvalue'])]
adj_p_val = sm.fdrcorrection(pd_out_df1.pvalue.to_list(), alpha=0.05, method='indep', is_sorted=False)[1]
pd_out_df1['adj_p_value'] = adj_p_val
sig_df = pd_out_df1[pd_out_df1.adj_p_value<=0.05]


pd_out_df1.to_csv("all_p_value_MMLR_MANOVA.csv", index = False)
