# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich
"""

import os
import pandas as pd
import pickle as pkl
import numpy as np

kmer_model_path = "/doctorai/puneetr/T1D_analysis_final/logreg_final/logreg_out/kmer_logreg_model.pkl"
kmer_file =  "/doctorai/puneetr/T1D_analysis_final/logreg_final/logreg_out/kmer_counts_l4.pkl"
outfile = "/doctorai/puneetr/T1D_analysis_final/logreg_final/logreg_out/kmer_importance.csv"

with open(kmer_model_path, 'rb') as fh:
    model_dict = pkl.load(fh)

with open(kmer_file, 'rb') as fh:
    kmer_features = pkl.load(fh)

fold_models = model_dict['fold_models']

kmer_importance = pd.DataFrame(
        index=["ensemble_mean"] + ["ensemble_std"] + [f"model_fold_{f}" for f in range(len(fold_models))],
        columns=kmer_features.columns,
        dtype='float64'
)

for fold, fold_model in enumerate(fold_models):
    kmer_importance.loc[f"model_fold_{fold}", :] = fold_model.coef_

kmer_importance.loc["ensemble_mean", :] = kmer_importance.loc[[f"model_fold_{f}" for f in range(len(fold_models))], :].mean(0)
kmer_importance.loc["ensemble_std", :] = kmer_importance.loc[[f"model_fold_{f}" for f in range(len(fold_models))], :].std(0)
kmer_importance_sorted = kmer_importance.sort_values(by='ensemble_mean', axis=1, ascending=False)

kmer_importance_sorted.to_csv(outfile, sep=',', index=True)
