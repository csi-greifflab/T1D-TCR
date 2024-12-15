# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich
"""
import argparse
import os
import numpy as np
import pandas as pd
import pickle as pkl
from helpers_bkup import load_data, apply_ensemble

parser = argparse.ArgumentParser(description='Creating kmer features from fasta files')
parser.add_argument('kmer_file', type=str, help='path to pickle file that contains kmer counts')
parser.add_argument('metadata_file', help='path to metadata CSV')
parser.add_argument('cv_split_file')
parser.add_argument('trained_model', help='path to where pickle file containing trained model shall be stored')
parser.add_argument('output_folder')

args = parser.parse_args()
kmer_file = args.kmer_file
metadata_file = args.metadata_file
cv_split_file = args.cv_split_file
trained_model = args.trained_model
output_folder = args.output_folder

os.makedirs(output_folder, exist_ok=True)
kmer_features, metadata, cv_split_inds = load_data(kmer_file, metadata_file, cv_split_file, normalize=False)


# Create predictions for hold-out testset with all fold models and average their predictions
test_set_mask = metadata['dataset'] != 1
train_ids = metadata.index[~test_set_mask].values
test_ids = metadata.index[test_set_mask].values

x_train = kmer_features.loc[train_ids]
x_test = kmer_features.loc[test_ids]

y_train = (metadata.loc[train_ids, 'diabetes_status'] == 'T1D').values
y_test = (metadata.loc[test_ids, 'diabetes_status'] == 'T1D').values

pred_train, individual_fold_model_predictions = apply_ensemble(x=x_train, trained_model_path=trained_model)
pred_test, _ = apply_ensemble(x=x_test, trained_model_path=trained_model)

metadata.loc[train_ids, 'predictions'] = pred_train
metadata.loc[test_ids, 'predictions'] = pred_test

metadata.to_csv(os.path.join(output_folder, 'predictions.csv'), sep=',')

from matplotlib import pyplot as plt
from sklearn import metrics
for dset in ['cv_data', 'hold-out']:
    if dset == 'cv_data':
        subtypes = ['CTRL', 'SDR', 'FDR', 'T1D']
        dset_mask = metadata['dataset'] == 1
    else:
        subtypes = ['CTRL', 'T1D']
        dset_mask = metadata['dataset'] != 1
    dset_metadata = metadata[dset_mask]
    y_score = dset_metadata['predictions']
    
    fig, ax = plt.subplots()
    fpr, tpr, thresholds = metrics.roc_curve(y_true=dset_metadata['diabetes_status'] == 'T1D', y_score=y_score)
    auc = metrics.roc_auc_score(y_true=dset_metadata['diabetes_status'] == 'T1D', y_score=y_score)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc * 100:0.4f}%)')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC {dset}')
    plt.legend(loc="lower right")
    fig.savefig(fname=os.path.join(output_folder, f"roc dset {dset}.png"))
    
    fig, ax = plt.subplots()
    ax.set_title(f'Scores {dset}')
    for subtype_i, (subtype, color) in enumerate(zip(subtypes, ['blue', 'green', 'orange', 'red'])):
        x = y_score[dset_metadata['diabetes_status'] == subtype]
        ax.scatter(x, np.ones_like(x) * subtype_i, label=subtype, color=color, marker='x', alpha=0.3)
        plt.xlim([np.min(y_score), np.max(y_score)])
    
    ax.set_xlabel('score')
    plt.yticks(np.arange(len(subtypes)), subtypes)
    ax.set_ylabel('group')
    fig.savefig(fname=os.path.join(output_folder, f"roc_subclasses dset {dset}.png"))
    
    fig, ax = plt.subplots()
    ax.set_title(f'Scores {dset}')
    y = [y_score[dset_metadata['diabetes_status'] == subtype] for subtype in subtypes]
    ax.boxplot(y)
    ax.set_xlabel('group')
    ax.set_xticklabels(subtypes)
    ax.set_ylabel('score')
    fig.savefig(fname=os.path.join(output_folder, f"subclasses_box dset {dset}.png"))
    
    fig, ax = plt.subplots()
    ax.set_title(f'Scores {dset}')
    y = [y_score[dset_metadata['diabetes_status'] == subtype] for subtype in subtypes]
    ax.violinplot(y)
    ax.set_xlabel('group')
    ax.set_xticks(np.arange(len(subtypes)) + 1)
    ax.set_xticklabels(subtypes)
    fig.savefig(fname=os.path.join(output_folder, f"subclasses_violin dset {dset}.png"))
    
    fig, ax = plt.subplots()
    ax.set_title(f'Scores vs age {dset}')
    
    age = np.array(dset_metadata['age'].values, dtype=np.float)
    scores = y_score
    scores = scores[np.isfinite(age)]
    age = age[np.isfinite(age)]
    ax.scatter(scores, age, marker='x', alpha=0.3)
    plt.xlim([np.min(scores), np.max(scores)])
    plt.ylim([np.min(age), np.max(age)])
    np.array(metadata['predictions'].values[metadata['predictions'].values != -1], dtype=np.float)
    ax.set_xlabel('predictions')
    ax.set_ylabel('age')
    fig.savefig(fname=os.path.join(output_folder, f"age dset {dset}.png"))
    
    #
    # Subclasses
    #
    combinations = [
        ['FDR', 'CTRL'],
        ['FDR', 'SDR'],
        ['SDR', 'CTRL'],
        ['T1D', 'CTRL'],
        ['T1D', 'SDR'],
        ['T1D', 'FDR']]
    subclasses_output_folder = os.path.join(output_folder, "subclasses")
    os.makedirs(subclasses_output_folder, exist_ok=True)
    
    for combination in combinations:
        print(f"combination: {combination}")
        comb_mask = np.logical_or(dset_metadata['diabetes_status'] == combination[0], dset_metadata['diabetes_status'] == combination[1])
        comb_metadata = dset_metadata[comb_mask]
        y_true = comb_metadata['diabetes_status'] == combination[0]
        y_score = comb_metadata['predictions']
        try:
            auc = metrics.roc_auc_score(y_true=y_true, y_score=y_score)
            
            fig, ax = plt.subplots()
            fpr, tpr, thresholds = metrics.roc_curve(y_true=y_true, y_score=y_score)
            ax.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve {combination[0]} vs. {combination[1]} (area = {auc * 100:0.4f}%)')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC {dset}\n{combination[0]} vs. {combination[1]}')
            plt.legend(loc="lower right")
            fig.savefig(fname=os.path.join(subclasses_output_folder, f"{combination[0]}_{combination[1]} dset{dset}.png"))
        except:
            pass
