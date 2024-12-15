# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich
"""
import argparse
from helpers import load_data, apply_ensemble
from make_kmer_dataset import make_kmer_dataset
from sklearn import metrics

parser = argparse.ArgumentParser(description='Creating kmer features from fasta files')
parser.add_argument('sequence_files_dir', type=str, help='folder that contains fasta files (if kmer_file already exists, use "none" here)')
parser.add_argument('kmer_file', type=str, help='path to pickle file that contains kmer counts')
parser.add_argument('metadata_file', help='path to metadata CSV')
parser.add_argument('trained_model', help='path to where pickle file containing trained model shall be stored')

args = parser.parse_args()
sequence_files_dir = args.sequence_files_dir
kmer_file = args.kmer_file
metadata_file = args.metadata_file
trained_model = args.trained_model

# Create kmers (skip if you already created kmer_file)
if kmer_file is 'none':
    make_kmer_dataset(sequence_files_dir, kmer_file)

# Load data
kmer_features, metadata, _ = load_data(kmer_file, metadata_file, None, normalize=False)

# Apply model
sample_ids = metadata.index.values
x = kmer_features.loc[sample_ids]
# x should be a array of shape [n_samples, n_kmers]. You can also just load all samples via kmer_features.values.
# kmer_features.loc[sample_ids] just makes sure that the samples from the metadata file correspond to the samples of the loaded kmers.
predictions, _ = apply_ensemble(x=x, trained_model_path=trained_model)

# Compute AUC
y = (metadata.loc[sample_ids, 'diabetes_status'] == 'T1D').values
auc = metrics.roc_auc_score(y_true=y, y_score=predictions)
print(f"ROC AUC: {auc}")
