# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich
"""
import argparse
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform
from helpers import load_data, nested_cv_for_estimator

parser = argparse.ArgumentParser(description='Creating kmer features from fasta files')
parser.add_argument('kmer_file', type=str, help='path to pickle file that contains kmer counts')
parser.add_argument('metadata_file', type=str, help='path to metadata CSV')
parser.add_argument('cv_split_file', type=str, help='path to pickle file with cross-validation split indices')
parser.add_argument('outfile', type=str, help='path to where pickle file containing trained model shall be stored')
parser.add_argument('--n_search_iter', type=int, help='number of random search iterations')
parser.add_argument('--n_jobs', type=int, help='number of worker jobs for cross validation (set to -1 to autoscale to number of CPUs)')

args = parser.parse_args()

kmer_file = args.kmer_file
metadata_file = args.metadata_file
cv_split_file = args.cv_split_file
outfile = args.outfile
n_search_iter = args.n_search_iter
n_jobs = args.n_jobs

kmer_features, metadata, cv_split_inds = load_data(kmer_file, metadata_file, cv_split_file, normalize=True)
estimator_class = LogisticRegression
param_dist = {
    'penalty': [ 'l1'],
    # 'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    # 'C': uniform(0.1, 10),
    'C': uniform(0.001, 10),
    # 'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'max_iter': [100, 500, 1000, 5000]
    # 'max_iter': [100]
}

### changed here by puneet penalty removed : 'l1',
### removed solver 'liblinear'
# param_dist = {
# }
try:
    nested_cv_for_estimator(kmer_features, metadata, cv_split_inds, outfile, estimator_class, param_dist,
                        n_search_iter=n_search_iter, n_jobs=n_jobs, precompute_min_max_kernel=False)
except:
    1
print("Done!")
