# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich
"""
import copy
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from sklearn.metrics import roc_auc_score
from typing import Callable, List, Tuple, Union
import tqdm

PRECOMPUTED_KERNEL: Union[None, pd.DataFrame] = None


def load_data(kmer_file: str, metadata_file: str, cv_split_file: Union[None, str], normalize: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Union[None,List[np.ndarray]]]:
    print(f"Loading kmers from {kmer_file}")
    with open(kmer_file, 'rb') as fh:
        kmer_features = pkl.load(fh)
    kmer_features = kmer_features.astype('float64', copy=False)
    if normalize:
        kmer_features.values[:] = (kmer_features.values - kmer_features.mean(axis=1)[:, None]) / kmer_features.std(axis=1)[:, None]
    else:
        kmer_features.values[:] = kmer_features.values / kmer_features.sum(axis=1)[:, None]
    
    print(f"Loading metadata from {metadata_file}")
    metadata = pd.read_csv(metadata_file, sep=',', index_col=0, header=0)
    if cv_split_file is not None:
        print(f"Loading cross-validation splits from {cv_split_file}")
        with open(cv_split_file, 'rb') as fh:
            split = pkl.load(fh)
        cv_split_inds = split['inds']
    else:
        cv_split_inds = None
    return kmer_features, metadata, cv_split_inds


def uniform_age_distribution_weights(age_per_sample: np.ndarray, min_age: float, max_age: float, age_bin_size: float = 1):
    """Takes an array of sample ages and returns sample weights such that the age distribution appears uniform within
    range [min_age, max_age]
    """
    # Use integer age values as bins for histogram
    n_bins = int(np.ceil((max_age - min_age) / age_bin_size))

    # Compute the histogram of the age values with the specified number of bins
    hist, edges = np.histogram(age_per_sample, bins=n_bins, range=(min_age, max_age))

    # Compute the desired uniform histogram
    uniform_hist = np.full_like(hist, fill_value=(len(age_per_sample) / n_bins))

    # Compute the weights for each age bin by dividing the uniform histogram by the actual histogram
    weights = np.asarray(uniform_hist, dtype=np.float) / np.asarray(hist, dtype=np.float)

    # Assign a weight to each sample based on its age bin
    age_bins = np.digitize(age_per_sample, bins=edges[:-1])
    weight_per_sample = weights[age_bins - 1] / np.sum(weights[age_bins - 1])

    return weight_per_sample


def get_fold_inds(cv_split_inds: List[np.ndarray], fold: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return training set indices, validation folds to be used with PredefinedSplit, and test set indices given list of indices in splits"""
    cv_split_inds = copy.copy(cv_split_inds)
    test_inds = cv_split_inds.pop(fold)
    
    train_inds = np.concatenate(cv_split_inds)
    validation_fold = np.concatenate([np.full_like(cv_split_inds[inner_fold], fill_value=inner_fold)
                                      for inner_fold in range(len(cv_split_inds))])
    
    return (train_inds, validation_fold, test_inds)


def minmax_kernel_raw(samples_1: np.ndarray, samples_2: np.ndarray) -> np.ndarray:
    """Custom MinMax Kernel for sklearn SVMs. Assumes all feature values to be in range [0, inf].

    Parameters
    -------
    samples_1
        Sample matrix of shape (n_samples_1, n_features), where n_samples is the number of samples and n_features is
         the number of features.
    samples_2
        Sample matrix of shape (n_samples_2, n_features), where n_samples is the number of samples and n_features is
         the number of features.
    Returns
    -------
    Gram_matrix
        Returns the Gram matrix of minmax_kernel(samples_1, samples_2) of shape (n_samples_1, n_samples_2)
    """
    if np.any(samples_1 < 0) or np.any(samples_2 < 0):
        raise ValueError(f"MinMax Kernel is not defined for feature values < 0"
                         f" but found {min(np.min(samples_1 < 0), np.min(samples_2 < 0))}!")
    max_scores = np.maximum(samples_1[:, None], samples_2[None]).sum(axis=-1)
    min_scores = np.minimum(samples_1[:, None], samples_2[None]).sum(axis=-1)
    # -> (n_samples_1, n_samples_2)
    min_max_scores = min_scores / max_scores
    min_max_scores[max_scores == 0] = 1.  # In case both samples don't contain any values >0, set score to 1.
    return min_max_scores


def minmax_kernel(samples_1: Union[np.ndarray, pd.DataFrame], samples_2: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """Custom MinMax Kernel for sklearn SVMs. Assumes all feature values to be in range [0, inf].

    Parameters
    -------
    samples_1
        Sample matrix of shape (n_samples_1, n_features), where n_samples is the number of samples and n_features is
         the number of features.
    samples_2
        Sample matrix of shape (n_samples_2, n_features), where n_samples is the number of samples and n_features is
         the number of features.
    Returns
    -------
    Gram_matrix
        Returns the Gram matrix of minmax_kernel(samples_1, samples_2) of shape (n_samples_1, n_samples_2)
    """
    if np.any(samples_1 < 0) or np.any(samples_2 < 0):
        raise ValueError(f"MinMax Kernel is not defined for feature values < 0"
                         f" but found {min(np.min(samples_1 < 0), np.min(samples_2 < 0))}!")
    if PRECOMPUTED_KERNEL is not None:
        # Check if sample IDs are in PRECOMPUTED_KERNEL
        # samples_1_inds = samples_1.index.isin(PRECOMPUTED_KERNEL.index)
        # samples_2_inds = samples_2.index.isin(PRECOMPUTED_KERNEL.columns)
        # if samples_1_inds.all() and samples_2_inds.all():
        try:
            return PRECOMPUTED_KERNEL.loc[samples_1.index.values, samples_2.index.values].values
        except (KeyError, AttributeError):
            pass
    
    return compute_min_max_scores(samples_1, samples_2)


def compute_min_max_scores(samples_1: Union[np.ndarray, pd.DataFrame], samples_2: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    is_dataframe = isinstance(samples_1, pd.DataFrame) and isinstance(samples_2, pd.DataFrame)
    if is_dataframe:
        index = samples_1.index
        samples_1 = samples_1.values
        columns = samples_2.index
        samples_2 = samples_2.values
    elif isinstance(samples_1, pd.DataFrame) or isinstance(samples_2, pd.DataFrame):
        is_dataframe = False
        samples_1 = samples_1.values
        samples_2 = samples_2.values
    
    min_max_scores = np.empty((len(samples_1), len(samples_2)), dtype=np.float64)
    for samples_1_i in tqdm.tqdm(range(len(samples_1))):
        min_max_scores[samples_1_i] = np.minimum(samples_1[samples_1_i:samples_1_i+1], samples_2).sum(axis=-1)
        min_max_scores[samples_1_i] /= np.maximum(samples_1[samples_1_i:samples_1_i+1], samples_2).sum(axis=-1)
    min_max_scores[~np.isfinite(min_max_scores)] = 1.  # In case both samples don't contain any values >0, set score to 1.
    if is_dataframe:
        min_max_scores = pd.DataFrame(index=index, columns=columns, data=min_max_scores)
    return min_max_scores


def nested_cv_for_estimator(kmer_features: pd.DataFrame, metadata: pd.DataFrame, cv_split_inds: List[np.ndarray],
                            outfile: str, estimator_class: Callable, param_dist: dict, random_state: int = 0,
                            n_search_iter: int = 100, n_jobs: int = -1, precompute_min_max_kernel: bool = False):
    # Initialize empty lists to store the results for each fold
    n_folds = len(cv_split_inds)
    best_params_list = []
    train_scores_list = []
    test_scores_list = []
    fold_models = []

    if precompute_min_max_kernel:
        x_train = kmer_features.loc[metadata.index[np.concatenate(cv_split_inds)].values]
        global PRECOMPUTED_KERNEL
        try:
            with open('PRECOMPUTED_KERNEL.pkl', 'rb') as fh:
                PRECOMPUTED_KERNEL = pkl.load(fh)
        except:
            PRECOMPUTED_KERNEL = compute_min_max_scores(x_train, x_train)
            with open('PRECOMPUTED_KERNEL.pkl', 'wb') as fh:
                pkl.dump(PRECOMPUTED_KERNEL, fh)
    
    # Loop over the folds
    for f in range(n_folds):
        print(f"Outer CV fold {f}/{n_folds}")
        train_inds, validation_fold, test_inds = get_fold_inds(cv_split_inds, f)
        train_ids = metadata.index[train_inds].values
        test_ids = metadata.index[test_inds].values
        
        x_train = kmer_features.loc[train_ids]
        x_test = kmer_features.loc[test_ids]
        
        y_train = (metadata.loc[train_ids, 'diabetes_status'] == 'T1D').values
        y_test = (metadata.loc[test_ids, 'diabetes_status'] == 'T1D').values
        
        ages_train = metadata.loc[train_ids, 'age'].values
        ages_sample_weights = metadata.loc[train_ids, 'weight'].values
        # ages_sample_weights = np.full_like(ages_train, fill_value=1./len(ages_train))
        
        # Initialize the logistic regression model with the hyperparameters to search over
        estimator = estimator_class(random_state=random_state)
        
        # Set up the grid search for hyperparameters
        random_search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=param_dist,
                n_iter=n_search_iter,
                cv=PredefinedSplit(test_fold=validation_fold),  # 4-fold inner CV
                scoring='roc_auc',
                n_jobs=n_jobs,
                verbose=10,
                random_state=random_state
        )
        
        # Fit the random search to the train data with the sample weights
        random_search.fit(x_train, y_train, sample_weight=ages_sample_weights)
        
        # Record the best hyperparameters
        best_params_list.append(random_search.best_params_)
        
        # Train a logistic regression model on the full train set with the best hyperparameters
        lr_best = estimator_class(**random_search.best_params_)
        lr_best.fit(x_train, y_train, sample_weight=ages_sample_weights)
        
        # Evaluate the model on the train and test sets without sample weighting
        y_train_pred = lr_best.predict_proba(x_train)[:, 1]
        y_test_pred = lr_best.predict_proba(x_test)[:, 1]
        train_score = roc_auc_score(y_train, y_train_pred)
        test_score = roc_auc_score(y_test, y_test_pred)
        train_scores_list.append(train_score)
        test_scores_list.append(test_score)
        
        # Store the trained model for this fold
        fold_models.append(lr_best)
    
    # Average the train and test scores over the folds
    train_score_avg = np.mean(train_scores_list)
    test_score_avg = np.mean(test_scores_list)
    
    # Save the results to a pickle file
    results = {'best_params': best_params_list,
               'train_scores': train_scores_list,
               'test_scores': test_scores_list,
               'train_score_avg': train_score_avg,
               'test_score_avg': test_score_avg,
               'fold_models': fold_models}
    with open(outfile, 'wb') as f:
        pkl.dump(results, f)
    print(f"Output saved to {outfile}")
    
    print('Best hyperparameters:')
    for i, params in enumerate(best_params_list):
        print(f'Fold {i + 1}: {params}')
    print(f'Average train AUC: {train_score_avg:.3f}')
    print(f'Average test AUC: {test_score_avg:.3f}')


def apply_ensemble(x, trained_model_path):
    with open(trained_model_path, 'rb') as fh:
        results = pkl.load(fh)
    fold_models = results['fold_models']
    individual_y = np.stack([fold_model.predict_proba(x)[:, 1] for fold_model in fold_models], -1)
    ensemble_y = individual_y.mean(-1)
    return ensemble_y, individual_y

