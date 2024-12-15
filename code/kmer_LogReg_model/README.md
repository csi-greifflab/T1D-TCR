# Training a kmer model

Open a terminal and `cd` into the directory where this file is located at.

## Creating the data

```shell
python make_kmer_dataset.py sequence_files_dir output_file
```
e.g.
```shell
python make_kmer_dataset.py "/media/greifflab/t1d/sequences_purged" "/media/greifflab/t1d/kmer_counts_l4.pkl"
```

will store the kmer counts in pickle file `output_file`.

## Training a logistic regression kmer model
(Modify file to change parameter search-space.)
```shell
python kmer_logreg.py kmer_file metadata_file cv_split_file outfile --n_search_iter --n_jobs
```
e.g.
```shell
python kmer_logreg.py  "/media/greifflab/t1d/kmer_counts_l4.pkl" "/media/greifflab/t1d/metadata.csv" "/media/greifflab/t1d/cv_splits_dataset1.pkl" "/media/greifflab/t1d/kmer_logreg_model.pkl"  --n_search_iter 1 --n_jobs 1
```

# Applying a trained model
```shell
python inference.py sequence_files_dir kmer_file metadata_file trained_model
```
e.g.
```shell
python inference.py "/media/greifflab/t1d/sequences_purged" "/media/greifflab/t1d/kmer_counts_l4.pkl" "/media/greifflab/t1d/metadata.csv" "/media/greifflab/t1d/kmer_logreg_model.pkl"
```

# Analyzing the trained T1D model
```shell
python analyse_results.py kmer_file metadata_file cv_split_file trained_model output_folder
```
e.g.
```shell
python analyse_results.py "/media/greifflab/t1d/kmer_counts_l4.pkl" "/media/greifflab/t1d/metadata.csv" "/media/greifflab/t1d/cv_splits_dataset1.pkl" "/media/greifflab/t1d/kmer_logreg_model.pkl" "/media/greifflab/t1d/kmer_logreg_model/analysis"
```

