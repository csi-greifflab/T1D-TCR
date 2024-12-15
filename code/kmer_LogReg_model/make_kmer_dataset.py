# -*- coding: utf-8 -*-
"""
Author -- Michael Widrich
"""

import os
import glob
import numpy as np
import pandas as pd
import pickle as pkl
import h5py
import itertools
# import multiprocessing
from typing import Tuple, Union
# import tqdm


def make_kmer_dataset(sequence_files_dir: str, output_file: Union[None, str], aa_alphabet: str = 'ACDEFGHIKLMNPQRSTVWY', kmer_size: int = 4):
    if output_file is not None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Get all .tsv sample files
    print(f"Searching for .tsv files in {sequence_files_dir}...")
    sequence_files = glob.glob(os.path.join(sequence_files_dir, '**', '*.tsv'), recursive=True)
    print(f"  found {len(sequence_files)} files")
    
    kmer_alphabet = [''.join(kmer) for kmer in itertools.product(aa_alphabet, repeat=kmer_size)]
    
    def get_kmer_counts_from_file(sequence_file_path: str) -> Tuple[str, np.ndarray, np.ndarray]:
        """Count kmers in file and return filename and counts"""
        sfc = pd.read_csv(sequence_file_path, sep = "\t")
        
        kmers_list = [
            sequence_str[aa_i:aa_i + kmer_size]
            for sequence_str in sfc['amino_acid'].apply(lambda x:x[4:-4])
            for aa_i in range(len(sequence_str) - kmer_size)
        ]
        sample_kmers, sample_kmers_counts = np.unique(kmers_list, return_counts=True)
        return os.path.basename(sequence_file_path), sample_kmers, sample_kmers_counts
    
    
    kmer_counts = pd.DataFrame(
            index=[os.path.basename(s) for s in sequence_files],
            columns=kmer_alphabet,
            data=np.zeros(shape=(len(sequence_files), len(kmer_alphabet)), dtype=np.int64)
    )
    
    print(f"Extracting kmer counts of length {kmer_size} from {len(sequence_files)} files...")
    # with multiprocessing.Pool(processes=8) as pool:
    #     for sequence_file, sample_kmers, sample_kmers_counts in tqdm.tqdm(
    #             pool.imap(get_kmer_counts_from_file, sequence_files),
    #             total=len(sequence_files)
    #     ):
    #         kmer_counts.loc[sequence_file, sample_kmers] = sample_kmers_counts
    for i in sequence_files:
        sequence_file, sample_kmers, sample_kmers_counts = get_kmer_counts_from_file(i)
        kmer_counts.loc[sequence_file, sample_kmers] = sample_kmers_counts
    print(f"  stored counts in dataframe of shape {kmer_counts.shape}")
    
    if output_file is None:
        print("Not writing to output file, since no output file was specified")
    else:
        print(f"Writing kmer counts to {output_file}...")
        # kmer_counts.iloc[:,6000:13000].T.to_csv('output.csv')
        # print(kmer_counts.head)
        with open(output_file, 'wb') as fh:
            pkl.dump(kmer_counts, fh)
    print("Done!")
    return kmer_counts


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Creating kmer features from fasta files')
    parser.add_argument('sequence_files_dir', type=str, help='path to fasta files with sequence information')
    parser.add_argument('output_file', help='path to store resulting pickle file at')

    args = parser.parse_args()
    # sequence_files_dir = "/media/michael/Data/t1d/t1d_workspace_20221124/t1d_20210113/restricteddata/greifflab/t1d_20210113/full/sequences_purged"
    # output_file = "/media/michael/Data/t1d/t1d_workspace_20221124/t1d_20210113/restricteddata/greifflab/t1d_20210113/full/kmer_counts_l4.pkl"

    make_kmer_dataset(args.sequence_files_dir, args.output_file)
