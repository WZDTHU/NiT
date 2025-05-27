from .ennlshp import ENNLSHP
from .lpfhp import LPFHP
from .nnlshp import NNLSHP
from .spfhp import SPFHP

import json
import torch
import numpy as np
from tqdm import tqdm

def get_strategy(algorithm, max_seq_len, max_seq_per_pack, dataset_seq_lens):
    def generate_histogram(dataset_seq_lens):
        histogram = np.zeros(max_seq_len, dtype=np.int64)
        seq_lens, counts = np.unique(np.array(dataset_seq_lens), return_counts=True)
        histogram[seq_lens - 1] = counts
        return histogram
    histogram = generate_histogram(dataset_seq_lens)
    if algorithm == "SPFHP":
        strategy = SPFHP(histogram, max_seq_len, max_seq_per_pack)
    elif algorithm == "LPFHP":
        strategy = LPFHP(histogram, max_seq_len, max_seq_per_pack)
    elif algorithm == 'ENNLSHP':
        strategy = ENNLSHP(histogram, max_seq_len, max_seq_per_pack)
    elif algorithm == 'NNLSHP':
        strategy = NNLSHP(histogram, max_seq_len, max_seq_per_pack)
    else:
        raise NotImplementedError("Algorithm type unsupported. Pass one of: LPFHP, SPFHP")
    return strategy

def pack_dataset(algorithm, max_seq_len, max_seq_per_pack, dataset_seq_lens, dataset_seq_idxs):
    dataset_seqs = torch.stack([torch.tensor(dataset_seq_lens), torch.tensor(dataset_seq_idxs)])
    strategy_set, strategy_repeat_count = get_strategy(
        algorithm, max_seq_len, max_seq_per_pack, dataset_seq_lens
    )
    
    packed_indices = []
    run_iters = sum(strategy_repeat_count)
    progress_bar = tqdm(range(run_iters))
    for i in range(len(strategy_repeat_count)):
        strategy = strategy_set[i]
        for _ in range(strategy_repeat_count[i]):
            progress_bar.update(1)
            ref_inds = []
            for x in strategy:
                ref_ind = torch.argwhere(dataset_seqs[0] == x)[-1]
                dataset_seqs[0, ref_ind] = -1
                ref_inds.append(ref_ind)
            inds = dataset_seqs[1, ref_inds].ravel()
            packed_indices.append(inds.tolist())
    return packed_indices

