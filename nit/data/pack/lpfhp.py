# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# modified from https://github.com/graphcore/examples/blob/v3.2.0/tutorials/blogs_code/packedBERT/lpfhp.py
"""Longest-pack-first histogram-packing."""
from collections import defaultdict
import numpy as np
import time


def add_pack(pack, count, tmp, final, limit, offset, max_sequence_length=512):
    """Filter out packs that reached maximum length or number of components."""
    # sanity checks
    assert max_sequence_length - sum(pack) == offset, "Incorrect offset."
    assert offset >= 0, "Too small offset."
    assert offset < max_sequence_length, "Too large offset."
    if len(pack) == limit or offset == 0:
        final[offset].append((count, pack))
    else:
        tmp[offset].append((count, pack))


def LPFHP(histogram, max_sequence_length, max_sequences_per_pack, distribute=True):
    """Longest-pack-first histogram-packing."""
    start = time.time()
    reversed_histogram = np.flip(histogram)
    # Initialize main strategy data dictionary.
    # The key indicates how many tokens are left for full length.
    # The value is a list of tuples, consisting of counts and respective packs.
    # A pack is a (sorted) list of sequence length values that get concatenated.
    tmp_strategies_per_length = defaultdict(list)
    strategies_per_length = defaultdict(list)
    if max_sequences_per_pack == "max":
        max_sequences_per_pack = max_sequence_length
    # Index i indicates here, how much space is left, due to reversed histogram
    for i in range(max_sequence_length):
        n_sequences_to_bin = reversed_histogram[i]
        length_to_bin = max_sequence_length - i
        offset = 0  # smallest possible offset for perfect fit
        while n_sequences_to_bin > 0:
            if (length_to_bin + offset) in tmp_strategies_per_length:
                # extract worst pack that will get modified
                n_sequences_to_pack, pack = tmp_strategies_per_length[length_to_bin + offset].pop()
                # calculate how often the current sequence maximally fits in
                repeat = min(1 + offset // length_to_bin, max_sequences_per_pack - len(pack))
                # correct dependent on count
                while n_sequences_to_bin // repeat == 0:
                    repeat -= 1
                if not distribute:
                    repeat = 1
                new_pack = pack + [length_to_bin] * repeat
                count = min(n_sequences_to_pack, n_sequences_to_bin // repeat)
                if n_sequences_to_pack > count:
                    # old pack gets reduced
                    n_sequences_to_pack -= count
                    tmp_strategies_per_length[length_to_bin + offset].append((n_sequences_to_pack, pack))
                    n_sequences_to_bin -= count * repeat
                else:
                    n_sequences_to_bin -= n_sequences_to_pack * repeat
                add_pack(
                    new_pack,
                    count,
                    tmp_strategies_per_length,
                    strategies_per_length,
                    max_sequences_per_pack,
                    offset - (repeat - 1) * length_to_bin,
                    max_sequence_length,
                )
                # clean up to speed up main key search
                if not tmp_strategies_per_length[length_to_bin + offset]:
                    tmp_strategies_per_length.pop(length_to_bin + offset)
                # reset offset in case best fit changed
                offset = 0
            else:
                offset += 1
            # Does not fit anywhere. Create new pack.
            if offset >= max_sequence_length - length_to_bin + 1:
                # similar repetition but no dependence on pack.
                repeat = min(max_sequence_length // length_to_bin, max_sequences_per_pack)
                while n_sequences_to_bin // repeat == 0:
                    repeat -= 1
                if not distribute:
                    repeat = 1
                add_pack(
                    [length_to_bin] * repeat,
                    n_sequences_to_bin // repeat,
                    tmp_strategies_per_length,
                    strategies_per_length,
                    max_sequences_per_pack,
                    max_sequence_length - length_to_bin * repeat,
                    max_sequence_length,
                )
                n_sequences_to_bin -= n_sequences_to_bin // repeat * repeat
    # merge all strategies
    for key in tmp_strategies_per_length:
        strategies_per_length[key].extend(tmp_strategies_per_length[key])
    # flatten strategies dictionary
    strategy_set = []
    strategy_repeat_count = []
    for key in strategies_per_length:
        for count, pack in strategies_per_length[key]:
            pack.reverse()
            strategy_set.append(pack)
            strategy_repeat_count.append(count)

    # Summarize efficiency of solution
    duration = time.time() - start
    sequence_lengths = np.arange(1, max_sequence_length + 1)
    strategy_repeat_count = np.array(strategy_repeat_count)
    n_strategies = len(strategy_set)
    old_number_of_samples = histogram.sum()
    new_number_of_samples = strategy_repeat_count.sum()
    sequences = sum([count * len(pack) for count, pack in zip(strategy_repeat_count, strategy_set)])
    total_tokens = max_sequence_length * new_number_of_samples
    empty_tokens = sum(
        [count * (max_sequence_length - sum(pack)) for count, pack in zip(strategy_repeat_count, strategy_set)]
    )
    efficiency = 100 - empty_tokens / total_tokens * 100
    speedup_upper_bound = 1.0 / (
        1 - (histogram * (1 - sequence_lengths / max_sequence_length)).sum() / old_number_of_samples
    )

    print(
        f"Packing efficiency (fraction of real tokens): {efficiency:3.4f}\n",
        f"Speed-up theoretical limit: {speedup_upper_bound:3.4f}\n",
        f"Achieved speed-up over un-packed dataset: {old_number_of_samples/new_number_of_samples:3.5f}",
        f"Runtime: Packed {old_number_of_samples} sequences in {duration:3.3f} seconds.",
    )

    return strategy_set, strategy_repeat_count
