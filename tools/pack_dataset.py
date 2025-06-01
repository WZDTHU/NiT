import json
from nit.data.pack import pack_dataset
import argparse



def create_pack(data_meta, max_seq_len, algorithm, split):  
    max_seq_per_pack = max_seq_len
    with open(data_meta, 'r') as fp:
        ori_dataset = [json.loads(line) for i, line in enumerate(fp)]
    dataset_seq_lens = []
    dataset_seq_idxs = []
    for idx, data in enumerate(ori_dataset):
        seq_len = int(data['latent_h']*data['latent_w'])   # patch_size=1
        dataset_seq_lens.append(seq_len)
        dataset_seq_idxs.append(idx)
    total_length = len(ori_dataset)

    run_length = int(total_length / split)
    all_packed_indices = []
    for i in range(split):
        seq_lens = dataset_seq_lens[i*run_length: (i+1)*run_length]
        seq_idxs = dataset_seq_idxs[i*run_length: (i+1)*run_length]
        packed_indices = pack_dataset(
            algorithm, max_seq_len, max_seq_per_pack, seq_lens, seq_idxs
        ) 
        all_packed_indices.extend(packed_indices)
    
    sampler_json_name = data_meta.split('/')[-1].replace('_meta.jsonl', '')
    sampler_json_name = f"{sampler_json_name}_{algorithm}_{max_seq_len}.json"
    with open(f'datasets/imagenet1k/sampler_meta/{sampler_json_name}', 'w') as fp:
        json.dump(all_packed_indices, fp, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # seed
    parser.add_argument("--data-meta", type=str, default='datasets/imagenet1k/data_meta/dc-ae-f32c32-sana-1.1-diffusers_merge_meta.jsonl')
    parser.add_argument("--max-seq-len", type=int, default=16384)
    parser.add_argument("--algorithm", type=str, default='LPFHP')
    parser.add_argument("--split", type=int, default=1)
    args = parser.parse_args()
    create_pack(
        data_meta=args.data_meta, max_seq_len=args.max_seq_len,
        algorithm=args.algorithm, split=args.split
    )
