target_dir="datasets/imagenet1k/data_meta"
mkdir -p $target_dir
base_url="https://huggingface.co/datasets/GoodEnough/NiT-Preprocessed-ImageNet1K/resolve/main/data_meta"
files=(
    "dc-ae-f32c32-sana-1.1-diffusers_256x256_meta.jsonl"
    "dc-ae-f32c32-sana-1.1-diffusers_512x512_meta.jsonl"
    "dc-ae-f32c32-sana-1.1-diffusers_nr_meta.jsonl"
    "dc-ae-f32c32-sana-1.1-diffusers_merge_meta.jsonl"
)
for file in "${files[@]}"; do
    echo "download $file ..."
    wget -c "$base_url/$file" -O "$target_dir/$file"
    echo "download $file finished"
    echo
done
echo "Successfully download all the data-meta"

