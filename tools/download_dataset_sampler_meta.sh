target_dir="datasets/imagenet1k/sampler_meta"
mkdir -p $target_dir
base_url="https://huggingface.co/datasets/GoodEnough/NiT-Preprocessed-ImageNet1K/resolve/main/sampler_meta"
files=(
    "dc-ae-f32c32-sana-1.1-diffusers_merge_LPFHP_16384.json"
    "dc-ae-f32c32-sana-1.1-diffusers_merge_LPFHP_32768.json"
    "dc-ae-f32c32-sana-1.1-diffusers_merge_LPFHP_65536.json"
)
for file in "${files[@]}"; do
    echo "download $file ..."
    wget -c "$base_url/$file" -O "$target_dir/$file"
    echo "download $file finished"
    echo
done
echo "Successfully download all the sampler-meta"

