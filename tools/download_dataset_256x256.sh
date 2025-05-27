target_dir="datasets/imagenet1k/dc-ae-f32c32-sana-1.1-diffusers-256x256"
mkdir -p $target_dir
base_url="https://huggingface.co/datasets/GoodEnough/NiT-Preprocessed-ImageNet1K/resolve/main/dc-ae-f32c32-sana-1.1-diffusers-256x256"
files=(
    "n01440764_n02097298.zip"
    "n02097474_n02667093.zip"
    "n02669723_n03530642.zip"
    "n03532672_n04239074.zip"
    "n04243546_n15075141.zip"
)
for file in "${files[@]}"; do
    echo "download $file ..."
    wget -c "$base_url/$file" -O "$target_dir/$file"
    echo "download $file finished"
    echo "start unzip $file ..."
    unzip "$target_dir/$file" -d "$target_dir"
    echo "unzip $file finished"
    rm "$target_dir/$file"
    echo
done
echo "Successfully download all the sampler-meta"

