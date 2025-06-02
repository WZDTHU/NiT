target_dir="datasets/imagenet1k/dc-ae-f32c32-sana-1.1-diffusers-512x512"
mkdir -p $target_dir
base_url="https://huggingface.co/datasets/GoodEnough/NiT-Preprocessed-ImageNet1K/resolve/main/dc-ae-f32c32-sana-1.1-diffusers-512x512"
files=(
    "n01440764_n01697457.zip"
    "n01698640_n01855672.zip"
    "n01860187_n02074367.zip"
    "n02077923_n02097298.zip"
    "n02097474_n02110063.zip"
    "n02110185_n02138441.zip"
    "n02165105_n02415577.zip"
    "n02417914_n02667093.zip"
    "n02669723_n02859443.zip"
    "n02860847_n03041632.zip"
    "n03042490_n03291819.zip"
    "n03297495_n03530642.zip"
    "n03532672_n03743016.zip"
    "n03759954_n03884397.zip"
    "n03887697_n04033901.zip"
    "n04033995_n04239074.zip"
    "n04243546_n04398044.zip"
    "n04399382_n04560804.zip"
    "n04562935_n07745940.zip"
    "n07747607_n15075141.zip"
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

