# Quick Start: CLIP Bench
```
cd /cache/ma-user/VisualTokenizer

conda create -n clipbench python=3.10 -y
conda activate clipbench

cd CLIP_benchmark

python -m pip install -U pip setuptools wheel -i https://pypi.org/simple 
python -m pip install -r requirements.txt -i https://pypi.org/simple 
python -m pip install -e . --no-build-isolation -i https://pypi.org/simple 

mkdir -p /cache/ma-user/VisualTokenizer/data/clip_benchmark
mkdir -p /cache/ma-user/VisualTokenizer/outputs/clip_benchmark
mkdir -p /cache/ma-user/VisualTokenizer/checkpoints/open_clip

clip_benchmark eval \
  --dataset=cifar10 \
  --task=zeroshot_classification \
  --model=ViT-B-32-quickgelu \
  --pretrained=laion400m_e32 \
  --model_cache_dir=/cache/ma-user/VisualTokenizer/checkpoints/open_clip \
  --dataset_root=/cache/ma-user/VisualTokenizer/data/clip_benchmark \
  --output=/cache/ma-user/VisualTokenizer/outputs/clip_benchmark/cifar10_openclip.json \
  --batch_size=256 \
  --num_workers=8
  
  clip_benchmark eval \
  --dataset=mscoco_captions \
  --task=zeroshot_retrieval \
  --model=ViT-B-32-quickgelu \
  --pretrained=laion400m_e32 \
  --model_cache_dir=/cache/ma-user/VisualTokenizer/checkpoints/open_clip \
  --dataset_root=/cache/ma-user/VisualTokenizer/data/clip_benchmark/mscoco_captions \
  --output=/cache/ma-user/VisualTokenizer/outputs/clip_benchmark/mscoco_openclip_retrieval.json \
  --batch_size=256 \
  --num_workers=8 \
  --recall_k 1 5 10
  
  clip_benchmark build \
  /cache/ma-user/VisualTokenizer/outputs/clip_benchmark/*.json \
  --output /cache/ma-user/VisualTokenizer/outputs/clip_benchmark/benchmark.csv
  
  
  
  python CLIP_benchmark/tools/train_gvt_projection.py \
  --dataset cifar10 \
  --dataset-root data/clip_benchmark \
  --split train \
  --model-cache-dir checkpoints/open_clip \
  --gvt-checkpoint checkpoints/gvt.pth \
  --output checkpoints/gvt_clip_attention_projection.pt \
  --projector-type attention \
  --num-query-tokens 4 \
  --attn-heads 8 \
  --epochs 5 \
  --batch-size 64 \
  --num-workers 8
  
  
  
  
  
  
  
  
 clip_benchmark build \
  /cache/ma-user/VisualTokenizer/outputs/clip_benchmark/*.json \
  --output /cache/ma-user/VisualTokenizer/outputs/clip_benchmark/benchmark.csv
```

