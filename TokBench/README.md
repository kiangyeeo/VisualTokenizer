# Quick Start: TokBench
## Run it (single A100, image pipeline)

**1. Create the isolated env + install deps**

```bash
conda create -n TokBench python=3.10 -y
conda activate TokBench

conda install -c pytorch -c nvidia pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=11.8 -y
conda install "mkl<2025" -y

cd /cache/ma-user/VisualTokenizer/TokBench
pip install -i https://pypi.org/simple -r requirements.txt
pip install -e . --no-deps

export USE_TORCH=1
export LD_LIBRARY_PATH="$(python -c 'import os,glob,nvidia; print(":".join(glob.glob(os.path.join(os.path.dirname(nvidia.__file__),"*","lib"))))'):$LD_LIBRARY_PATH"
```

**2. Download the benchmark (~2.7 GB, proxy already set on 7890)**

```bash
bash download_data.sh
# -> tokbench_data/annotations/*.json  and  tokbench_data/images/{text_data,face_data}
```

**3. Reconstruct (resize baseline) → evaluate → aggregate**



```bash
cd /home/ma-user/VisualTokenizer/TokBench
bash tokenzier_vae_scripts/image_scripts/resize.sh   # step 1: reconstruction (CPU, ~1-2 min)
bash image_eval.sh                                   # steps 2-4: OCR + face + aggregate


cd /cache/ma-user/VisualTokenizer/TokBench/tokenzier_vae_scripts/image_scripts
PADDING_SIZES="256 512 1024" bash resize.sh

cd /cache/ma-user/VisualTokenizer/TokBench
RES=256 bash image_eval.sh
RES=512 bash image_eval.sh
RES=1024 bash image_eval.sh
```

The final `compute_all_metrics.py` prints two PrettyTables: **T-ACC / T-NED** (text, bucketed by text size ratio) and **F-Sim** (face). Models auto-download on first run: docTR `parseq` from `doctr-static.mindee.com` and InsightFace `antelopev2` to `~/.insightface/models/`.

## Pipeline logic in one paragraph

`resize_rec.py` simulates a tokenizer by smart-padding each image to 256², resizing back to original — the lossy round-trip a real VAE/tokenizer would do — and writes a mirrored tree `image_reconstruction_results/resize/{text_data,face_data}/<dataset>_256/`. Then [eval_text.py](vscode-webview://147e1ufhpr00k8peb4jmuis3kfgc4bc61lqevuu5mcr895mphrr4/TokBench/eval_text.py) crops every annotated text box from the *reconstructed* image and runs `parseq` OCR, scoring exact-match accuracy and 1-NED per instance, tagged with the text's size ratio. [eval_face.py](vscode-webview://147e1ufhpr00k8peb4jmuis3kfgc4bc61lqevuu5mcr895mphrr4/TokBench/eval_face.py) extracts `antelopev2` embeddings from original vs. reconstructed faces (using the GT keypoints) and computes cosine similarity. [compute_all_metrics.py](vscode-webview://147e1ufhpr00k8peb4jmuis3kfgc4bc61lqevuu5mcr895mphrr4/TokBench/compute_all_metrics.py) pools all per-instance results and bins them into size-ratio buckets (the smallest text/faces are where tokenizers fail) to produce the leaderboard tables.

## Two gotchas to expect

1. **antelopev2 nesting** — InsightFace sometimes extracts to `~/.insightface/models/antelopev2/antelopev2/*.onnx`. If `eval_face.py` errors finding models, flatten it: `mv ~/.insightface/models/antelopev2/antelopev2/* ~/.insightface/models/antelopev2/`.
2. **onnxruntime-gpu/cuDNN** — if `CUDAExecutionProvider` fails to init, the script's provider list falls back to CPU automatically (the face set is small, so CPU is fine).







视频模块完整启动命令（在已建好的 `TokBench` 环境里，从仓库根目录执行）：



```bash
cd /home/ma-user/VisualTokenizer/TokBench
conda activate TokBench
export USE_TORCH=1                       # 强制 docTR 用 torch 后端

# 1. 下载视频数据（video_annotations/ + videos.zip ~1.35GB，不影响已下的图像数据）
WITH_VIDEO=1 bash download_data.sh

# 2. 视频重建（resize 基线，纯 CPU）
bash tokenzier_vae_scripts/video_scripts/resize.sh

# 3. 评测 + 聚合（文本 T-ACC/T-NED + 人脸 F-Sim，输出到 video_outputs/）
bash video_eval.sh



cd /cache/ma-user/VisualTokenizer/TokBench/tokenzier_vae_scripts/video_scripts
SHORT_SIZES="256 480" bash resize.sh

cd /cache/ma-user/VisualTokenizer/TokBench
RES=256 bash video_eval.sh
RES=480 bash video_eval.sh
```

默认跑 **256** 档。要跑 **480**（视频另一档）：



```bash
# 重建 480（或一次出两档：SHORT_SIZES="256 480"）
SHORT_SIZES=480 bash tokenzier_vae_scripts/video_scripts/resize.sh
# 评测 480（聚合按单一 setting 出表，256/480 各跑一次）
RES=480 bash video_eval.sh
```

关键路径（都从 `REPO_ROOT=/home/ma-user/VisualTokenizer/TokBench` 推导，可用同名环境变量覆盖）：

| 变量                   | 默认值                          | 含义                                     |
| ---------------------- | ------------------------------- | ---------------------------------------- |
| `DATA_ROOT`            | `tokbench_data/`                | `videos/` + `video_annotations/`         |
| `RECON_ROOT`           | `video_reconstruction_results/` | 重建输出（已和 resize.sh 统一）          |
| `OUT_DIR`              | `video_outputs/`                | 评测结果，与图像的 `image_outputs/` 隔离 |
| `CUDA_VISIBLE_DEVICES` | `0`                             | 单卡                                     |

跟图像那套完全独立，互不影响。需要的话我可以把图像+视频两套命令合并成一个 `run_all.sh` 一键脚本。

