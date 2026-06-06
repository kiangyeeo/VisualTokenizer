# Project Plan: Benchmark for Visual Tokenizers

# Goal

This project aims to develop a benchmark for evaluating visual tokenizers used in text\-output multimodal large language models\. The benchmark shoulS be better aligned with downstream MLLM performance than reconstruction\-based evaluation, while being much cheaper than running full MLLM finetuning for every tokenizer\.

- The benchmark should support both continuous and discrete visual tokenizers\.

- The benchmark should consider different ways visual tokenizers are used in MLLMs:

    - models that use visual tokens only for encoding image inputs;

    - models that use visual tokenizers during both encoding and decoding;

    - models capable of generating interleaved image\-text outputs\.

- As a first step, we can focus on:

    - continuous visual tokenizers;

    - image\-conditioned MLLMs that generate text only\.

- In later stages, we can generalize the framework to:

    - discrete visual tokenizers;

    - MLLMs with image\-generation or interleaved image\-text generation capabilities\.

# Actionable Items

## 1\. Set Up a Pipeline for Collecting \&\#34;Ground\-Truth\&\#34; Data Points 

The ground\-truth data points are the results of actual MLLM finetuning runs\. To evaluate whether a proposed tokenizer benchmark is meaningful, we need to compare its predicted tokenizer ranking against the ranking obtained from downstream MLLM performance\.

**Baseline Tokenizers\.** To collect enough data points, we need tokenizers with varying levels of quality\. We can start with publicly available visual tokenizers to cover a diverse set of existing designs\. In addition, we can train a visual tokenizer from scratch and use its intermediate checkpoints as lower\-quality variants\. This would give us a more controlled set of tokenizers with different levels of training progress and representation quality\. For training new tokenizers, we can start from the OpenCLIP training framework and use their smaller model config: https://github\.com/mlfoundations/open\_clip

**MLLM Finetuning\.** For the MLLM finetuning pipeline, we can start from the LLaVA training framework:

https://github\.com/haotian\-liu/LLaVA/tree/main

## 2\. Set Up a Pipeline for Collecting Baseline Data Points

We will use existing tokenizer benchmarks as baselines\. Candidate baselines include:

For each baseline, we need to rerun its evaluation pipeline using our selected baseline tokenizers\. This will allow us to directly compare how well existing evaluation methods predict actual MLLM finetuning performance\.

## 3\. Develop Early Metrics for Measuring Tokenizer Quality

We need to design efficient metrics that estimate tokenizer quality without running the full MLLM finetuning pipeline\.

## 4\. Compare Against Prior Evaluation Methods

We will evaluate whether the proposed benchmark predicts actual MLLM finetuning results better than existing tokenizer evaluation signals\.

# Success Criteria

The main success criterion is a higher correlation with downstream MLLM performance\. We can measure this using ranking\-based correlation metrics such as Spearman correlation, Kendall’s tau, or pairwise ranking accuracy\.





# Useful Resources

- CLIP checkpoints: https://github\.com/laion\-ai/scaling\-laws\-openclip

- 

```
conda activate wky
cd VisualTokenizer
GVT/gvt/scripts/run_all_eval.sh \ 
--data-root /home/ma-user/VisualTokenizer/data/gvt/arrow \ 
--vicuna-path /path/to/vicuna-7b-v1.1 \ 
--load-path /path/to/gvt.pth \ 
--batch-size 32 \ 
--output-dir /home/ma-user/VisualTokenizer/outputs/gvt_eval \ 
--tasks task_eval_coco_count,task_eval_coco_multiclass,task_eval_coco_caption,task_eval_vqav2 \ 
--skip-missing-data

GVT/gvt/scripts/run_all_eval.sh \
  --data-root /home/ma-user/VisualTokenizer/data/gvt/arrow \
  --vicuna-path /home/ma-user/VisualTokenizer/checkpoints/vicuna-7b-v1.1 \
  --load-path /home/ma-user/VisualTokenizer/checkpoints/gvt.pth \
  --batch-size 8 \
  --output-dir /home/ma-user/VisualTokenizer/outputs/gvt_eval \
  --tasks task_eval_vqav2 \
  --skip-missing-data

```

