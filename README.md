```bash


GVT/gvt/scripts/run_all_eval.sh \
  --data-root /home/ma-user/VisualTokenizer/data/gvt/arrow \
  --vicuna-path /home/ma-user/VisualTokenizer/checkpoints/vicuna-7b-v1.1 \
  --load-path /home/ma-user/VisualTokenizer/checkpoints/gvt.pth \
  --batch-size 16 \
  --output-dir /home/ma-user/VisualTokenizer/outputs/gvt_eval \
  --tasks task_eval_coco_count,task_eval_coco_multiclass,task_eval_coco_caption,task_eval_vqav2 \
  --skip-missing-data
```

