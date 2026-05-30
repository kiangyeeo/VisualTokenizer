baseline:

for discrete tokenizers: 
https://github.com/SilentView/GigaTok.git: They propose a discrete VQ-style encoder-decoder tokenizer that maps images into codebook indices and reconstructs images from these indices, with evaluation mainly focused on reconstruction quality and downstream autoregressive image generation.

**VTBench**（2025）:Encoder-decoder / 生成

**重建指标**（rFID / PSNR / gFID）:**Encoder-decoder**



for continuous tokenizers:

**GVT-Bench**（What Makes for Good Visual Tokenizers, 2023）: encoder-only

**Law of Vision Representation in MLLMs / AC policy**（2024）: encoder-only

 **CLIP 式表征代理**（zero-shot 分类 / linear probing / 图文检索）:OpenCLIP 评测体系





1. **metric baseline**：PSNR、SSIM、LPIPS、rFID 这种“算一个分数”的指标。
2. **benchmark / pipeline baseline**：TokBench、VTBench、GVTBench 这种“有数据集 + 评测脚本 + 论文/仓库”的完整项目。

你们项目计划的第二部分说的是“use existing tokenizer benchmarks as baselines”，并且要比较这些 baseline 排名和真实 MLLM finetuning 排名的一致性；所以两类都可以收集。

下面是我建议你后续复现优先看的 baseline。

------

## 第一组： tokenizer-level baseline

| Name                                                         | Level                       | Category                                                     | 支持 continuous / discrete？                                 | Need decoder? | 适合哪类 MLLM 使用方式                                 | Code                                                         | Paper / Source                                               | 我们项目中的角色                                             |
| ------------------------------------------------------------ | --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------- | ------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **PSNR / SSIM / MS-SSIM**                                    | metric                      | 传统重构质量                                                 | **Both**，只要能输出重构图                                   | Yes           | encoding-decoding / generation tokenizer               | PIQ 提供 PSNR、SSIM 等图像质量指标实现 ([GitHub](https://github.com/photosynthesis-team/piq?utm_source=chatgpt.com)) | 通用 IQA 指标                                                | 最基础重构 baseline                                          |
| **LPIPS**                                                    | metric                      | 感知重构质量                                                 | **Both**，只要能输出重构图                                   | Yes           | encoding-decoding / generation tokenizer               | 官方 LPIPS repo ([GitHub](https://github.com/richzhang/perceptualsimilarity?utm_source=chatgpt.com)) | LPIPS project page 说明其基于人类感知相似度判断 ([richzhang.github.io](https://richzhang.github.io/PerceptualSimilarity/?utm_source=chatgpt.com)) | 比 PSNR 更接近人眼，但仍是重构指标                           |
| **FID / rFID / KID**                                         | metric                      | 分布级重构 / 生成质量                                        | **Both**，只要有一批重构/生成图                              | Yes           | generation / reconstruction tokenizer                  | clean-fid、torch-fidelity 都可用 ([GitHub](https://github.com/GaParmar/clean-fid?utm_source=chatgpt.com)) | clean-fid 说明 FID 实现差异会影响可比性 ([CMU School of Computer Science](https://www.cs.cmu.edu/~clean-fid/?utm_source=chatgpt.com)) | 生成模型领域常用 baseline                                    |
| **TokBench**                                                 | benchmark project           | 细粒度重构：text / face                                      | **Both**。面向 visual tokenizers 和 VAEs；离散 tokenizer、连续 VAE 只要能 decode 都可评 | Yes           | encoding-decoding / generation tokenizer               | 官方 repo ([GitHub](https://github.com/wjf5203/TokBench))    | arXiv / paper 页面 ([arXiv](https://arxiv.org/html/2505.18142v2?utm_source=chatgpt.com)) | 很重要。比普通 PSNR/LPIPS 更关注 OCR、人脸等 MLLM 也敏感的信息 |
| **VTBench**                                                  | benchmark project           | AR image generation tokenizer 评测：reconstruction / detail / text preservation | **主要面向 discrete VT**，但也把 continuous VAE 作为对照；有 decode 即可比较 | Yes           | visual tokenizer during decoding / AR image generation | 官方 repo ([GitHub](https://github.com/huawei-lin/VTBench))  | arXiv 摘要说明其评估 image reconstruction、detail preservation、text preservation ([arXiv](https://arxiv.org/abs/2505.13439?utm_source=chatgpt.com)) | 后续做 discrete / image generation tokenizer 时很关键        |
| **GVTBench / What Makes for Good Visual Tokenizers for LLMs?** | benchmark / empirical suite | MLLM understanding：semantic understanding + fine-grained perception | **主要是 continuous feature tokenizer / vision encoder**，如 CLIP、MAE、DINO、DeiT | No            | encoding image inputs only                             | 官方 repo ([GitHub](https://github.com/TencentARC/GVT))      | arXiv 论文研究不同视觉预训练方式对 MLLM visual tokenizer 的影响 ([arXiv](https://arxiv.org/abs/2305.12223?utm_source=chatgpt.com)) | 和你们第一阶段最贴近：image-conditioned MLLM text-only       |
| **swiss-ai benchmark-image-tokenzier**                       | engineering pipeline        | 多 tokenizer 重构 sweep：PSNR / SSIM / LPIPS                 | **Both in principle**，但要写 tokenizer wrapper；当前 repo 提供若干 tokenizer 的 encode-decode | Yes           | reconstruction / tokenization pipeline                 | repo 中说明可对 14 个 tokenizer encode-decode 并计算 PSNR/SSIM/LPIPS ([GitHub](https://github.com/swiss-ai/benchmark-image-tokenzier)) | 无正式论文                                                   | 很适合作为工程参考，帮你快速搭建统一 wrapper                 |
| **CLIP / DINO feature similarity**                           | custom metric baseline      | 语义保留，而不是像素重构                                     | **Continuous 直接支持；discrete 需要 embedding 或 decode 后再算** | No / optional | encoding-only MLLM 最相关                              | 可用 OpenCLIP / DINOv2 等现成模型                            | Awesome-Visual-Tokenizers 将 visual tokens 定义为 continuous、discrete 或 hybrid 单元 ([GitHub](https://github.com/lavinal712/Awesome-Visual-Tokenizers/blob/main/README.md)) | 建议你们自己实现，作为 semantic-preservation baseline        |

------

## 第二组：更像“下游 ground truth / future extension”，不是 cheap tokenizer baseline

这些也要记录，但不要和 TokBench、VTBench 混为一类。它们评的是 **完整 MLLM / U-MLLM**，不是单独评 tokenizer。

| Name                                                         | Level                            | Category                                        | 支持 continuous / discrete？           | Need decoder? | 适合哪类 MLLM 使用方式                | Code                                                         | Paper / Source                                               | 我们项目中的角色                                             |
| ------------------------------------------------------------ | -------------------------------- | ----------------------------------------------- | -------------------------------------- | ------------- | ------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **VQAv2 / GQA / TextVQA / OCRBench / MME / MM-Vet / MMBench** | downstream benchmark             | MLLM 图像理解                                   | tokenizer-agnostic；接入 MLLM 后都能评 | No            | encoding image inputs only            | 各 benchmark 官方实现                                        | OCRBench repo 说明其评估 MLLM OCR 能力，含 text recognition、document VQA 等 ([GitHub](https://github.com/yuliang-liu/multimodalocr?utm_source=chatgpt.com)) | 用来产生真实下游 ranking，不是 cheap baseline                |
| **UniEval / UniBench**                                       | benchmark framework              | unified multimodal understanding + generation   | tokenizer-agnostic；评完整统一模型     | Usually yes   | understanding + generation            | 官方 repo ([GitHub](https://github.com/xmed-lab/UniEval))    | 论文称 UniEval 面向 unified multimodal models，UniBench 支持 unified 和 visual generation models ([arXiv](https://arxiv.org/abs/2505.10483?utm_source=chatgpt.com)) | 后续做 interleaved / unified generation 时可用               |
| **MME-Unify**                                                | benchmark framework              | unified MLLM 评测，含理解、生成、编辑、图文交错 | tokenizer-agnostic；评完整模型         | Usually yes   | interleaved image-text / unified MLLM | 官方 repo ([GitHub](https://github.com/MME-Benchmarks/MME-Unify)) | repo 说明其含传统任务、统一任务和多模态 reasoning/generation 评测 ([GitHub](https://github.com/MME-Benchmarks/MME-Unify)) | 后续阶段用，不建议第一阶段复现                               |
| **TokenFlow eval suite**                                     | tokenizer model + eval reference | unified tokenizer：understanding + generation   | **Discrete**                           | Yes           | encoding + decoding / unified MLLM    | 官方 repo ([GitHub](https://github.com/ByteFlow-AI/TokenFlow?utm_source=chatgpt.com)) | CVPR 2025 paper 说明它用 dual-codebook 桥接 understanding 和 generation ([CVF开放获取](https://openaccess.thecvf.com/content/CVPR2025/html/Qu_TokenFlow_Unified_Image_Tokenizer_for_Multimodal_Understanding_and_Generation_CVPR_2025_paper.html?utm_source=chatgpt.com)) | 不是 benchmark baseline，但可作为 future tokenizer / evaluation reference |
| **UniTok eval suite**                                        | tokenizer model + eval reference | unified tokenizer：generation + understanding   | **Discrete**                           | Yes           | encoding + decoding / unified MLLM    | 官方 repo ([GitHub](https://github.com/FoundationVision/UniTok)) | project page 称 UniTok 是 discrete visual tokenizer，兼顾 generation 和 understanding ([Foundation Vision](https://foundationvision.github.io/UniTok/?utm_source=chatgpt.com)) | 可作为 discrete unified tokenizer 复现对象，不是纯 benchmark |

------

## 我建议你的复现优先级

### 第一优先级：一定要做

**1. PSNR / SSIM / LPIPS / rFID**

这是最基本的 reconstruction baseline。你们之后可以说：

> Traditional reconstruction metrics are used as simple baselines.

这组最好自己写统一 pipeline：

```text
image
→ tokenizer.encode()
→ tokenizer.decode()
→ reconstruction
→ PSNR / SSIM / LPIPS / rFID
```

注意：这组只适合有 decoder 的 tokenizer。如果是 CLIP / SigLIP / DINO 这种只有 encoder 的视觉编码器，就不能直接做重构。

------

**2. TokBench**

TokBench 是你现在最应该重点复现的 benchmark project。它不是单个指标，而是完整的细粒度重构 benchmark。它关注 text 和 face，因为这些细节很容易在压缩/量化后丢失，而且对人眼和多模态任务都很敏感。TokBench 官方也指出，rFID、LPIPS、PSNR 这类常见指标对文字和人脸重构质量不够敏感。([GitHub](https://github.com/wjf5203/TokBench))

你可以在报告里这样定位：

> TokBench is used as a fine-grained reconstruction-oriented benchmark baseline.

------

**3. GVTBench**

如果你们第一阶段确实是：

```text
continuous visual tokenizers
+
image-conditioned MLLMs that generate text only
```

那 GVTBench 反而比 VTBench 更贴近你们。它不是重构 benchmark，而是研究不同视觉 tokenizer / vision encoder 接入 LLM 后对视觉语义理解和细粒度感知的影响。官方 repo 里提供了 Object Counting 和 Multi-Class Identification 的 GVTBench。([GitHub](https://github.com/TencentARC/GVT))

它可以作为：

> MLLM-understanding-oriented tokenizer baseline.

------

### 第二优先级：看你们后续是否做 discrete / generation

**4. VTBench**

VTBench 非常适合 discrete visual tokenizer / AR image generation 方向。它的定位是隔离 visual tokenizer 本身的质量，而不是只看完整生成模型的最终效果。它评估三类任务：image reconstruction、detail preservation、text preservation。([GitHub](https://github.com/huawei-lin/VTBench))

但是如果你们当前第一阶段只做 continuous encoder + text-only MLLM，VTBench 可以先调研，不一定马上复现。

------

**5. TokenFlow / UniTok 的 evaluation**

这两个不是 benchmark baseline，而是“统一 tokenizer 论文的评测流程参考”。如果之后你们扩展到：

```text
MLLM understands image
+
MLLM generates image
+
interleaved image-text output
```

它们会很有参考价值。

------

## 你最终表格可以这样归类

我建议你后续写文档时分成四类：

```text
A. Simple reconstruction metric baselines
   PSNR, SSIM, LPIPS, rFID

B. Reconstruction-oriented benchmark baselines
   TokBench, VTBench, swiss-ai tokenizer reconstruction benchmark

C. MLLM-understanding-oriented tokenizer baselines
   GVTBench, CLIP/DINO feature similarity, OCR/TextVQA-style semantic preservation

D. Downstream validation benchmarks
   VQAv2, GQA, TextVQA, OCRBench, MME, MM-Vet, MMBench, UniEval, MME-Unify
```

其中 **A/B/C 是你负责第二部分要找的 baseline**；**D 是用来验证这些 baseline 到底准不准的 downstream ground truth**。

------

## 最关键的一点

你可以这样记：

> TokBench 和 PSNR 都属于 reconstruction-based baseline，但 TokBench 是 benchmark project，PSNR 是 metric baseline。
> GVTBench 不属于重构，它更接近你们第一阶段的 MLLM tokenizer 评估。
> VTBench 更适合后续 discrete tokenizer / image generation tokenizer。

所以你现在最推荐先复现：

```text
PSNR / SSIM / LPIPS / rFID
+
TokBench
+
GVTBench
```

然后再考虑：

```text
VTBench
+
UniEval / MME-Unify
+
TokenFlow / UniTok evaluation
```
