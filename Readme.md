# MuNG: Exploring Beneficial Noise Injection in MLLMs

Multimodal Large Language Models (MLLMs) have played an increasingly important role in multimodal intelligence. However, the existing fine-tuning methods often ignore cross-modal heterogeneity, limiting their full potential. In this work, we propose a novel fine-tuning strategy by injecting beneficial random noise, which outperforms previous methods and even surpasses full fine-tuning, with minimal additional parameters. The proposed Multimodal Noise Generator (MuNG) enables efficient modality fine-tuning by injecting customized noise into the frozen MLLMs. Specifically, we reformulate the reasoning process of MLLMs from a variational inference perspective, upon which we design a multimodal noise generator that dynamically analyzes cross-modal relationships in image-text pairs to generate task-adaptive beneficial noise. Injecting this type of noise into the MLLMs effectively suppresses irrelevant semantic components, leading to significantly improved cross-modal representation alignment and enhanced performance on downstream tasks. Experiments on two mainstream MLLMs, QwenVL and LLaVA, demonstrate that our method surpasses full-parameter fine-tuning and other existing fine-tuning approaches, while requiring adjustments to only about $1\%$ additional parameters. The relevant code is uploaded in the supplementary.

---

## 🚀 Quick Start

###  Step 1. Clone Repositories

First, clone the two required base repositories:

```bash
git clone https://github.com/haotian-liu/LLaVA.git
git clone https://github.com/QwenLM/Qwen2.5-VL.git
```

Place the MuNG-related code in the same directory level as these two repositories, or adjust the script paths accordingly.

---

###  Step 2. Install Dependencies

Please follow the respective instructions to set up the environments:

* [LLaVA Installation Guide](https://github.com/haotian-liu/LLaVA#installation)
* [Qwen2.5-VL Installation Guide](https://github.com/QwenLM/Qwen2.5-VL#installation)

---

###  Step 3. Prepare Datasets

#### 📂 LLaVA Dataset

Download the instruction tuning annotation file:

* [llava\_v1\_5\_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json)

Download the corresponding image datasets:

* **COCO**: [train2017.zip](http://images.cocodataset.org/zips/train2017.zip)
* **GQA**: [images.zip](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
* **OCR-VQA**: [Download Script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing) *(save all images as `.jpg`)*
* **TextVQA**: [train\_val\_images.zip](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
* **VisualGenome**: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

---

#### 📂 QwenVL Dataset

Download the [MMPR v1.1 dataset](https://huggingface.co/datasets/OpenGVLab/MMPR-v1.1) and place it under the `./datasets` directory.

Then convert it to the format required by Qwen:

```bash
cd Qwen2.5-VL-noise
python datasets/MMPR-shuffle.py
```

---

### 🔧 Step 4. Fine-tuning

Select the appropriate script to fine-tune either the LLaVA or QwenVL model with noise injection:

```bash
# Fine-tune LLaVA
cd LLaVA-noise
bash scripts/v1_5/finetune_noise_lora.sh

# Fine-tune QwenVL
cd Qwen2.5-VL-noise
bash noise_generator/finetune/MMPR-shuff/mmpr_finetune_ng.sh
```

---

### 🧪 Step 5. Evaluation

#### ✅ LLaVA Evaluation

Please refer to the official documentation:
[LLaVA Evaluation Guide](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md)

#### ✅ QwenVL Evaluation

QwenVL uses [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for evaluation.

---

## 🙏 Acknowledgments

This project is built upon the following open-source resources:

* [LLaVA](https://github.com/haotian-liu/LLaVA)
* [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
* [InternVL](https://github.com/OpenGVLab/InternVL)
* [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)