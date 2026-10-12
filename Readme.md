# **README.md**

## 🔍 Project Overview

This repository provides the official implementation of Gradient-Guided Mutation (GGM), a jailbreak attack method for large language models (LLMs). GGM leverages gradient-guided mutation to optimize adversarial prompts and evaluate their effectiveness against LLM safety mechanisms.
------

## 📁 Repository Structure

```
├── main.py                            # Adversarial suffix optimization 
├── function.py                        # Helper functions used by main.py
├── LLM_generate.py                    # Generate 256-token responses using the optimized suffix
├── deepseek_check_template.py         # Semantic safety audit using DeepSeek LLM
├── NNL.py                             # Compute NNL scores for risk evaluation
├── requirements.txt                   # All dependencies
└── README.md
```

------

## 🚀 Usage

### **1️⃣ Step 1 — Adversarial Suffix Optimization**

Run the optimization script to produce gradient-informed adversarial suffixes.

```bash
python main.py --start 0 --end 520 --save_name main_result --attack_model Llama2-7b-chat
```

------

### **2️⃣ Step 2 — Generate 256-Token Outputs**

Use the optimized suffix to query the target LLM.

```bash
python LLM_generate.py --test_topk 3 --attack_model Llama-2-7B-chat --method Ours --start 0 --end 520
```

------

### **3️⃣ Step 3 — DeepSeek Semantic Safety Audit**

This stage performs contextual safety evaluation beyond keyword filtering.

```bash
python deepseek_check_template.py
```

------

### **4️⃣ Step 4 — NNL Calculation**

Compute the **Normalized Negative Likelihood** score for each sample to quantify attack severity.

```bash
python NNL.py
```





------

## 📝 Requirements

All dependencies are listed in the provided `requirements.txt` file.
 You can install them with:

```
pip install -r requirements.txt
```

This will automatically set up all required libraries for running the optimization, generation, auditing, and evaluation scripts.

### FastChat Dependency

This project relies on the FastChat library  (https://github.com/lm-sys/FastChat) for model inference and chat formatting.

For reproducibility, FastChat is installed from source at a fixed commit, as specified in `requirements.txt`.
