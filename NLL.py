import math, random, re, numpy as np, torch, pandas as pd
import torch.nn.functional as F
from typing import List, Optional, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
from tqdm import tqdm
import json
import os

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
EVAL_LM = "gpt2-large"  
random.seed(42); np.random.seed(42); torch.manual_seed(42)

tokenizer = AutoTokenizer.from_pretrained(EVAL_LM)
PPL_model = AutoModelForCausalLM.from_pretrained(EVAL_LM).to(DEVICE).eval()


def PPL_HF(encodings, model):
    # print(f"encodings = {encodings}")
    max_length = model.config.n_positions
    stride = 512
    seq_len = len(encodings)
    seq_len = encodings.input_ids.size(1)
    device = model.device

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        # Accumulate the total negative log-likelihood and the total number of tokens
        num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token

    return avg_nll

SPECIAL_STR = {"<s>", "</s>", "<pad>", "<unk>", "[INST] "," [/INST]","\nuser: ", "\nassistant:", "\nuser: ", "\nassistant:"," USER: ", " ASSISTANT:", "<|prompter|>", "</s><|assistant|>"
}
PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)

def is_content_token(tok_str: str):
    return (tok_str not in SPECIAL_STR) and (not PUNCT_RE.match(tok_str))



def delta_ppl_for_template(PPL_model,
                           template: str
                           ) -> dict:
    encodings = tokenizer(template, return_tensors='pt')

    ppl_orig = PPL_HF(encodings, PPL_model)
    return {"ppl": ppl_orig}

def NNL_for_templates(PPL_model,
                         templates) -> pd.DataFrame:
    results = []
    
    out = delta_ppl_for_template(PPL_model, templates)
    results.append({"template": templates,
                        "ppl": out["ppl"]})
    df = pd.DataFrame(results)

    return df

def cal_PPL(data_path, save_path, attack_model_i, PPL_model, strat_idx=0, end_idx=520):
    
    with open(data_path) as f:
        data = json.load(f)
    keys_list = list(data.keys())[strat_idx:end_idx]
    data_recoder = {}
    for key_i in keys_list:
        data_recoder_i = {}
        goal = data[key_i]['goal']
        data_test_i = data[key_i]['data_test_i']

        if attack_model_i == "Llama-2-7B-chat":
            template_start = "[INST] "
            template_end = " [/INST]"

        if attack_model_i == "Llama-3.1-8B-Instruct":
            template_start = "\nuser: "
            template_end = "\nassistant:"

        if attack_model_i == "Llama-3.2-3B-Instruct":
            template_start = "\nuser: "
            template_end = "\nassistant:"

        if attack_model_i == "vicuna-7B":
            template_start = " USER: "
            template_end = " ASSISTANT:"

        if attack_model_i == "guanaco-7B":
            template_start = "<|prompter|>"
            template_end = "</s><|assistant|>"
        data_re = data_test_i
        data_re = data_re.split(template_start)[-1]
        data_re = data_re.split(template_end)[0]
        print(f"data_re = {data_re}")

        df = NNL_for_templates(PPL_model=PPL_model, templates=data_re)

     
        # PPL_i, a, b = PPL_cal.filter_window(data_re)
        data_recoder_i['goal'] = goal
        data_recoder_i['data_test'] = data_test_i
        data_recoder_i['data_re'] = data_re   # 可以用于后续其他实验
        data_recoder_i['PPL_i'] = float(df['ppl'].values[0])
        data_recoder_i['is_success_origin'] = str(data[key_i]['is_success_origin'])
        data_recoder_i['is_success_256'] = str(data[key_i]['is_success_256'])
        data_recoder[key_i] = data_recoder_i

        with open(save_path, 'w') as f:
            json.dump(data_recoder, f)
    print(f"finish")


method_list = ["GCG", "AutoDAN_HGA", "Ours"]
attack_model_list = ["vicuna-7B", "guanaco-7B", "Llama-2-7B-chat", "Llama-3.1-8B-Instruct", "Llama-3.2-3B-Instruct"]

for check_i in range(3):
    folder_path = f"./Result_F/LLM_generate/check_3_template_{str(check_i)}"
    save_folder = f"./Result_F/Delta_score/check_3_template_{str(check_i)}"
    for method_i in method_list:
        attack_model_path = os.path.join(folder_path, method_i)
        print(f"attack_model_path: {attack_model_path}")
        for attack_model_i in attack_model_list:
            data_path_i = os.path.join(attack_model_path, attack_model_i, f"{method_i}_{attack_model_i}_all_256_result_0_520.json")
            # break
            save_folder_i = os.path.join(save_folder, method_i, attack_model_i)
            os.makedirs(save_folder_i, exist_ok=True)
            save_path_i = os.path.join(save_folder_i, f"{method_i}_{attack_model_i}_test_NLL.json")
            with torch.no_grad():
                cal_PPL(data_path_i, save_path_i, attack_model_i, PPL_model, strat_idx=0, end_idx=520)

