import gc
import re
import numpy as np
import torch
import torch.nn as nn
from difflib import SequenceMatcher
from nltk.corpus import stopwords, wordnet
from llm_attacks import get_embedding_matrix, get_embeddings
import fastchat
from difflib import SequenceMatcher
def token_shift(origin_token, now_token, prefix, suffix):

    sm = SequenceMatcher(a=origin_token, b=now_token, autojunk=False)
    opcodes = sm.get_opcodes()
    delta_prefix = 0
    delta_suffix = 0

    for tag, i1, i2, j1, j2 in opcodes:

        if tag == "equal":
            pass
        elif tag == "replace":
            origin_s = i1
            origin_e = i2
            now_s = j1
            now_e = j2
            if origin_s >= prefix.start and origin_e <= prefix.stop:
                delta_prefix += (now_e - now_s) - (origin_e - origin_s)
            if origin_s >= suffix.start and origin_e <= suffix.stop:
                delta_suffix += (now_e - now_s) - (origin_e - origin_s)
        elif tag == "insert":
            origin_s = i1
            origin_e = i2
            now_s = j1
            now_e = j2
            insert_len = now_e - now_s
            if origin_s >= prefix.start and origin_e <= prefix.stop:
                delta_prefix += insert_len
            if origin_s >= suffix.start and origin_e <= suffix.stop:
                delta_suffix += insert_len
        elif tag == "delete":
            origin_s = i1
            origin_e = i2

            delete_len = origin_s - origin_e

            if origin_s >= prefix.start and origin_e <= prefix.stop:
                delta_prefix += delete_len
            if origin_s >= suffix.start and origin_e <= suffix.stop:
                delta_suffix += delete_len
        else:
            assert 1==2
    prefix_n_start = prefix.start
    prefix_n_stop = prefix.stop + delta_prefix
    suffix_n_start = suffix.start + delta_prefix
    suffix_n_stop = suffix.stop + delta_prefix + delta_suffix

    return prefix_n_start, prefix_n_stop, suffix_n_start, suffix_n_stop

def get_developer(model_name):
    developer_dict = {"llama-2": "Meta", "vicuna": "LMSYS",
                      "guanaco": "TheBlokeAI", "WizardLM": "WizardLM",
                      "mpt-chat": "MosaicML", "mpt-instruct": "MosaicML", "falcon": "TII"}
    return developer_dict[model_name]

def generate(model, tokenizer, input_ids, assistant_role_slice, gen_config=None):
    if gen_config is None:
        gen_config = model.generation_config
        gen_config.max_new_tokens = 64 # 32

    if gen_config.max_new_tokens > 50:
        print('WARNING: max_new_tokens > 32 may cause testing to slow down.')

    input_ids = input_ids[:assistant_role_slice.stop].to(model.device).unsqueeze(0)
    attn_masks = torch.ones_like(input_ids).to(model.device)
    output_ids = model.generate(input_ids,
                                attention_mask=attn_masks,
                                generation_config=gen_config,
                                pad_token_id=tokenizer.pad_token_id)[0]

    return output_ids[assistant_role_slice.stop:]


def check_for_attack_success(model, tokenizer, input_ids, assistant_role_slice, test_prefixes, gen_config=None):
    gen_str = tokenizer.decode(generate(model,
                                        tokenizer,
                                        input_ids,
                                        assistant_role_slice,
                                        gen_config=gen_config)).strip()
    jailbroken = not any([prefix in gen_str for prefix in test_prefixes])
    return jailbroken



class autodan_SuffixManager:
    def __init__(self, *, tokenizer, conv_template, instruction, target, adv_string):

        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.instruction = instruction
        print(f"self.instruction = {instruction}")
        self.target = target
        self.adv_string = adv_string

    def get_prompt(self, adv_string=None, attack_model=None):

        """
        大致思路是：
        1. 找到每个tokenizer编码后[REPLACE]的位置，用于替换而已问题
        2. Replace的前半部分作为前缀，后半部分作为后缀，这两部分是可替换部分
        3. 最后输出的结果
            self.prompt 是嵌入攻击模板后的整个提示词
            self._control_slice_prefix、_control_slice_suffix 分别是前缀和后缀的切片
            self._target_slice 是目标结果部分切片
            self._loss_slice 是计算loss的切片
        4. 由于不同的模型tokenizer编码结果和模型自身的系统标志符号不同， 因此每个模型单独计算
        """

        assert attack_model in ["Llama2-7b-chat", "vicuna", "Llama-3.1-8B-Instruct", "Llama-3.2-3B-Instruct", "guanaco"]

        if attack_model == "Llama-3.1-8B-Instruct" or attack_model == "Llama-3.2-3B-Instruct":
            
            self.conv_template.messages = []
            self.conv_template.append_message(self.conv_template.roles[0], f"{adv_string}")
            prompt_user = self.conv_template.get_prompt()
            toks_adv_string = self.tokenizer(prompt_user).input_ids
            origin_temp_len = len(toks_adv_string) - 1  
            end_token = None
            self.prefix_start = 4  
            self.origin_suffix_end = len(toks_adv_string) - 1 
            for i in range(len(toks_adv_string)):

                replace_len = 4  
                if toks_adv_string[i:i + 4] == [510, 793, 36975, 60]:   
                    self.start_idx = i
                    end_token = toks_adv_string[i + replace_len:i + replace_len + 3]
                    print(f"end_token = {end_token}")
                    self.prefix_end = i  
                    self.origin_suffix_start = i + replace_len  
                    break
            assert end_token != None

            if adv_string is not None:
                self.adv_string = adv_string.replace('[REPLACE]', self.instruction.lower())
            print(f"self.conv_template.name = {self.conv_template.name}")

            self.conv_template.messages = []
            self.conv_template.append_message(self.conv_template.roles[0], f"{self.adv_string}")
            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks_role_none = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._assistant_role_slice = slice(0, len(toks_role_none))     
            if self.conv_template.name == 'meta-llama-3.1':
                print(f"check name = {self.conv_template.name}")
                self.conv_template.messages = []
                self.conv_template.append_message(self.conv_template.roles[0], f"{self.adv_string}")
                prompt_user = self.conv_template.get_prompt()
                prompt_user_id = self.tokenizer(prompt_user).input_ids
                self.suffix_end = len(prompt_user_id) - 1 


                replace_temp_len = len(prompt_user_id)
                replace_len = replace_temp_len - origin_temp_len 
                self.suffix_start = self.origin_suffix_start + replace_len
                self.suffix_end = self.origin_suffix_end + replace_len - 6

                self._control_slice_prefix = slice(self.prefix_start, self.prefix_end)
                self._control_slice_suffix = slice(self.suffix_start, self.suffix_end)
                self._control_slice = slice(self.prefix_start, self.suffix_end)  

                self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
                self.prompt = self.conv_template.get_prompt()
                encoding = self.tokenizer(self.prompt)
                toks = encoding.input_ids

                self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 6) 
                self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 6 - 1)

            self.conv_template.messages = []
            return self.prompt

        if attack_model=="vicuna":
            self.conv_template.messages = []
            self.conv_template.append_message(self.conv_template.roles[0], f"{adv_string}")
            prompt_user = self.conv_template.get_prompt()
            toks_adv_string = self.tokenizer(prompt_user).input_ids
            origin_temp_len = len(toks_adv_string) - 1 
            end_token = None
            self.prefix_start = 5  
            self.origin_suffix_end = len(toks_adv_string) - 1  
            for i in range(len(toks_adv_string)):

                replace_len = 5  
                if toks_adv_string[i:i + 3] ==  [518, 1525, 7390]:   
                    self.start_idx = i
                    end_token = toks_adv_string[i + replace_len:i + replace_len + 3]
                    print(f"end_token = {end_token}")
                    self.prefix_end = i  
                    self.origin_suffix_start = i + replace_len  
                    break
            assert end_token != None


            if adv_string is not None:
                self.adv_string = adv_string.replace('[REPLACE]', self.instruction.lower())
                print(f"self.conv_template.name = {self.conv_template.name}")

                self.conv_template.messages = []
                self.conv_template.append_message(self.conv_template.roles[0], f"{self.adv_string}")
                self.conv_template.append_message(self.conv_template.roles[1], None)
                toks_role_none = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._assistant_role_slice = slice(0, len(toks_role_none))     
                if self.conv_template.name == 'vicuna_v1.1':
                    print(f"check name = {self.conv_template.name}")
                    self.conv_template.messages = []
                    self.conv_template.append_message(self.conv_template.roles[0], f"{self.adv_string}")
                    prompt_user = self.conv_template.get_prompt()
                    prompt_user_id = self.tokenizer(prompt_user).input_ids
                    self.suffix_end = len(prompt_user_id) - 1  

                    replace_temp_len = len(prompt_user_id)
                    replace_len = replace_temp_len - origin_temp_len 
                    self.suffix_start = self.origin_suffix_start + replace_len
                    self.suffix_end = self.origin_suffix_end + replace_len - 1  
                    self._control_slice_prefix = slice(self.prefix_start, self.prefix_end)
                    self._control_slice_suffix = slice(self.suffix_start, self.suffix_end)
                    self._control_slice = slice(self.prefix_start, self.suffix_end)  

                    self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
                    self.prompt = self.conv_template.get_prompt()
                    encoding = self.tokenizer(self.prompt)
                    toks = encoding.input_ids

                    self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 6)  

                    self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 6 - 1)

                self.conv_template.messages = []
                return self.prompt


  
        if attack_model=="guanaco":
            self.conv_template.messages = []
            self.conv_template.append_message(self.conv_template.roles[0], f"{adv_string}")
            prompt_user = self.conv_template.get_prompt()
            toks_adv_string = self.tokenizer(prompt_user).input_ids
            origin_temp_len = len(toks_adv_string) - 1  
            end_token = None
            self.prefix_start = 8  
            self.origin_suffix_end = len(toks_adv_string) - 1  
            for i in range(len(toks_adv_string)):

                replace_len = 5  
                if toks_adv_string[i:i + 3] ==  [518, 1525, 7390]:   
                    self.start_idx = i
                    end_token = toks_adv_string[i + replace_len:i + replace_len + 3]
                    print(f"end_token = {end_token}")
                    self.prefix_end = i  
                    self.origin_suffix_start = i + replace_len  
                    break
            assert end_token != None


            if adv_string is not None:
                self.adv_string = adv_string.replace('[REPLACE]', self.instruction.lower())
                print(f"self.conv_template.name = {self.conv_template.name}")

                self.conv_template.messages = []
                self.conv_template.append_message(self.conv_template.roles[0], f"{self.adv_string}")
                self.conv_template.append_message(self.conv_template.roles[1], None)
                toks_role_none = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._assistant_role_slice = slice(0, len(toks_role_none))     
                if self.conv_template.name == 'oasst_llama':
                    print(f"check name = {self.conv_template.name}")
                    self.conv_template.messages = []
                    self.conv_template.append_message(self.conv_template.roles[0], f"{self.adv_string}")
                    prompt_user = self.conv_template.get_prompt()
                    prompt_user_id = self.tokenizer(prompt_user).input_ids
                    self.suffix_end = len(prompt_user_id) - 1 


                    replace_temp_len = len(prompt_user_id)
                    replace_len = replace_temp_len - origin_temp_len  
                    self.suffix_start = self.origin_suffix_start + replace_len
                    self.suffix_end = self.origin_suffix_end + replace_len - 1  

                    self._control_slice_prefix = slice(self.prefix_start, self.prefix_end)
                    self._control_slice_suffix = slice(self.suffix_start, self.suffix_end)
                    self._control_slice = slice(self.prefix_start, self.suffix_end) 

                    self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
                    self.prompt = self.conv_template.get_prompt()
                    encoding = self.tokenizer(self.prompt)
                    toks = encoding.input_ids

                    self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 6) 

                    self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 6 - 1)

                self.conv_template.messages = []
                return self.prompt    
            


        if attack_model == "Llama2-7b-chat":

            self.conv_template.messages = []
            self.conv_template.append_message(self.conv_template.roles[0], f"{adv_string}")
            prompt_user = self.conv_template.get_prompt()
            toks_adv_string = self.tokenizer(prompt_user).input_ids
            origin_temp_len = len(toks_adv_string) - 1 
            end_token = None

            self.prefix_start =4  
            self.origin_suffix_end = len(toks_adv_string) - 1  
            for i in range(len(toks_adv_string)):
                if toks_adv_string[i:i + 4] == [518, 1525, 7390, 11538]: 
                    self.start_idx = i
                    end_token = toks_adv_string[i + 5:i + 5 + 3]
                    print(f"end_token = {end_token}")
                    self.prefix_end = i  
                    self.origin_suffix_start = i + 5  
                    break
            assert end_token != None

            if adv_string is not None:
                self.adv_string = adv_string.replace('[REPLACE]', self.instruction.lower())
            print(f"self.conv_template.name = {self.conv_template.name}")

            self.conv_template.messages = []
            self.conv_template.append_message(self.conv_template.roles[0], f"{self.adv_string}")
            self.conv_template.append_message(self.conv_template.roles[1], None)
            toks_role_none = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._assistant_role_slice = slice(0, len(toks_role_none))              
            
            if self.conv_template.name == 'llama-2':
                print("check name")
                self.conv_template.messages = []
                self.conv_template.append_message(self.conv_template.roles[0], f"{self.adv_string}")
                prompt_user = self.conv_template.get_prompt()
                prompt_user_id = self.tokenizer(prompt_user).input_ids
                self.suffix_end = len(prompt_user_id) - 1 


                replace_temp_len = len(prompt_user_id)
                replace_len = replace_temp_len - origin_temp_len 
                self.suffix_start = self.origin_suffix_start + replace_len
                self.suffix_end = self.origin_suffix_end + replace_len - 1
  
                self._control_slice_prefix = slice(self.prefix_start, self.prefix_end)
                self._control_slice_suffix = slice(self.suffix_start, self.suffix_end)
                self._control_slice = slice(self.prefix_start, self.suffix_end) 

                self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
                self.prompt = self.conv_template.get_prompt()
                encoding = self.tokenizer(self.prompt)
                toks = encoding.input_ids

                self._target_slice = slice(self._assistant_role_slice.stop, len(toks) - 2) 
                self._loss_slice = slice(self._assistant_role_slice.stop - 1, len(toks) - 2 - 1)

            self.conv_template.messages = []
            return self.prompt

    def get_input_ids(self, adv_string=None, attack_model=None):
        prompt = self.get_prompt(adv_string=adv_string, attack_model=attack_model)
        print(f"prompt = {prompt}")
        # toks = self.tokenizer(prompt).input_ids
        if attack_model == "guanaco":
            toks = self.tokenizer(text = prompt).input_ids
        else:
            toks = self.tokenizer(prompt).input_ids
        input_ids = torch.tensor(toks[:self._target_slice.stop])

        return input_ids

def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):

    embed_weights = get_embedding_matrix(model)

    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [
            embeds[:, :input_slice.start, :],
            input_embeds,
            embeds[:, input_slice.stop:, :]
        ],
        dim=1)
    logits = model(inputs_embeds=full_embeds, use_cache=False).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0, loss_slice, :], targets)
    loss.backward()
    grad = one_hot.grad.clone()
    return grad, loss


def target_loss(logits, ids, target_slice):
    crit = nn.CrossEntropyLoss(reduction='none')
    loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)
    loss = crit(logits[loss_slice, :], ids[target_slice])
    return loss.mean(dim=-1)


def sample_control(control_toks, grad, batch_size, topk=256, temp=1, not_allowed_tokens=None,
                   freeze_start=None, freeze_end=None):
    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.infty

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)
    original_control_toks = control_toks.repeat(batch_size, 1)
    control_len = len(control_toks)

    if freeze_start is not None and freeze_end is not None:
        allowed_pos = [i for i in range(control_len) if not (freeze_start <= i < freeze_end)]
    else:
        allowed_pos = list(range(control_len))  

    allowed_pos_tensor = torch.tensor(allowed_pos, dtype=torch.float32, device=grad.device)

    sampled_indices = torch.linspace(0, len(allowed_pos_tensor) - 1, steps=batch_size)
    new_token_pos = allowed_pos_tensor[sampled_indices.long()].to(torch.int64)

    rand_idx = torch.randint(0, topk, (batch_size, 1), device=grad.device)
    new_token_val = torch.gather(top_indices[new_token_pos], 1, rand_idx)

    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    return new_control_toks


def assistant_role_slice_cal(tokenizer, conv_template, new_suffix_i, target):
    conv_template.messages = []

    conv_template.append_message(conv_template.roles[0], f"{new_suffix_i}")
    conv_template.append_message(conv_template.roles[1], None)
    toks = tokenizer(conv_template.get_prompt()).input_ids
    _assistant_role_slice = slice(0, len(toks))

    conv_template.update_last_message(f"{target}")
    toks = tokenizer(conv_template.get_prompt()).input_ids
    _target_slice = slice(_assistant_role_slice.stop, len(toks) - 2)
    _loss_slice = slice(_assistant_role_slice.stop - 1, len(toks) - 3)
    return _assistant_role_slice, _target_slice, _loss_slice


def batch_encode_i(origin_adv_suffix, new_adv_suffix, target, conv_template, tokenizer, origin_prefix_slice_list, origin_suffix_slice_list, batch_size):
    
    new_adv_suffix_list = []
    new_adv_suffix_idx_list = []
    new_adv_suffix_attention_list = []

    prefix_slice_list = []
    suffix_slice_list = []
    _control_slice_list = []
    assistant_role_slice_list = []
    _target_slice_list = []
    _loss_slice_list = []

    for batch_i in range(batch_size):
        new_suffix_i = new_adv_suffix[batch_i]
        conv_template.messages = []
        conv_template.append_message(conv_template.roles[0], f"{new_suffix_i}")
        conv_template.append_message(conv_template.roles[1], f"{target}")
        new_prompt = conv_template.get_prompt()

        new_adv_suffix_list.append(new_prompt)
        encoding = tokenizer(new_prompt)
        new_input_ids = encoding["input_ids"]
        new_attention = encoding["attention_mask"]
        new_adv_suffix_idx_list.append(new_input_ids)
        new_adv_suffix_attention_list.append(new_attention)

        origin_suffix_i = origin_adv_suffix[batch_i]
        conv_template.messages = []
        conv_template.append_message(conv_template.roles[0], f"{origin_suffix_i}")
        conv_template.append_message(conv_template.roles[1], f"{target}")
        origin_prompt = conv_template.get_prompt()
        origin_input_ids = tokenizer(origin_prompt)["input_ids"]

        origin_prefix_slice_i = origin_prefix_slice_list[batch_i]
        origin_suffix_slice_i = origin_suffix_slice_list[batch_i]

        prefix_n_start_i, prefix_n_stop_i, suffix_n_start_i, suffix_n_stop_i = token_shift(origin_input_ids,
                                                                                           new_input_ids,
                                                                                           origin_prefix_slice_i,
                                                                                           origin_suffix_slice_i)
        prefix_slice = slice(prefix_n_start_i, prefix_n_stop_i)
        prefix_slice_list.append(prefix_slice)

        suffix_slice = slice(suffix_n_start_i, suffix_n_stop_i)
        suffix_slice_list.append(suffix_slice)

        _control_slice = slice(prefix_slice.start, suffix_slice.stop)
        _control_slice_list.append(_control_slice)

        _assistant_role_slice, _target_slice, _loss_slice = assistant_role_slice_cal(tokenizer, conv_template, new_suffix_i, target)

        assistant_role_slice_list.append(_assistant_role_slice)
        _target_slice_list.append(_target_slice)
        _loss_slice_list.append(_loss_slice)

    return new_adv_suffix_list, new_adv_suffix_idx_list, new_adv_suffix_attention_list, prefix_slice_list, \
        suffix_slice_list, _control_slice_list, assistant_role_slice_list, _target_slice_list, _loss_slice_list

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)


def compute_batch_control_grads_and_losses_i(model,
                                             tokenizer,
                                             new_adv_suffix_list,
                                             new_adv_suffix_idx_list,
                                             new_adv_suffix_attention_list,
                                             prefix_slice_list,
                                             suffix_slice_list,
                                             _control_slice_list,
                                             assistant_role_slice_list,
                                             _target_slice_list,
                                             _loss_slice_list,
                                             batch_size,
                                             device):
    embed_layer = model.get_input_embeddings()
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    grad_control_list = []
    tokens_batch_list = []
    loss_list = []
    for cal_i in range(batch_size):
        input_ids = torch.tensor(new_adv_suffix_idx_list[cal_i]).to(device).unsqueeze(0)
        attention_mask = torch.tensor(new_adv_suffix_attention_list[cal_i]).to(device).unsqueeze(0)
        control_slice = _control_slice_list[cal_i]
        target_slice = _target_slice_list[cal_i]
        loss_slice = _loss_slice_list[cal_i]

        with torch.no_grad():
            embeds = embed_layer(input_ids)  

        control_embeds = embed_layer(input_ids[0, control_slice]).detach().clone()
        control_embeds.requires_grad_() 

        full_embeds = torch.cat([
            embeds[0, :control_slice.start, :],
            control_embeds,
            embeds[0, control_slice.stop:, :]
        ], dim=0).unsqueeze(0)  
    
        logits = model(inputs_embeds=full_embeds, attention_mask=attention_mask, use_cache=False).logits  # [B, L, V]
        targets = input_ids[0, target_slice]            
        logits_slice = logits[0, loss_slice, :] 


        loss_per_token = loss_fn(logits_slice, targets)  

        loss = loss_per_token.mean()  # scalar
        loss.backward()

        grad_control = control_embeds.grad.detach()  
        tokens_batch = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]

        grad_control_list.append(grad_control)
        loss_list.append(loss.detach().item())
        tokens_batch_list.append(tokens_batch)

        del full_embeds, embeds, control_embeds, logits, logits_slice, loss, loss_per_token, targets
        torch.cuda.empty_cache()
        gc.collect()
    return grad_control_list, loss_list, tokens_batch_list

def generate(model, tokenizer, input_ids, assistant_role_slice, gen_config=None):
    if gen_config is None:
        gen_config = model.generation_config
        gen_config.max_new_tokens = 32

    if gen_config.max_new_tokens > 50:
        print('WARNING: max_new_tokens > 32 may cause testing to slow down.')

    input_ids = input_ids[:assistant_role_slice.stop].to(model.device).unsqueeze(0)
    attn_masks = torch.ones_like(input_ids).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(input_ids,
                                    attention_mask=attn_masks,
                                    generation_config=gen_config,
                                    pad_token_id=tokenizer.pad_token_id)[0]

    return output_ids[assistant_role_slice.stop:], output_ids

def check_for_attack_success(model, tokenizer, input_ids, assistant_role_slice, test_prefixes, gen_config=None):
    gen_str = tokenizer.decode(generate(model,
                                        tokenizer,
                                        input_ids,
                                        assistant_role_slice,
                                        gen_config=gen_config)[0]).strip()

    jailbroken = not any([prefix in gen_str for prefix in test_prefixes])
    return jailbroken, gen_str 

import re

def normalize_syn(syn: str) -> str:
    return syn.replace('_', ' ')

_space_collapse_re = re.compile(r'\s+')

def postprocess_text(text: str) -> str:
    return _space_collapse_re.sub(' ', text).strip()

def evolve_batch_control_suffixes_from_grad_dot_i(
        grad_control_list,
        input_ids_list,
        control_slice_list,
        tokenizer,
        model,
        get_synonyms_func,
        prefix_slice_list,
        suffix_slice_list,
        top_percent=0.2,
        batch_size=64

):

    embed_layer = model.get_input_embeddings()
    offspring_suffixes = []
    special_tokens = {"<s>", "</s>", "[INST]", "[/INST]", "<", ">", "/", "[", "[/", "<<", ">>", "<</", "<</SYS>>", "<<SYS>>", 
                      tokenizer.pad_token, tokenizer.eos_token, tokenizer.bos_token}

    for evolve_i in range(batch_size):

        control_slice = control_slice_list[evolve_i]
        prefix_slice = prefix_slice_list[evolve_i]
        suffix_slice = suffix_slice_list[evolve_i]
        input_id_i = input_ids_list[evolve_i]
        grads = grad_control_list[evolve_i] 
        scores = torch.norm(grads, dim=1)  
        scores[prefix_slice.stop:suffix_slice.start] = 0
        Lc = grads.shape[0]
        top_k = max(1, int(Lc * top_percent))
        top_indices = torch.topk(scores, k=top_k).indices.tolist()
        top_indices.sort(reverse=True)   # 降序

        control_ids = input_id_i[:control_slice.stop]
        new_control_ids = control_ids.copy()

    
        for idx in top_indices:   


            if not (idx >= prefix_slice.start and idx < prefix_slice.stop) and not (idx >= suffix_slice.start and idx < suffix_slice.stop):

                continue

            token_id = control_ids[idx]
            token_str = tokenizer.convert_ids_to_tokens(token_id)

            word = tokenizer.convert_tokens_to_string([token_str]).strip()

            if token_str.startswith("▁") and len(token_str.strip("▁")) == 0:
                continue

            if token_str in special_tokens:
                continue

            if not re.fullmatch(r"[A-Za-z\-]+", word):
                continue

            raw_syns = get_synonyms_func(word)
            syns = [normalize_syn(s) for s in raw_syns if '_' in s or s.isalpha()]
            if not syns:
                continue

            token_grad = grads[idx]
            dot_scores = []
            original_token = tokenizer.convert_ids_to_tokens(token_id)
            prefix_space = original_token.startswith("▁")  
            orig_embed = embed_layer(torch.tensor([token_id], device=token_grad.device)).squeeze(0)
            for syn in syns:
                syn_text = (" " if prefix_space and not syn.startswith("▁") else "") + syn
                syn_tokens = tokenizer.tokenize(syn_text)
                syn_ids = tokenizer.convert_tokens_to_ids(syn_tokens)
                if tokenizer.unk_token_id in syn_ids:
                    continue
                syn_embed = embed_layer(torch.tensor(syn_ids, device=token_grad.device)).mean(dim=0)
                delta_embed = syn_embed - orig_embed
                dot_score = torch.dot(token_grad, delta_embed).item()
                dot_scores.append((syn, dot_score))

            if not dot_scores:
                continue

            dot_scores.sort(key=lambda x: x[1])  
            best_syn = dot_scores[0][0]

            new_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" " + best_syn))

            del new_control_ids[idx]
            for nid in reversed(new_ids):
                new_control_ids.insert(idx, nid)
        torch.cuda.empty_cache()
        new_control_suffix = tokenizer.decode(new_control_ids[control_slice.start:], skip_special_tokens=True)
        offspring_suffixes.append(new_control_suffix.strip())

    return offspring_suffixes



def load_conversation_template(template_name):
    if template_name == 'llama2':
        template_name = 'llama-2'
    conv_template = fastchat.model.get_conversation_template(template_name)
    if conv_template.name == 'zero_shot':
        assert 1==2
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()
        conv_template.system = "[INST] <<SYS>>\n\n<</SYS>>\n\n"
    elif conv_template.name == 'meta-llama-3.1':
        conv_template.system_message = ""
    else:
        conv_template.system_message = ""
    return conv_template
