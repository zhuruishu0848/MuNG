# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


from dataclasses import dataclass, field
import json
import re
import math
import logging
import os
import random
# from datasets import Dataset
from PIL import Image
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import (
    Trainer, GPTQConfig,
    AutoProcessor, 
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Qwen2_5_VLForConditionalGeneration, 
    Qwen2_5_VLProcessor
)
from transformers.integrations import deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from qwen_vl_utils import process_vision_info
from peft import (
        LoraConfig, get_peft_model, prepare_model_for_kbit_training, 
        PromptTuningConfig, TaskType, PromptTuningInit,
        PrefixEncoder, PrefixTuningConfig)
from accelerate.utils import DistributedType

from noise_model import Qwen2_5_VL_Noise
import wandb
random.seed(1299)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")
    train_noise_generator: bool = False
    ng_heads: int = 4
    noise_generator_type: str = "NG_vlt_CA"



@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    use_dora: bool = False
    use_prompt_tuning: bool = False
    use_prefix_tuning: bool = False
    # max_grad_norm: float = 500
    fix_vit: bool = True

@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        # default_factory=lambda: ["c_attn", "attn.c_proj", "w1", "w2"] 
        # default_factory=lambda: ["in_proj","out_proj","c_fc"]
        default_factory=lambda: ["k_proj", "o_proj", "q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        if trainer.args.use_lora or trainer.args.use_dora:
            state_dict = get_peft_state_maybe_zero_3(
                trainer.model.named_parameters(), bias
            )
        else:
            state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        trainer._save(output_dir, state_dict=state_dict)


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    input_ids, attention_mask, labels = [], [], []
    sources = sources[0]
    input_content = sources["question"]
    output_content = sources["answer"]

    if type(sources["image"]) == list:
        file_path_1 =sources["image"][0]
        file_path_2 =sources["image"][1]
        messages = [
            {"role": "user", "content": [
                    {
                        "type": "image",
                        "image": f"{file_path_1}",
                        "image": f"{file_path_2}",
                        "resized_height": 512,
                        "resized_width": 512,
                    },
                    {"type": "text", "text": f"{input_content}"},
                ],
            }
        ]
    else:
        file_path =sources["image"]
        messages = [
            {"role": "user", "content": [
                    {
                        "type": "image",
                        "image": f"{file_path}",
                        "resized_height": 512,
                        "resized_width": 512,
                    },
                    {"type": "text", "text": f"{input_content}"},
                ],
            }
        ]
        
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    ) 


    image_inputs, video_inputs = process_vision_info(messages) 
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = {key: value.tolist() for key, value in inputs.items()}
    instruction = inputs

    response = tokenizer(f"{output_content}", add_special_tokens=False)


    input_ids = (
            instruction["input_ids"][0] + response["input_ids"] 
    )
    input_ids += [tokenizer.pad_token_id] * max((max_len - len(input_ids)),0)

    attention_mask = instruction["attention_mask"][0] + response["attention_mask"]
    attention_mask += [1] * max((max_len - len(attention_mask)),0)
    assert len(input_ids) == len(attention_mask)
    labels = [-100] * len(instruction["input_ids"][0]) + response["input_ids"]
    labels += [tokenizer.pad_token_id] * max((max_len - len(labels)),0)
    assert len(labels) == len(attention_mask)

    if len(input_ids) > max_len: 
        input_ids = input_ids[:max_len]
        attention_mask = attention_mask[:max_len]
        labels = labels[:max_len]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0) 
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw']}


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, processor, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]
        self.pixel_values = data_dict["pixel_values"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return {
            "input_ids": self.input_ids[i],
            "labels": self.labels[i],
            "attention_mask": self.attention_mask[i],
            "pixel_values": self.pixel_values[i] if self.pixel_values else None
        }

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int, cache_size=100):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        ret = preprocess([self.raw_data[i]], self.tokenizer, self.max_len)
        ret = dict(
            input_ids = ret["input_ids"],
            labels = ret["labels"],
            attention_mask = ret["attention_mask"],
            pixel_values = ret["pixel_values"],
            image_grid_thw = ret["image_grid_thw"]
        )

        return ret
    



def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    with open(data_args.data_path, "r", encoding="utf-8") as f:
        train_json = [json.loads(line) for line in f]


    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

def train():
    global local_rank
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    
    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    config.use_cache = False

    if model_args.train_noise_generator:
        model = Qwen2_5_VL_Noise.from_pretrained(
                    model_args.model_name_or_path,
                    config=config,
                    n_heads=model_args.ng_heads,
                    noise_generator_type=model_args.noise_generator_type,
                    cache_dir=training_args.cache_dir,
                    device_map=device_map,
                    quantization_config=GPTQConfig(
                        bits=4, disable_exllama=True
                    )
                    if training_args.use_lora and lora_args.q_lora
                    else None,
                )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_args.model_name_or_path,
                    config=config,
                    cache_dir=training_args.cache_dir,
                    device_map=device_map,
                    quantization_config=GPTQConfig(
                        bits=4, disable_exllama=True
                    )
                    if training_args.use_lora and lora_args.q_lora or (training_args.use_dora and lora_args.q_lora)
                    else None,
                    )

    if not training_args.use_lora and not training_args.use_dora:
        if training_args.fix_vit and hasattr(model,'transformer') and hasattr(model.transformer,'visual'):
            model.transformer.visual.requires_grad_(False)
            if hasattr(model.transformer.visual,'attn_pool'):
                model.transformer.visual.attn_pool.requires_grad_(True)


    global processor, tokenizer
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    if training_args.use_lora or training_args.use_dora:
        if lora_args.q_lora or "chat" in model_args.model_name_or_path.lower():
            modules_to_save = None
        else:
            modules_to_save = ["wte"]
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    if training_args.use_dora:
        if lora_args.q_lora or "chat" in model_args.model_name_or_path.lower():
            modules_to_save = None
        else:
            modules_to_save = ["wte"]
        lora_config = LoraConfig(
            use_dora=True,
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save  # This argument serves for adding new tokens.
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()


    if model_args.train_noise_generator:
        model.requires_grad_(False)
        for p in model.noise_generator.parameters():
            p.requires_grad = True
 

    # Load data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    for name, param in model.named_parameters():
        if not param.requires_grad:
            # pass
            print(f"{name} is frozen: shape={param.shape}, dtype:{param.dtype}")
        else:
            # pass
            print(f"{name} is trainable: shape={param.shape}, requires_grad={param.requires_grad}")

    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    try:
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
    except:
        pass

    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)


if __name__ == "__main__":
    train()
