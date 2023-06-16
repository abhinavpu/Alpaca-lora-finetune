# !pip install -U pip
# !pip install accelerate==0.19.0
# !pip install appdirs==1.4.4
# !pip install bitsandbytes==0.37.2
# !pip install datasets==2.10.1
# !pip install git+https://github.com/huggingface/peft.git
# !pip install git+https://github.com/huggingface/transformers.git
# !pip install torch==2.0.0
# !pip install sentencepiece==0.1.97

#packages
import sys
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)

import torch
from datasets import load_dataset
import ast

#prompt generator
def generate_prompt(data):
    
    inp = ast.literal_eval(data['input'])[0]['source_data'][0]
    inp_data = []
    for idx, (tab_col, data_list) in enumerate(inp.items()):
        inp_data.append(f"{idx + 1}. {tab_col}: {data_list}")
    final_input = "\n".join(inp_data)
    
    prompt = f"""Problem:
Given the target field below, you need to generate a set of sequential rules or instructions based on the provided source inputs.

target_field = {ast.literal_eval(data['input'])[0]['target_field']}

Source Inputs:
{final_input}

Desired Output (Sequential Rules):
{data['output']}

"""

    return prompt


def fine_tune():
    BASE_MODEL = "decapoda-research/llama-7b-hf"

    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "left"
    
    def tokenize(prompt, add_eos_token=True):

        result = tokenizer(
            prompt,
            truncation=True,
            max_length=CUTOFF_LEN,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < CUTOFF_LEN
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    #generates tokenized prompts
    def generate_and_tokenize_prompt(data_point):
        full_prompt = generate_prompt(data_point)
        tokenized_full_prompt = tokenize(full_prompt,add_eos_token=True)
        return tokenized_full_prompt


    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    training_arguments = transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS ,#100,
        max_steps=TRAIN_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10 ,#10,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=50 ,#50,
        save_steps=50 ,#50,
        output_dir=OUTPUT_DIR,
        save_total_limit=3,
        load_best_model_at_end=True,
        # report_to="tensorboard" 
    )
    

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    ########## data ###########

    data = load_dataset("json", data_files=DATA_FILES_PATH)

    train_val = data["train"].train_test_split(
        test_size=TEST_SIZE, shuffle=True, seed=42
    )
    train_data = (
        train_val["train"].shuffle().map(lambda x: generate_and_tokenize_prompt(x))
    )
    val_data = (
        train_val["test"].shuffle().map(lambda x: generate_and_tokenize_prompt(x))
    )

    ########### train ###########
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_arguments,
        data_collator=data_collator
        )
    
    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)


if __name__ == '__main__':

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    #LoRa params
    CUTOFF_LEN = 512
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT= 0.05
    LORA_TARGET_MODULES = [
        "q_proj",
        "v_proj",
    ]

    #Training Params
    BATCH_SIZE = 512
    WARMUP_STEPS = 100
    MICRO_BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS =  BATCH_SIZE // MICRO_BATCH_SIZE 
    LEARNING_RATE = 3e-4 
    TRAIN_STEPS = 1000
    OUTPUT_DIR = "MODEL_SAVE_DIR"
    
    TEST_SIZE = 10000 #total record size is 114100
    DATA_FILES_PATH = "C:\Users\sahil.patil\OneDrive - Fresh Gravity\Work\LLM\smaller5.json"
    
    #train
    fine_tune()