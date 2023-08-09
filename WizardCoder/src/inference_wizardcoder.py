import itertools
import sys
import os
import fire
import torch
import transformers
import json
import jsonlines

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass

torch.cuda.empty_cache()

def evaluate(
        batch_data,
        tokenizer,
        model,
        input=None,
        temperature=1,
        top_p=0.9,
        top_k=40,
        num_beams=1,
        max_new_tokens=2048,
        **kwargs,
):
    prompts = generate_prompt_batch(batch_data, input)
    inputs = tokenizer(prompts, return_tensors="pt", max_length=256, truncation=True, padding=True)
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences
    output = tokenizer.batch_decode(s, skip_special_tokens=True)
    return output


def generate_prompt(docstring, input=None):
    return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Generate a python function that has this docstring:
\"\"\"{docstring}\"\"\"
Do not respond with anything other than the function.

### Response:"""


def generate_prompt_batch(batch_data, input=None):
    if isinstance(batch_data, str):
        return batch_data
    return [generate_prompt(js["docstring"], input) for js in batch_data]

def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(itertools.islice(it, n))
        if not batch:
            return
        yield batch

def main(
    load_8bit: bool = False,
    base_model: str = "Model_Path",
    input_data_path = "Input.jsonl",
    output_data_path = "Output.jsonl",
    batch_size: int = 1,
):
    assert base_model, (
        "Please specify a --base_model, e.g. --base_model='WizardLM/WizardCoder-15B-V1.0'"
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    elif device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )

    model.config.pad_token_id = tokenizer.pad_token_id

    if not load_8bit:
        model.half()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    input_data = jsonlines.open(input_data_path, mode='r')
    output_data = jsonlines.open(output_data_path, mode='w')


    for jsons in batched(input_data, batch_size):
        # id = one_data["idx"]
        # instruction = one_data["Instruction"]
        # print(instruction)
        # docstring = one_data["docstring"]
        # print(docstring)
        _output = evaluate(jsons, tokenizer, model)
        print(_output)
        # final_output = _output[0].split("### Response:")[1].strip()
        # new_data = {
        #     "id": id,
        #     "docstring": docstring,
        #     "wizardcoder": final_output
        # }
        # output_data.write(new_data)

        new_data = [
            {"id": js["idx"],
             "docstring": js["docstring"],
             "wizardcoder": output.split("### Response:")[1].strip()} for
            js, output in zip(jsons, _output)
        ]
        output_data.write_all(new_data)

    output_data.close()
    input_data.close()



if __name__ == "__main__":
    fire.Fire(main)