import re
import random
import torch
import transformers
from transformers.models.llama.modeling_llama import *


import waverope1

device = torch.device('cuda:0')

def generate_prompt(n_garbage):
    """Generates a text file and inserts an execute line at a random position."""
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 2000)

    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key?"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question
    ]
    return "\n".join(lines), pass_key  # prompt, number


def load_model(models_path):
    models = list()
    for model_path in models_path:
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device
        )

        # from tensor_parallel import tensor_parallel
        # model = tensor_parallel(model)

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        models.append((model, tokenizer))

    return models


def evaluate_model(models_path, num_tests=2):
    models_list = load_model(models_path)
    results = {f"model_{i}": {"num_tokens": [], "accuracy": []}
               for i in range(len(models_path))}

    n_values = [5000, 8000, 10000, 12000, 14000, 18000, 20000, 25000, 30000,
                35000, 40000, 45000, 50000]


    for n in n_values:
        print(f"the garbage length is {n}.")
        models_metric = {f"model_{i}": {"num_tokens": 0, "accuracy": 0}
                         for i in range(len(models_path))}

        for i in range(num_tests):
            prompt_text, pass_key = generate_prompt(n)
            query_text = f"[INST] {prompt_text.strip()} [/INST] Pass key is "



            for i, (model, tokenizer) in enumerate(models_list):
                num_tokens = len(tokenizer.encode(prompt_text))
                inputs = tokenizer(query_text, return_tensors="pt").to(device)

                models_metric[f'model_{i}']['num_tokens'] += num_tokens


                generate_ids = model.generate(inputs.input_ids,
                                              max_new_tokens=30,
                                              temperature=0.6,
                                              top_p=0.9)

                print(inputs.input_ids.shape)

                response = tokenizer.batch_decode(generate_ids)[0][
                           len(query_text)+4:]
                try:
                    predict_pass_key = int(re.search(r'\d+', response).group())
                    models_metric[f'model_{i}']['accuracy'] += (
                            pass_key == predict_pass_key)
                except Exception:
                    pass

                print((num_tokens, pass_key, response,
                       models_metric[f'model_{i}']['accuracy']))

        for i in range(len(models_path)):
            results[f'model_{i}']['num_tokens'].append(
                int(models_metric[f'model_{i}']['num_tokens'] / num_tests))
            results[f'model_{i}']["accuracy"].append(
                models_metric[f'model_{i}']['accuracy'] / num_tests)
        print(results)

    return results


if __name__ == '__main__':
    # models_path = ["llama2-7b-chat", ]
    models_path = ["llama-3b", ]

    results = evaluate_model(models_path)
    print(results)


