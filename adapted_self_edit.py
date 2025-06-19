import os
import re
import json
import glob
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Optional
from datetime import datetime
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
from builtins import input
import argparse
import google.generativeai as genai # Added for Gemini

# Define a sample ARC-like dataset
sample_arc_data = {
    "task_1_inversion": {
        "train": [
            {"input": [[0,1],[1,0]], "output": [[1,0],[0,1]]},
            {"input": [[1,1],[0,0]], "output": [[0,0],[1,1]]}
        ],
        "test": [ # ARC test sets usually only provide "input". Solution is used for eval.
            {"input": [[0,0,1],[0,1,0],[1,0,0]], "output": [[1,1,0],[1,0,1],[0,1,1]]}
        ]
    },
    "task_2_fill_diagonal": {
        "train": [
            {"input": [[0,0,0],[0,0,0],[0,0,0]], "output": [[1,0,0],[0,1,0],[0,0,1]]},
            {"input": [[2,0,0],[0,2,0],[0,0,2]], "output": [[1,0,0],[0,1,0],[0,0,1]]} # Output is fixed for this simple example
        ],
        "test": [
            {"input": [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], "output": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
        ]
    }
}


from peft import LoraConfig
import arclib
from arclib.arc import Example, Task
from arclib.arc import (
    make_submission,
    read_tasks_from_single_file,
    to_list,
    to_tuple,
)
from arclib.representers import (
    CompositeRepresenter,
    ConnectedComponentRepresenter,
    DelimitedGridRepresenter,
    DiffExampleRepresenter,
    GridRepresenter,
    ImageTaskRepresenter,
    PythonListGridRepresenter,
    TaskRepresenter,
    TextTaskRepresenter,
    TextExampleRepresenter,
    WordGridRepresenter,
)
from arclib.messagers import GPTTextMessageRepresenterForBarc, GPTTextMessageRepresenterV2
from arclib.update_model import TTT
from inference.preprocess import get_preprocessed_tasks_single

from arclib.voting import vote
from arclib.eval import evaluate
from inference.engine_vllm import get_sampling_params, initialize_engine, process_requests
from inference.preprocess import get_preprocessed_tasks

import itertools
from typing import List

import numpy as np

from arclib.arc import Task
from arclib.augmenters import (
    Augmenter,
    Chain,
    Concat,
    Flip,
    IdentityAugmenter,
    IncreaseHeight,
    IncreaseResolution,
    IncreaseWidth,
    PermuteColors,
    PermuteExamples,
    RandomTranslateXY,
    Reflect,
    Repeat,
    Rotate,
    Transpose,
)
from arclib.messagers import MessageRepresenter

from vllm import LLM, SamplingParams
from peft import PeftModel # Added for LoRA inference

from utils.prompts import self_edit_prompt, system_message # Assuming this path is correct or will be adjusted


def run_inference_with_lora(
    base_model_name: str,
    lora_adapter_path: str,
    tokenizer_name: str,
    test_input_prompt: str,
    device: str = "cuda"
) -> str:
    """
    Runs inference using a base model with a LoRA adapter applied.
    """
    print(f"\n--- Running Inference ---")
    print(f"Base Model: {base_model_name}")
    print(f"LoRA Adapter: {lora_adapter_path}")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Device: {device}")
    print(f"Input Prompt:\n{test_input_prompt}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16, # Consistent with TTT
            # device_map="auto" # device_map="auto" might conflict with .to(device) later, manage manually
        ).to(device)

        print("Loading PeftModel...")
        model = PeftModel.from_pretrained(base_model, lora_adapter_path).to(device)
        model.eval() # Set to evaluation mode

        inputs = tokenizer(test_input_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        print("Generating output with LoRA model...")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.pad_token_id)

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean up to free memory
        del model
        del base_model
        del tokenizer
        torch.cuda.empty_cache()

        print(f"Decoded output: {decoded_output}")
        print(f"--- Inference Complete ---")
        return decoded_output
    except Exception as e:
        print(f"Error during LoRA inference: {e}")
        # Clean up attempt on error
        if 'model' in locals(): del model
        if 'base_model' in locals(): del base_model
        if 'tokenizer' in locals(): del tokenizer
        torch.cuda.empty_cache()
        return f"Error during inference: {e}"

def run_inference_base_model(
    base_model_name: str,
    tokenizer_name: str,
    test_input_prompt: str,
    device: str = "cuda"
) -> str:
    """
    Runs inference using only the base model.
    """
    print(f"\n--- Running Base Model Inference ---")
    print(f"Base Model: {base_model_name}")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"Device: {device}")
    print(f"Input Prompt:\n{test_input_prompt}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16, # Consistent with TTT
        ).to(device)
        model.eval()

        inputs = tokenizer(test_input_prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        print("Generating output with base model...")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, pad_token_id=tokenizer.pad_token_id)

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        del model
        del tokenizer
        torch.cuda.empty_cache()

        print(f"Decoded output: {decoded_output}")
        print(f"--- Base Model Inference Complete ---")
        return decoded_output
    except Exception as e:
        print(f"Error during base model inference: {e}")
        if 'model' in locals(): del model
        if 'tokenizer' in locals(): del tokenizer
        torch.cuda.empty_cache()
        return f"Error during base model inference: {e}"


def generate_config_with_gemini(prompt_text: str, gemini_api_key: str, gemini_model_name: str = "gemini-1.5-flash") -> dict:
    """
    Generates a configuration using the Gemini API.
    """
    try:
        # Configure the genai client if not already configured
        # In a real scenario, API key configuration might happen once globally
        if not genai.get_model(f"models/{gemini_model_name}"): # Basic check, configure might be better
             genai.configure(api_key=gemini_api_key)

        model = genai.GenerativeModel(gemini_model_name)
        response = model.generate_content(prompt_text)

        # Attempt to parse the response text as JSON
        # The response object might have the text in response.text or parts[0].text
        # depending on the Gemini library version and response structure.
        # Adjust as per actual Gemini client behavior.
        if hasattr(response, 'text') and response.text:
            response_text = response.text
        elif response.parts and response.parts[0].text:
            response_text = response.parts[0].text
        else:
            print("Error: Could not extract text from Gemini response.")
            # Fallback to a safe default or raise an error
            return {
                "data_generation": {
                    "use_basic_augmentations": False, "use_size_augmentations": False,
                    "use_chain_augmentations": False, "use_repeat_augmentations": False
                },
                "training": {"strategy": "train_using_output_tokens", "learning_rate": 1e-5, "num_train_epochs": 1}
            }

        # Clean the response text if it's wrapped in markdown
        cleaned_response_text = re.sub(r"```json\n(.*?)\n```", r"\1", response_text, flags=re.DOTALL)

        config = json.loads(cleaned_response_text)
        return config
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Gemini response: {e}")
        print(f"Problematic response text: {response_text}") # Log the problematic text
        # Fallback to a safe default config or raise an error
        return {
            "data_generation": {
                "use_basic_augmentations": False, "use_size_augmentations": False,
                "use_chain_augmentations": False, "use_repeat_augmentations": False
            },
            "training": {"strategy": "train_using_output_tokens", "learning_rate": 1e-5, "num_train_epochs": 1}
        }
    except Exception as e:
        print(f"An unexpected error occurred with Gemini API: {e}")
        # Fallback to a safe default config or raise an error
        return {
            "data_generation": {
                "use_basic_augmentations": False, "use_size_augmentations": False,
                "use_chain_augmentations": False, "use_repeat_augmentations": False
            },
            "training": {"strategy": "train_using_output_tokens", "learning_rate": 1e-5, "num_train_epochs": 1}
        }


def mode_array(array_list):
    """Return the most common array from a list of arrays."""
    tuple_shape_list = [(tuple(arr.flatten()), arr.shape) for arr in array_list]
    most_common_tuple_shape, _ = Counter(tuple_shape_list).most_common(1)[0]
    most_common_tuple, original_shape = most_common_tuple_shape
    mode_arr = np.array(most_common_tuple).reshape(original_shape)
    return mode_arr


def read_tasks_from_folder(task_folder: str, test: bool = False) -> List[Task]:
    """Read tasks from a folder of JSON files."""
    all_tasks = []
    for file in glob.glob(f"{task_folder}/*.json"):
        basename = os.path.basename(file)
        idx = basename.replace(".json", "")
        tasks = read_tasks_from_file(file, test=test)
        for i, task in enumerate(tasks):
            task.name = idx + "-" + str(i)
        all_tasks += tasks
    return all_tasks


def read_tasks_from_single_file(
    challenge_file: str, test: bool = False, solution_file: Optional[str] = None
) -> List[Task]:
    """Read tasks from a single JSON file with optional solutions."""
    with open(challenge_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if solution_file is not None:
        test = False
        with open(solution_file, "r", encoding="utf-8") as handle:
            solutions = json.load(handle)
            for key, value in solutions.items():
                for idx, solution in enumerate(value):
                    data[key]["test"][idx]["output"] = solution

    all_tasks = []
    for task_name, subtasks in data.items():
        parsed_tasks = Task.read_tasks_from_dict(subtasks, test=test)
        for i, task in enumerate(parsed_tasks):
            task.name = task_name + "-" + str(i)
            all_tasks.append(task)

    return all_tasks


def read_tasks_from_file(task_file: str, test: bool = False) -> List[Task]:
    """Read tasks from a JSON file."""
    with open(task_file, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return Task.read_tasks_from_dict(data, test=test)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


def get_augmenters(
    include_basic: bool = True,
    include_size: bool = True,
    include_chain: bool = True,
    include_repeat: bool = True,
    include_concat: bool = False,
) -> List[Augmenter]:
    basic_augmenters_to_apply = (
        [
            Rotate(90),
            Rotate(270),
            Rotate(180),
            Flip(0),
            Flip(1),
            Reflect(0, reverse=True),
            Reflect(1, reverse=True),
            Reflect(0, reverse=False),
            Reflect(1, reverse=False),
            RandomTranslateXY(),
            Transpose(),
        ]
        if include_basic
        else []
    )

    size_augmenters_to_apply = (
        [
            IncreaseResolution(2),
            IncreaseHeight(2),
            IncreaseWidth(2),
        ]
        if include_size
        else []
    )

    concat_augmenters_to_apply = (
        [
            Concat((IdentityAugmenter(), Rotate(180)), axis=0),
            Concat((IdentityAugmenter(), Rotate(180)), axis=1),
        ]
        if include_concat
        else []
    )

    chain_augmenters_to_apply = (
        [
            Chain([Rotate(90), IncreaseResolution(2)]),
            Chain([Rotate(270), IncreaseResolution(2)]),
            Chain([Rotate(180), IncreaseResolution(2)]),
            Chain([Flip(0), IncreaseResolution(2)]),
            Chain([Flip(1), IncreaseResolution(2)]),
            Chain([Transpose(), IncreaseResolution(2)]),
        ]
        if include_chain
        else []
    )

    repeat_augmenters_to_apply = (
        [
            Repeat(0, 2),
            Repeat(1, 2),
            Repeat(2, 2),
        ]
        if include_repeat
        else []
    )

    augmenters_to_apply = (
        basic_augmenters_to_apply
        + size_augmenters_to_apply
        + concat_augmenters_to_apply
        + chain_augmenters_to_apply
        + repeat_augmenters_to_apply
    )

    #print("Augmenters to apply: ", augmenters_to_apply, "len: ", len(augmenters_to_apply))
    return augmenters_to_apply

def _tokenize_and_process(text: str, tokenizer):
    """
    Tokenize text and set up labels for instruction fine-tuning.
    Specifically looks for assistant header token sequence to identify the response part.
    """
    # Tokenize the entire string
    outputs = tokenizer(
        text,
        truncation=True,
    )
    input_ids = outputs["input_ids"]

    # Find the special sequence: token 128007 followed immediately by 271
    # This is "<|start_header_id|>assistant<|end_header_id|>"
    special_indexes = []
    for i in range(len(input_ids) - 1):
        if input_ids[i] == 128007 and input_ids[i + 1] == 271:
            special_indexes.append(i + 1)  # include the 271 token in the conditioning


    special_index = special_indexes[2] if len(special_indexes) == 4 else None
    # If special sequence not found, handle the error
    if special_index is None:
        print("Warning: Special sequence not found, using fallback strategy")
        # Fallback: set labels for the last 20% of tokens
        assert False

    # Create labels: we want the model to predict tokens after the special sequence
    labels = input_ids.copy()
    for j in range(special_index + 1):
        labels[j] = -100

    outputs["labels"] = labels
    return outputs

def format_and_filter(formatter, tokenizer, task, train_on_input: False):
    task = formatter.encode(task)
    data = {"input": task[0], "output": task[1]}
    task_text = tokenizer.apply_chat_template(task[0] + [task[1]], tokenize=False, add_generation_prompt=True)
    #messages = arc_to_messages(data, train_on_input=False)
    outputs = _tokenize_and_process(task_text, tokenizer)
    data["total_tokens"] = len(outputs["input_ids"])
    data["full_text"] = task_text
    return data


def get_test_time_train_data(
    original_task: Task, augmenters: List[Augmenter], n: int = 1, permute_n: int = 1, seed: int = 0
) -> List[Task]:
    rng = np.random.RandomState(seed)
    train_examples = original_task.train_examples.copy()
    initial_tasks = []
    N = len(train_examples)
    for i in range(len(train_examples)):
        examples = train_examples.copy()
        indices = set(range(N)) - {i}
        # we already remove i, so we need to remove n-1 more
        combs = list(itertools.combinations(indices, n - 1))
        combs = [indices - set(comb) for comb in combs]
        for comb in combs:
            initial_tasks.append(
                Task(name="", train_examples=[examples[j] for j in comb], test_example=examples[i])
            )

    augmented_tasks = []
    for augmenter in augmenters:
        for task in initial_tasks:
            task = augmenter.apply_to_task(task, to_input=True, to_output=True, rng=rng)
            # some augmentations increase shapes
            if not (task.max_height() <= 30 and task.max_width() <= 30):
                continue
            augmented_tasks.append(task)

    augmented_tasks = list(set(augmented_tasks + initial_tasks))

    color_and_permute_augmented_tasks = []

    for _ in range(permute_n):
        for task in augmented_tasks:
            if len(augmenters) != 0:
                new_task = PermuteColors().apply_to_task(task, to_input=True, to_output=True, rng=rng)
            else:
                new_task = task
            new_task = PermuteExamples().apply_to_task(
                new_task, rng=rng, to_input=True, to_output=True
            )
            color_and_permute_augmented_tasks.append(new_task)

    augmented_tasks = color_and_permute_augmented_tasks + augmented_tasks

    augmented_tasks = list(set(augmented_tasks))

    return augmented_tasks


def get_formatted_data(
    task: Task,
    augmenters: List[Augmenter],
    formatter: MessageRepresenter,
    tokenizer,
    leave_n: int = 1,
    permute_n: int = 1,
    seed: int = 0,
    max_tokens: int = 8192,
):

    train_data = get_test_time_train_data(
        task, augmenters, n=leave_n, permute_n=permute_n, seed=seed
    )

    formatted_data = []
    for task in train_data:
        formatted = format_and_filter(formatter, tokenizer, task, train_on_input=False)
        if formatted["total_tokens"] < max_tokens:
            formatted_data.append(formatted)

    return formatted_data


def process_task(
    task: Task,
    augmenters: List[Augmenter],
    formatter: MessageRepresenter,
    tokenizer,
    leave_n: List[int],
    permute_n: int = 1,
    Nmax: int = 250,
    seed: int = 0,
):
    rng = np.random.RandomState(seed)

    train = []
    # Generate training data for each n in leave_n
    for n in leave_n:
        leave_n_train_data = get_formatted_data(
            task, augmenters, formatter, tokenizer, leave_n=n, permute_n=permute_n, seed=seed
        )
        train.extend(leave_n_train_data)

    # Shuffle and limit the total number of examples if needed
    if len(train) > Nmax:
        rng.shuffle(train)
        train = train[:Nmax]

    return train

def get_prompt(task: Task, system_message: str, self_edit_prompt_text: str): # Renamed self_edit_prompt to self_edit_prompt_text
    train_examples = task.serialize()['train']
    formatted_examples = ""

    for example in train_examples:
        # Format input grid
        input_grid = example['input']
        input_str = "Input:\n"
        for row in input_grid:
            input_str += " ".join(map(str, row)) + "\n"

        # Format output grid
        output_grid = example['output']
        output_str = "\nOutput:\n"
        for row in output_grid:
            output_str += " ".join(map(str, row)) + "\n"

        # Combine with separator
        formatted_examples += input_str + output_str + "\n"

    user_message = formatted_examples
    user_message = user_message + "------\n\n" + self_edit_prompt_text # Use the renamed variable
    # The prompt format for Gemini might be simpler, just the user_message part.
    # Or if using a specific chat structure, it would be different.
    # For now, returning the user_message directly as the prompt for Gemini.
    # The original prompt with system messages was for the vLLM model.
    # prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    # return prompt
    # For Gemini, the prompt structure is simpler. We'll pass the combined user message.
    # The `self_edit_prompt` (from prompts.py) already contains instructions for the LLM.
    # The `system_message` might not be directly applicable or used in the same way with Gemini's API.
    # We are passing the `self_edit_prompt` content (which is `self_edit_prompt_text` here)
    # combined with task examples to `generate_config_with_gemini`.
    # The `generate_config_with_gemini` function will then send this to Gemini.
    return user_message # This will be the text_prompt for Gemini

def main(experiment_name, skip_repeated_configs, challenge_file, solution_file, model_name, n_tasks, n_self_edits_per_task, gemini_api_key: str): # Added gemini_api_key
    # lora config
    lora_config = LoraConfig(
        r=128,
        lora_alpha=16,
        lora_dropout=0.00,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear" # Updated based on PEFT docs for QLoRA-style training
    )

    # training config
    batch_size = 2
    gradient_accumulation_steps = 1
    lr_scheduler_type = "cosine"

    standard_formatter = TextTaskRepresenter(
        example_representer=TextExampleRepresenter(
            io_sep=" -> ",
            input_header="",
            output_header="",
            output_footer="#",
            grid_representer=PythonListGridRepresenter(),
        )
    )

    representer = GPTTextMessageRepresenterV2(task_representer=standard_formatter)

    # Load tasks
    # tasks = read_tasks_from_single_file(
    #     challenge_file=challenge_file,
    #     solution_file=solution_file
    # )

    print("Loading tasks from embedded sample_arc_data...")
    tasks = []
    for task_name, task_data in sample_arc_data.items():
        # Task.read_tasks_from_dict expects a dict of subtasks (e.g. {'0': task_data_for_subtask_0})
        # For simplicity, we'll consider each entry in sample_arc_data as a single subtask.
        # The 'test' key in ARC's JSON typically holds a list of test examples.
        # Our sample_arc_data has "test": [example_dict]
        # Task.read_tasks_from_dict can handle this structure if task_data is passed directly.

        # We need to ensure the structure matches what Task.read_tasks_from_dict expects or adapt.
        # Task.read_tasks_from_dict(subtasks_dict, test=is_test_set)
        # 'subtasks_dict' is like: {"0": {"train": [...], "test": [...]}, "1": ...}
        # Our `task_data` is already in the format of a single subtask.
        # So we can wrap it: temp_subtasks_dict = {"0": task_data}

        # The `test` flag in read_tasks_from_dict affects whether it expects 'output' in test examples.
        # If test=True, it doesn't strictly need 'output'. If test=False, it can use it.
        # Since our embedded data has 'output' for test examples, we can use test=False.
        parsed_task_list = Task.read_tasks_from_dict({"": task_data}, test=False) # Pass task_data for a single unnamed subtask
        for i, parsed_task in enumerate(parsed_task_list):
            parsed_task.name = task_name + (f"-{i}" if len(parsed_task_list) > 1 else "") # Assign name
            tasks.append(parsed_task)

    if not tasks:
        print("Warning: No tasks were loaded from sample_arc_data. Check data structure and parsing logic.")
    else:
        print(f"Loaded {len(tasks)} tasks from embedded data.")


    # Setup tokenizer (still needed for TTT and data processing)
    tokenizer = AutoTokenizer.from_pretrained(model_name) # model_name is now Qwen by default

    # Phase 1: Generate configs using Gemini
    print("Phase 1: Generating configs using Gemini...")
    # self_edit_model = LLM(model=model_name) # Original line, commented out
    # sampling_params = SamplingParams( # Original sampling params, may not be needed for Gemini
    #     max_tokens=128,
    #     temperature=0.8,
    # )

    # Dictionary to store explored configs per task
    explored_configs = {}
    task_configs = {}  # Store full configs for each task

    # Ensure GEMINI_API_KEY is available
    if not gemini_api_key:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not provided as argument or environment variable.")

    # The self_edit_prompt is imported from utils.prompts
    # The system_message is also imported but might be less relevant for direct Gemini call

    for i in range(n_tasks):
        task = tasks[i]
        # The get_prompt function was modified to return a simpler text prompt for Gemini
        # It combines task examples with the self_edit_prompt instructions.
        prompt_for_gemini = get_prompt(task, system_message, self_edit_prompt)

        base_task_name = task.name
        if base_task_name.endswith("-0"):
            base_task_name = base_task_name[:-2]
        if base_task_name.endswith("-1"):
            continue

        if base_task_name not in explored_configs:
            explored_configs[base_task_name] = set()
            task_configs[base_task_name] = []

        while len(task_configs[base_task_name]) < n_self_edits_per_task:
            # response = self_edit_model.generate(prompt, sampling_params=sampling_params) # Original call
            # New call to Gemini
            # The prompt_for_gemini already includes the instructions from self_edit_prompt
            config = generate_config_with_gemini(prompt_text=prompt_for_gemini, gemini_api_key=gemini_api_key)

            # The response from Gemini is already the parsed JSON (or a default if error)
            # No need for response[0].outputs[0].text here

            # Convert config to a hashable format (tuple of tuples) for storing in set
            # Ensure config is not None or the default error dict if you want to skip them
            if config.get("training", {}).get("num_train_epochs") == 0 and config.get("training", {}).get("learning_rate") == 0 : # Skip default error configs
                 print(f"Skipping default/error config for task {base_task_name}")
                 continue # Or break, depending on desired behavior

            config_key = (
                ("data_generation", tuple(sorted(config.get("data_generation", {}).items()))),
                ("training", tuple(sorted(config.get("training", {}).items())))
            )

            if skip_repeated_configs and config_key in explored_configs[base_task_name]:
                print(f"Skipping already explored config for task {base_task_name}")
                continue

            explored_configs[base_task_name].add(config_key)
            task_configs[base_task_name].append({
                "config": config,
                "prompt": prompt_for_gemini, # Store the prompt sent to Gemini
                "response": json.dumps(config), # Store the JSON string of the config received
                # "token_ids": None # Gemini API doesn't directly give token IDs like vLLM
            })
            print(f"New config for task {base_task_name} from Gemini:", config)

    # del self_edit_model # No longer using self_edit_model (vLLM) for config gen
    print("Phase 1 complete.")

    # Phase 2: Train models using generated configs
    print("\nPhase 2: Training models using generated configs...")

    # setup ttt
    ttt = TTT(
        model_name=model_name, # This is the Qwen model for fine-tuning
        lora_config=lora_config
    )

    final_configs_and_indices = {}
    for base_task_name, configs_data_list in task_configs.items(): # Renamed configs to configs_data_list
        task = next(t for t in tasks if t.name.startswith(base_task_name))
        task_ttt = 0
        curr_task_configs = {}
        for config_data in configs_data_list: # Iterate through the list of stored data
            config = config_data["config"]
            try:
                augmenters_to_apply = get_augmenters(
                    include_basic=config.get("data_generation", {}).get("use_basic_augmentations", False),
                    include_size=config.get("data_generation", {}).get("use_size_augmentations", False),
                    include_chain=config.get("data_generation", {}).get("use_chain_augmentations", False),
                    include_repeat=config.get("data_generation", {}).get("use_repeat_augmentations", False)
                )
            except Exception as e:
                print(f"Error getting augmenters for task {base_task_name}: {e}")
                augmenters_to_apply = get_augmenters( # Default to no augmenters
                    include_basic=False, include_size=False,
                    include_chain=False, include_repeat=False
                )
                if "training" not in config: config["training"] = {}
                config["training"]["num_train_epochs"] = 0 # Avoid training with bad aug config

            train_data = process_task(
                task=task,
                augmenters=augmenters_to_apply,
                formatter=representer,
                tokenizer=tokenizer,
                leave_n=[1,2],
                permute_n=1,
                Nmax=250,
                seed=0
            )

            if len(train_data) == 0:
                print(f"No training data generated for task {base_task_name} with config {config_data.get('config_id', task_ttt)}. Skipping training for this config.")
                continue

            task_text_list = [data["full_text"] for data in train_data]

            current_training_config = config.get("training", {})
            if not all(k in current_training_config for k in ["strategy", "num_train_epochs", "learning_rate"]):
                print(f"Skipping training for task {base_task_name} (config {task_ttt}) due to incomplete training config.")
                current_training_config["num_train_epochs"] = 0 # Ensure it doesn't train

            if current_training_config.get("strategy") not in ["train_using_all_tokens", "train_using_output_tokens"]:
                print(f"Skipping training for task {base_task_name} (config {task_ttt}) due to invalid strategy.")
                current_training_config["num_train_epochs"] = 0 # Ensure it doesn't train

            if current_training_config.get("num_train_epochs", 0) * len(train_data) // batch_size > 375 : # check gradient_accumulation_steps if used
                print(f"Skipping training for task {base_task_name} (config {task_ttt}) because the number of steps is > 375.")
                current_training_config["num_train_epochs"] = 0


            adapter_path = ttt.update_model(
                task_text_list=task_text_list,
                output_dir=f"loras/self-edit/{experiment_name}/{base_task_name}/{task_ttt}",
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=current_training_config.get("learning_rate", 0), # Default to 0 if missing
                num_train_epochs=current_training_config.get("num_train_epochs", 0), # Default to 0 if missing
                lr_scheduler_type=lr_scheduler_type,
                loss_on_all_tokens=current_training_config.get("strategy") == "train_using_all_tokens"
            )

            # --- Inference Step ---
            if adapter_path and os.path.exists(adapter_path): # Check if adapter was successfully created
                print(f"\n--- Running Inference for Task: {task.name}, Adapter: {adapter_path} ---")
                if task.test_example:
                    test_ex = task.test_example # In our setup, test_example is a single Example object

                    # Simple string representation of the input grid
                    input_grid_list = test_ex.input.tolist() # Convert numpy array to list
                    input_grid_str = "Input:\n" + "\n".join([" ".join(map(str, row)) for row in input_grid_list])

                    # Prepare a simplified prompt for the ARC task - this might need refinement
                    # For many models, just the input grid might not be enough context.
                    # They might expect a more conversational or instruction-like prompt.
                    # Example: "Solve the following puzzle. Given the input grid, what is the output grid?\n"
                    # For now, using a direct representation of the input.
                    # The prompt used for config generation (get_prompt) is too complex for direct inference here.
                    # The training prompt format (apply_chat_template) is also specific.
                    # We need a prompt that the model can complete to produce the output grid.
                    # A simple prompt might be:
                    # "Given the input grid:\n[[0,1],[1,0]]\nWhat is the output grid?"
                    # Or just the input grid if the model was trained on such continuations.
                    # Let's use a simple prefix for now.
                    test_input_prompt = f"Solve this ARC puzzle. Input grid:\n{input_grid_str}\nOutput grid:\n"

                    print(f"Formatted Test Input Prompt for Inference:\n{test_input_prompt}")

                    # Inference with base model
                    output_before_lora = run_inference_base_model(
                        base_model_name=model_name, # args.model_name
                        tokenizer_name=model_name, # args.model_name
                        test_input_prompt=test_input_prompt,
                    )
                    print(f"\nOutput from BASE MODEL for {task.name} (Test Example):\n{output_before_lora}")

                    # Inference with LoRA model
                    output_after_lora = run_inference_with_lora(
                        base_model_name=model_name, # args.model_name
                        lora_adapter_path=adapter_path,
                        tokenizer_name=model_name, # args.model_name
                        test_input_prompt=test_input_prompt
                    )
                    print(f"\nOutput from LoRA MODEL for {task.name} (Adapter {task_ttt}):\n{output_after_lora}")
                    print(f"Expected output for reference: \n{test_ex.output.tolist()}")

                else:
                    print(f"Task {task.name} has no test example for inference.")
            else:
                print(f"Skipping inference for {base_task_name}/{task_ttt} as adapter path is invalid or training failed.")
            # --- End Inference Step ---


            curr_task_configs[task_ttt] = config_data
            task_ttt += 1

        final_configs_and_indices[base_task_name] = curr_task_configs

    if 'ttt' in locals() and ttt is not None: # Ensure ttt was initialized
        del ttt
        torch.cuda.empty_cache()


    configs_file = os.path.join(f"loras/self-edit/{experiment_name}", "final_configs_and_indices.json")
    os.makedirs(os.path.dirname(configs_file), exist_ok=True)
    with open(configs_file, "w") as f:
        json.dump(final_configs_and_indices, f, cls=NumpyEncoder) # Added NumpyEncoder for safety

    print("Training complete. Final configs and indices saved to:", configs_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run self-edit training with specified parameters')
    parser.add_argument('--experiment_name', type=str, required=True,
                      help='Name of the experiment')
    parser.add_argument('--skip_repeated_configs', action='store_true',
                      help='Whether to skip repeated configs')
    parser.add_argument('--challenge_file', type=str, required=False, default=None, # Made optional
                      help='Path to the challenge file (optional, uses embedded data if not provided)')
    parser.add_argument('--solution_file', type=str, required=False, default=None, # Made optional
                      help='Path to the solution file (optional, uses embedded data if not provided)')
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen1.5-1.8B', # Changed default and removed required=True
                      help='Name of the model to use for fine-tuning (e.g., Qwen model)')
    parser.add_argument('--n_tasks', type=int, default=len(sample_arc_data), # Default to number of tasks in sample_arc_data
                      help='Number of tasks to process')
    parser.add_argument('--n_self_edits_per_task', type=int, required=True,
                      help='Number of tasks to process')
    parser.add_argument('--n_self_edits_per_task', type=int, required=True,
                      help='Number of self-edits per task')
    parser.add_argument('--gemini_api_key', type=str, default=None, # Added for Gemini API Key
                        help='Gemini API Key. Can also be set via GEMINI_API_KEY environment variable.')

    args = parser.parse_args()

    # Get Gemini API key from arg or environment variable
    api_key = args.gemini_api_key if args.gemini_api_key else os.getenv('GEMINI_API_KEY')
    if not api_key:
        # Try to get from userdata if available (e.g. Kaggle/Colab)
        try:
            from google.colab import userdata
            api_key = userdata.get('GEMINI_API_KEY')
        except ImportError: # Not in Colab/Kaggle, or userdata not available
            pass # api_key remains None
        except Exception as e: # userdata.get() failed
            print(f"Could not retrieve GEMINI_API_KEY from userdata: {e}")
            pass


    if not api_key:
        print("Error: GEMINI_API_KEY not found. Please provide it via --gemini_api_key argument or set it as an environment variable.")
        # exit(1) # Allow running without API key if only using embedded data and not hitting Gemini
        print("Warning: GEMINI_API_KEY not found. Config generation with Gemini will fail.")


    # If challenge_file is not provided, n_tasks should ideally be based on sample_arc_data
    # The default for n_tasks is already set to len(sample_arc_data)
    # However, if the user explicitly provides a different n_tasks, we should respect that,
    # up to the number of available tasks.
    num_available_tasks = len(tasks) if tasks else len(sample_arc_data)
    actual_n_tasks = min(args.n_tasks, num_available_tasks) if args.n_tasks else num_available_tasks
    if args.n_tasks > num_available_tasks:
        print(f"Warning: Requested n_tasks ({args.n_tasks}) is greater than available tasks ({num_available_tasks}). Using {num_available_tasks} tasks.")


    main(
        experiment_name=args.experiment_name,
        skip_repeated_configs=args.skip_repeated_configs,
        challenge_file=args.challenge_file, # Will be None if not provided
        solution_file=args.solution_file, # Will be None if not provided
        model_name=args.model_name, # This is the Qwen model for TTT
        n_tasks=actual_n_tasks, # Use the adjusted number of tasks
        n_self_edits_per_task=args.n_self_edits_per_task,
        gemini_api_key=api_key # Pass the API key to main
    )
