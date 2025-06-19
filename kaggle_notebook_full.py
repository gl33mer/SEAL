# kaggle_notebook_full.py

# <markdown cell>
# # SEAL Framework on Kaggle with Gemini and Qwen
#
# This notebook demonstrates a proof-of-concept implementation of the SEAL (Self-critique and Edit to Learn)
# framework using the Gemini API for generating training configurations and a Qwen model as the learner.
# The goal is to fine-tune the Qwen model using LoRA based on configurations suggested by Gemini
# and observe its performance on a small, embedded ARC-like dataset.
#
# The Self-critique and Edit to Learn (SEAL) framework is designed to iteratively improve a model's performance
# by generating configurations, training with them, and then (conceptually) evaluating or critiquing the outcome
# to inform the next round of configuration generation. In this simplified PoC, we focus on:
# 1. Using Gemini to generate LoRA fine-tuning configurations.
# 2. Fine-tuning a Qwen model using these configurations on ARC-like tasks.
# 3. Basic inference to see the effect of the LoRA fine-tuning.
#
# **Note**: This notebook uses embedded data and simplified versions of some components from the
# original SEAL repository for demonstration purposes.
# ```

# <code cell>
# ### Cell 1: Install Dependencies
# print("Installing necessary libraries...")
# # For this script, we'll just print them as a reminder.
# # In a real Kaggle notebook, uncomment and run these with '!' prefix.
print("Installation commands (run these in a Kaggle code cell with !):")
print("!pip install transformers==4.40.1 peft==0.10.0 torch==2.2.1 accelerate==0.28.0 datasets==2.18.0 google-generativeai==0.5.0 bitsandbytes==0.42.0") # Added bitsandbytes for 8-bit adam
print("Done with installation instructions.")
# ```

# <markdown cell>
# ### Cell 2: Imports and Gemini API Key Setup
#
# This cell handles essential imports and configures the Gemini API key.
# You'll need to have your Gemini API key available in your Kaggle secrets or environment.
# ```

# <code cell>
print("Setting up imports and Gemini API key...")
import os
import re
import json
import glob # Still used by a function in adapted_self_edit, might remove that function later
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Optional, Dict, Any, Callable, Tuple, Union # Combined typings
from datetime import datetime
from collections import Counter
import dataclasses # For dataclasses used in arclib
import sys # For checking google.colab environment

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType as PeftTaskType # Renamed TaskType to avoid conflict
from datasets import Dataset # For TTT class
# from torch.utils.data import DataLoader # Not directly used by TTT's main method, but good for dataset handling
# from builtins import input # Not needed for notebook usually

import google.generativeai as genai
# from google.colab import userdata # Kaggle uses kaggle_secrets for userdata, or environment variables

# Configure Gemini API Key
GEMINI_API_KEY = None
# Try to get from Kaggle secrets if available
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")
    print("Successfully loaded GEMINI_API_KEY from Kaggle secrets.")
except Exception as e:
    print(f"Could not load GEMINI_API_KEY from Kaggle secrets: {e}. Trying environment variable.")
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not GEMINI_API_KEY:
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE" # Fallback
    print("GEMINI_API_KEY not found in Kaggle secrets or environment. Using placeholder.")

if GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("Gemini API key configured successfully.")
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
else:
    print("Please replace 'YOUR_GEMINI_API_KEY_HERE' with your actual key for Gemini functionality.")

print(f"Gemini API Key Loaded (check if it's 'YOUR_GEMINI_API_KEY_HERE'): {GEMINI_API_KEY[:5]}...")
print("Done with imports and API key setup.")
# ```

# <markdown cell>
# ### Cell 3: ARC Library Code (`arclib`)
# This cell contains the necessary classes from the `arclib` directory of the SEAL repository,
# as it's not a pip-installable package.
# We include `Task`, `Example`, `Grid` from `arc.py`, `TTT` from `update_model.py`,
# and relevant representer/messager classes.
# ```

# <code cell>
print("Defining arclib components...")

# --- From few-shot/arclib/arc.py ---
Grid = np.ndarray

def to_tuple(arr):
    return tuple(tuple([int(e) for e in row]) for row in arr)

def to_list(arr):
    return [[int(e) for e in row] for row in arr]

@dataclasses.dataclass
class Example:
    input: Grid
    output: Grid
    cot: Optional[List[Grid]] = None

    def input_size(self) -> int: return self.input.size
    def output_size(self) -> int: return self.output.size
    def size(self) -> int: return max(self.input_size(), self.output_size())
    def __hash__(self) -> int: return hash((self.input.tobytes(), self.output.tobytes()))
    def __repr__(self) -> str: return f"Example(input={self.input}, output={self.output})"
    def serialize(self) -> dict:
        example = {"input": self.input.tolist(), "output": self.output.tolist()}
        if self.cot: example["cot"] = [cot.tolist() for cot in self.cot]
        return example
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Example): return NotImplemented
        return np.array_equal(self.input, other.input) and np.array_equal(self.output, other.output)
    @classmethod
    def deserialize(cls, data: dict, test: bool = False) -> "Example":
        input_arr = np.array(data["input"])
        if test: output_arr = input_arr.copy()
        elif "output" in data: output_arr = np.array(data["output"])
        else: output_arr = input_arr.copy()
        cot_list = [np.array(c) for c in data["cot"]] if "cot" in data else None
        return cls(input_arr, output_arr, cot_list)

@dataclasses.dataclass
class Task:
    test_example: Example
    train_examples: List[Example] = dataclasses.field(default_factory=list)
    name: str = ""

    def size(self) -> int: return max([example.size() for example in self.train_examples]) if self.train_examples else self.test_example.size()
    def max_height(self) -> int:
        max_x = 0
        for ex in self.train_examples + [self.test_example]:
            max_x = max(max_x, ex.input.shape[0], ex.output.shape[0])
        return max_x
    def max_width(self) -> int:
        max_y = 0
        for ex in self.train_examples + [self.test_example]:
            max_y = max(max_y, ex.input.shape[1], ex.output.shape[1])
        return max_y
    def __repr__(self) -> str: return f"Task(name={self.name}, train_len={len(self.train_examples)}, test_in_shape={self.test_example.input.shape})"
    def serialize(self) -> dict:
        return {"train": [train.serialize() for train in self.train_examples], "test": [self.test_example.serialize()], "name": self.name}
    def __hash__(self) -> int: return hash((tuple(self.train_examples), self.test_example, self.name))
    @classmethod
    def deserialize(cls, data: dict, test: bool = False) -> "Task":
        assert len(data["test"]) == 1, "Only one test example is allowed for this simplified Task structure"
        train = [Example.deserialize(ex_data) for ex_data in data["train"]]
        test_ex = Example.deserialize(data["test"][0], test=test)
        return cls(train_examples=train, test_example=test_ex, name=data.get("name", ""))
    @classmethod
    def read_tasks_from_dict(cls, data: dict, test: bool = False) -> List["Task"]: # Simplified for single task definition
        # This was originally designed to parse multiple sub-tasks from a single dict entry.
        # For our embedded data, each key in sample_arc_data is one task.
        # The parsing logic in adapted_self_edit.py handles this by calling Task.deserialize directly.
        # This method can be adapted or bypassed if `Task.deserialize` is used directly.
        # For the current structure of adapted_self_edit.py, this specific method isn't strictly needed
        # as it iterates sample_arc_data and calls deserialize.
        # However, keeping a version of it for completeness or potential future use.
        # The version in adapted_self_edit.py `Task.read_tasks_from_dict({"": task_data}, test=False)`
        # is a good way to use `deserialize`.
        # For simplicity here, we'll assume direct use of Task.deserialize in the main logic.
        # If `adapted_self_edit.py` uses this, it should be:
        # parsed_tasks = []
        # for task_name_key, task_content in data.items(): # if data is the full multi-task dict
        #    # Assuming task_content is {"train": [...], "test": [...]}
        #    parsed_task = cls.deserialize(task_content, test=test)
        #    parsed_task.name = task_name_key
        #    parsed_tasks.append(parsed_task)
        # return parsed_tasks
        # Given current adapted_self_edit.py, this function is not directly called.
        # The logic is: `Task.read_tasks_from_dict({"": task_data}, test=False)` which effectively means
        # `Task.deserialize(task_data, test=False)` after wrapping.
        # So, we can largely rely on `Task.deserialize`.
        # Let's provide the version from arc.py for completeness, though.
        tasks = []
        # data is expected to be like: {"task_id": {"train": [...], "test": [...]}}
        # or for the simplified case in adapted_self_edit.py: {"": {"train": [...], "test": [...]}}
        # This method from arc.py actually expects `data` to be the content of ONE task ID,
        # i.e., data = {"train": [...], "test": [test_ex_1_data, test_ex_2_data]}
        # and it creates multiple Task objects if there are multiple test examples.
        # This is slightly different from how adapted_self_edit.py uses it with embedded data.
        # For embedded data, `task_data` (value of sample_arc_data) has ONE test example.
        # The loop in `adapted_self_edit.py` is `Task.read_tasks_from_dict({"": task_data}...`
        # This means `data` here becomes `{"": task_data}`.
        # The original loop for `data.items()` would then run once with key `""`.
        # And `subtasks` would be `task_data`.
        # Then `Task.deserialize(subtasks...)` is called.
        # This seems to be what happens in `adapted_self_edit.py`.
        # The original `read_tasks_from_single_file` iterates through the outer dict.
        # The `Task.read_tasks_from_dict` in `arc.py` is for parsing a single task's data which might have multiple test cases.
        # Let's use the simplified approach from `adapted_self_edit.py` which means this function is less critical.
        # The critical part is `Task.deserialize`.
        # For this cell, we'll define Task and Example. The loading will be in Cell 5.
        pass # Placeholder as direct Task.deserialize is used more in adapted_self_edit.py

# --- From few-shot/arclib/representers.py ---
# Minimal set of representers needed for adapted_self_edit.py
class GridRepresenter(ABC):
    @abstractmethod
    def encode(self, grid: Grid) -> str: pass
    @abstractmethod
    def decode(self, encoded_str: str, **kwargs) -> Grid: pass

class PythonListGridRepresenter(GridRepresenter):
    def encode(self, grid: Grid) -> str: return str(grid.tolist()) # Ensure it's a list of lists
    def decode(self, encoded_str: str) -> Grid:
        try: return np.array(eval(encoded_str), dtype=np.int8)
        except: return np.array([[]], dtype=np.int8) # Default on error
    def __repr__(self) -> str: return "PythonListGridRepresenter()"

class ExampleRepresenter(ABC):
    grid_representer: GridRepresenter
    @abstractmethod
    def encode(self, example: Example, **kwargs) -> Union[str, Tuple[str, str]]: pass
    @abstractmethod
    def decode(self, encoded: Tuple[str, str], **kwargs) -> Example: pass

class TextExampleRepresenter(ExampleRepresenter):
    def __init__(self, io_sep: str = " -> ", input_header: str = "", output_header: str = "", output_footer="", grid_representer: GridRepresenter = PythonListGridRepresenter()):
        self.io_sep = io_sep
        self.input_header = input_header
        self.output_header = output_header
        self.output_footer = output_footer
        self.grid_representer = grid_representer
    def encode(self, example: Example, **kwargs) -> Tuple[str, str]:
        input_str = self.grid_representer.encode(example.input, **kwargs)
        output_str = self.grid_representer.encode(example.output, **kwargs)
        return (f"{self.input_header}{input_str}{self.io_sep}{self.output_header}", f"{output_str}{self.output_footer}")
    def decode(self, encoded: Tuple[str, str], **kwargs) -> Example:
        input_full_str, output_full_str = encoded
        input_str = input_full_str.replace(self.input_header, "").replace(self.io_sep, "").replace(self.output_header, "").strip()
        output_str = output_full_str.replace(self.output_footer, "").strip()
        input_grid = self.grid_representer.decode(input_str, **kwargs)
        output_grid = self.grid_representer.decode(output_str, **kwargs)
        return Example(input=input_grid, output=output_grid)
    def __repr__(self) -> str: return f"TextExampleRepresenter(...)"

class TaskRepresenter(ABC):
    example_representer: ExampleRepresenter
    @abstractmethod
    def encode(self, task: Task, **kwargs) -> Union[Tuple[str, str, str], str, Tuple[List[Dict[str, str]], Dict[str, str]]]: pass # Adjusted for messagers
    @abstractmethod
    def decode(self, encoded: Tuple[str, str], **kwargs) -> Task: pass

class TextTaskRepresenter(TaskRepresenter):
    def __init__(self, train_header: str = "==TRAIN==\n", train_test_sep: str = "\n\n", test_header: str = "==TEST==\n", example_sep: str = "\n\n", example_representer: ExampleRepresenter = TextExampleRepresenter()):
        self.train_header = train_header
        self.train_test_sep = train_test_sep
        self.test_header = test_header
        self.example_sep = example_sep
        self.example_representer = example_representer
    def encode(self, task: Task, **kwargs) -> Tuple[str, str, str]: # Returns (demonstrations, test_query, test_output_ground_truth)
        trains_str = self.train_header
        for train_ex in task.train_examples:
            query, output = self.example_representer.encode(train_ex, **kwargs)
            trains_str += query + output + self.example_sep
        trains_str = trains_str.strip()

        test_query_part, test_output_part = self.example_representer.encode(task.test_example, **kwargs)
        full_test_query = self.test_header + test_query_part
        return trains_str, full_test_query, test_output_part # demonstrations, test_input_prompt_for_model, test_output_for_eval
    def decode(self, encoded: Tuple[str, str, str], **kwargs) -> Task: raise NotImplementedError
    def __repr__(self) -> str: return f"TextTaskRepresenter(...)"

# --- From few-shot/arclib/messagers.py ---
# Only GPTTextMessageRepresenterV2 is used by adapted_self_edit.py
MESSAGE = Dict[str, Union[str, Dict]]
MESSAGES = List[MESSAGE]

class MessageRepresenter(ABC):
    task_representer: TaskRepresenter
    @abstractmethod
    def encode(self, task: Task, **kwargs) -> Tuple[MESSAGES, MESSAGE]: pass # Returns (input_messages_for_llm, output_message_for_llm_response_eval)

class GPTTextMessageRepresenterV2(MessageRepresenter):
    def __init__(self, prompt: Optional[str] = "Figure out the underlying transformation...", task_representer: TaskRepresenter = TextTaskRepresenter()):
        self.prompt = prompt # This prompt is not directly used in adapted_self_edit.py's get_prompt, but kept for class structure
        self.task_representer = task_representer
    def encode(self, task: Task, **kwargs) -> Tuple[MESSAGES, MESSAGE]: # (input_messages_for_llm, output_message_for_llm_response_eval)
        # This encode is for creating the prompt for the *learner* model (Qwen) during training data prep.
        # It's used by format_and_filter -> tokenizer.apply_chat_template(task[0] + [task[1]]...
        # where task[0] are input messages and task[1] is the output message.
        # The structure should be:
        # messages = [
        #    {"role": "system", "content": system_prompt_for_learner},
        #    {"role": "user", "content": formatted_train_examples + formatted_test_input},
        # ]
        # output_message = {"role": "assistant", "content": formatted_test_output}

        # The adapted_self_edit.py's `format_and_filter` uses `tokenizer.apply_chat_template`
        # which expects a list of message dicts.
        # This `encode` method should produce that list of message dicts for input, and one for output.

        system_prompt_for_learner = self.prompt # Or could be a more specific one for ARC solving
        if hasattr(task, "description") and task.description: # adapted_self_edit.py tasks don't have descriptions
            system_prompt_for_learner += f"\nTask Description: {task.description}"

        # Formatting training examples for the prompt
        formatted_train_examples = ""
        for ex in task.train_examples:
            # Using the TextTaskRepresenter's example encoding logic
            q, o = self.task_representer.example_representer.encode(ex, **kwargs)
            formatted_train_examples += q + o + "\n" # query contains input and output headers from TextExampleRepresenter

        # Formatting the test input
        test_input_q, test_output_o = self.task_representer.example_representer.encode(task.test_example, **kwargs)

        user_content = f"Observe the following examples:\n{formatted_train_examples.strip()}\n\nNow, predict the output for this test input:\n{test_input_q.strip()}"

        input_messages_for_llm: MESSAGES = [
            {"role": "system", "content": system_prompt_for_learner},
            {"role": "user", "content": user_content}
        ]
        output_message_for_llm_response_eval: MESSAGE = {"role": "assistant", "content": test_output_o.strip()}

        return input_messages_for_llm, output_message_for_llm_response_eval
    def __repr__(self) -> str: return f"GPTTextMessageRepresenterV2(...)"


# --- From few-shot/arclib/update_model.py ---
# Augmenter classes are needed if get_augmenters is called with certain flags.
# For simplicity, we can assume no augmenters initially or define a minimal set.
class Augmenter(ABC): @abstractmethod
def apply_to_task(self, task: Task, **kwargs) -> Task: pass
class IdentityAugmenter(Augmenter):
    def apply_to_task(self, task: Task, **kwargs) -> Task: return task
# Define other augmenters if they are actually enabled by default configs from Gemini
# For now, keeping it minimal. The `get_augmenters` function in adapted_self_edit.py
# will return empty lists if specific classes aren't defined and flags are false.
# Minimal dummy augmenters for now if specific ones are not critical for the PoC
class Rotate(Augmenter): def __init__(self, k): self.k = k
def apply_to_task(self, task: Task, **kwargs) -> Task: return task # Dummy
class Flip(Augmenter): def __init__(self, k): self.k = k
def apply_to_task(self, task: Task, **kwargs) -> Task: return task # Dummy
class Reflect(Augmenter): def __init__(self, k, reverse): self.k, self.reverse = k, reverse
def apply_to_task(self, task: Task, **kwargs) -> Task: return task # Dummy
class RandomTranslateXY(Augmenter): def apply_to_task(self, task: Task, **kwargs) -> Task: return task # Dummy
class Transpose(Augmenter): def apply_to_task(self, task: Task, **kwargs) -> Task: return task # Dummy
class IncreaseResolution(Augmenter): def __init__(self, k): self.k = k
def apply_to_task(self, task: Task, **kwargs) -> Task: return task # Dummy
class IncreaseHeight(Augmenter): def __init__(self, k): self.k = k
def apply_to_task(self, task: Task, **kwargs) -> Task: return task # Dummy
class IncreaseWidth(Augmenter): def __init__(self, k): self.k = k
def apply_to_task(self, task: Task, **kwargs) -> Task: return task # Dummy
class Concat(Augmenter): def __init__(self, k, axis): self.k, self.axis = k, axis
def apply_to_task(self, task: Task, **kwargs) -> Task: return task # Dummy
class Chain(Augmenter): def __init__(self, k): self.k = k
def apply_to_task(self, task: Task, **kwargs) -> Task: return task # Dummy
class Repeat(Augmenter): def __init__(self, k, times): self.k, self.times = k, times
def apply_to_task(self, task: Task, **kwargs) -> Task: return task # Dummy
class PermuteColors(Augmenter): def apply_to_task(self, task: Task, **kwargs) -> Task: return task # Dummy
class PermuteExamples(Augmenter): def apply_to_task(self, task: Task, **kwargs) -> Task: return task # Dummy


class TTT:
    def __init__(self, model_name: str, state_dict_path: Optional[str] = None, lora_config: Optional[LoraConfig] = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token

        model_load_args = {"torch_dtype": torch.bfloat16}
        # device_map="auto" can be problematic if we explicitly move model to device later.
        # For single GPU Kaggle, "cuda" is usually fine.
        # if torch.cuda.is_available(): model_load_args["device_map"] = "auto" # Use GPU if available

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_load_args)
        if torch.cuda.is_available(): model = model.to("cuda")


        if state_dict_path is not None:
            state_dict = torch.load(state_dict_path)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded state dict from {state_dict_path}")

        if lora_config:
            self.model = get_peft_model(model, lora_config)
            self.initial_lora_A = {name: param.data.clone().detach() for name, param in self.model.named_parameters() if "lora_A" in name}
        else: # If no lora_config, use the base model directly (though SEAL implies LoRA)
            self.model = model
            self.initial_lora_A = {}


    def update_model(self, task_text_list: List[str], output_dir: str, batch_size: int, gradient_accumulation_steps: int, learning_rate: float, num_train_epochs: int, lr_scheduler_type: str, loss_on_all_tokens: bool) -> str:
        if not self.initial_lora_A: # If not a PEFT model / no LoRA config was given
            print("Warning: TTT.update_model called without LoRA config. Training full model (not recommended).")
        else:
             self.reset_lora()

        training_data = self._tokenize_and_process_for_ttt(task_text_list, loss_on_all_tokens)
        torch.cuda.empty_cache()
        self._train_model(training_data, output_dir, batch_size, gradient_accumulation_steps, learning_rate, num_train_epochs, lr_scheduler_type)

        if self.initial_lora_A: # Save only if it's a PEFT model
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print(f"LoRA adapter saved to {output_dir}")
        else:
            print("Full model was trained. Saving not implemented for full model in TTT within this script.")

        return output_dir if self.initial_lora_A else "" # Return path only if LoRA adapter was saved

    def reset_lora(self):
        if not self.initial_lora_A: return
        for name, param in self.model.named_parameters():
            if "lora_B" in name: param.data.fill_(0.0)
            elif "lora_A" in name and name in self.initial_lora_A: param.data.copy_(self.initial_lora_A[name])

    def _tokenize_and_process_for_ttt(self, text_list: List[str], loss_on_all_tokens: bool): # Renamed from adapted_self_edit's _tokenize_and_process
        outputs = self.tokenizer(text_list, truncation=True, max_length=8192, padding="longest", return_tensors="pt")
        input_ids = outputs["input_ids"]
        labels = input_ids.clone()

        if not loss_on_all_tokens:
            for i in range(input_ids.shape[0]):
                sample_input_ids_list = input_ids[i].tolist()
                special_indexes = [j + 1 for j in range(len(sample_input_ids_list) - 1) if sample_input_ids_list[j] == 128007 and sample_input_ids_list[j+1] == 271] # Qwen specific

                # Logic from update_model.py: use second-to-last occurrence of assistant header
                # This seems to be specific to Qwen's chat template tokenization.
                # <|start_header_id|>assistant<|end_header_id|> -> token IDs 128007, 271
                # The prompt structure usually is: Sys, User, Asst, User, Asst...
                # If we have 4 special_indexes, it means Sys, User1, Asst1(demos), User2(query), Asst2(target)
                # We want to predict Asst2. So the label masking should go up to User2's assistant header.
                # The original code uses `special_indexes[-2]` if `len(special_indices) == 4`.
                # This implies the structure is: Sys | User | Asst (examples) | User (test_input) | Asst (test_output)
                # There are 3 headers before the final assistant response.
                # The _tokenize_and_process in adapted_self_edit.py used `special_indexes[2]` if `len(special_indexes) == 4`.
                # Let's stick to the logic from update_model.py for TTT's internal processing.

                idx_to_mask_until = -1
                if len(special_indexes) >= 2: # Need at least User and Assistant sections
                    # To predict the final assistant response, mask everything up to and including
                    # the header of that final assistant response.
                    # Example: SYS <content> EOS USER <content> EOS ASSISTANT <content> EOS (this is the target)
                    # We need to find the tokens for "ASSISTANT" (the last one)
                    idx_to_mask_until = special_indexes[-1] # Mask up to the last assistant header
                else: # Fallback or if structure is different
                    print(f"Warning: Unexpected number of assistant headers ({len(special_indexes)}) in TTT tokenization. Check chat template and data format.")
                    # Default to masking a large portion if unsure, or handle error
                    idx_to_mask_until = int(len(sample_input_ids_list) * 0.5) # Example fallback

                if idx_to_mask_until != -1:
                    for j_token in range(idx_to_mask_until + 1): # +1 to include the header itself
                        labels[i, j_token] = -100
        outputs["labels"] = labels
        return outputs

    def _train_model(self, training_data: Dict[str, Any], output_dir: str, batch_size: int, gradient_accumulation_steps: int, learning_rate: float, num_train_epochs: int, lr_scheduler_type: str):
        ds = Dataset.from_dict(training_data)
        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            lr_scheduler_type=lr_scheduler_type,
            logging_steps=max(1, int(num_train_epochs * len(ds) / (batch_size * gradient_accumulation_steps * 10))), # Log ~10 times per training
            save_strategy="no", # No intermediate saves by Trainer
            report_to="none",
            bf16=True if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else False,
            fp16=not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) and torch.cuda.is_available(), # FP16 if bf16 not available
            remove_unused_columns=False,
            optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch", # Fused AdamW if available
            warmup_ratio=0.1, # 10% warmup
        )
        trainer = Trainer(model=self.model, args=training_args, train_dataset=ds)
        print(f"Training on {len(ds)} examples for {num_train_epochs} epochs. LR: {learning_rate}. Batch: {batch_size}. Grad Accum: {gradient_accumulation_steps}.")
        trainer.train()
        print("Training complete.")

print("Done defining arclib components.")
# ```

# <markdown cell>
# ### Cell 4: Utility Prompts
# These are the prompts used for interacting with the Gemini model to generate configurations.
# ```

# <code cell>
print("Defining utility prompts...")
# From few-shot/utils/prompts.py
self_edit_prompt = """
You are configuring a model training pipeline by selecting from predefined tools.

You must make two decisions:

1. **Data Generation Tools** — For each of the following, choose true or false:
    - use_basic_augmentations
    - use_size_augmentations
    - use_chain_augmentations
    - use_repeat_augmentations

2. **Training Configuration** — Choose one of:
    - "train_using_all_tokens"
    - "train_using_output_tokens"

Also specify:
    - learning_rate (float between 1e-6 and 1e-3)
    - num_train_epochs (integer between 1 and 5)

### Output Format

Respond with a valid JSON object. Do not include any explanation, markdown, or extra text. Use lowercase `true`/`false` for booleans and ensure correct JSON syntax.

Example output:

{
  "data_generation": {
    "use_basic_augmentations": true,
    "use_size_augmentations": false,
    "use_chain_augmentations": true,
    "use_repeat_augmentations": false
  },
  "training": {
    "strategy": "train_using_output_tokens",
    "learning_rate": 5e-5,
    "num_train_epochs": 3
  }
}
"""

system_message = "You are a helpful assistant that provides the correct output for the given task immediately." # This is for the config generation model (Gemini)

print("Done defining utility prompts.")
# ```

# <markdown cell>
# ### Cell 5: Main SEAL Logic and Execution
# This cell contains the core logic adapted from `adapted_self_edit.py`.
# It includes:
# - The embedded sample dataset.
# - Functions for Gemini interaction (`generate_config_with_gemini`).
# - Data augmentation and processing functions.
# - Inference functions.
# - The main function orchestrating the SEAL process.
# - Configuration parameters.
# ```

# <code cell>
print("Defining and running main SEAL logic...")

# --- Content from adapted_self_edit.py, with modifications ---

# Define a sample ARC-like dataset (already defined in adapted_self_edit.py, will be part of this cell)
sample_arc_data = {
    "task_1_inversion": {
        "train": [{"input": [[0,1],[1,0]], "output": [[1,0],[0,1]]}, {"input": [[1,1],[0,0]], "output": [[0,0],[1,1]]}],
        "test": [{"input": [[0,0,1],[0,1,0],[1,0,0]], "output": [[1,1,0],[1,0,1],[0,1,1]]}]
    },
    "task_2_fill_diagonal": {
        "train": [{"input": [[0,0,0],[0,0,0],[0,0,0]], "output": [[1,0,0],[0,1,0],[0,0,1]]}, {"input": [[2,0,0],[0,2,0],[0,0,2]], "output": [[1,0,0],[0,1,0],[0,0,1]]}],
        "test": [{"input": [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]], "output": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]
    }
}

# generate_config_with_gemini (already defined in adapted_self_edit.py)
def generate_config_with_gemini(prompt_text: str, gemini_api_key_global: str, gemini_model_name_arg: str = "gemini-1.5-flash-latest") -> dict:
    print(f"Generating config with Gemini model: {gemini_model_name_arg}")
    # Ensure genai is configured (might have been done in Cell 2, but good to check/re-init if needed)
    if not genai.API_KEY: # Check if API key is set on the genai module
        if gemini_api_key_global and gemini_api_key_global != "YOUR_GEMINI_API_KEY_HERE":
            genai.configure(api_key=gemini_api_key_global)
            print("Re-configured Gemini API key for generate_config_with_gemini.")
        else:
            print("Error: Gemini API key not available for generate_config_with_gemini.")
            # Return a default error config
            return {"data_generation": {}, "training": {"strategy": "error", "learning_rate": 0, "num_train_epochs": 0}}
    try:
        model = genai.GenerativeModel(gemini_model_name_arg)
        response = model.generate_content(prompt_text)

        response_text_cleaned = ""
        if hasattr(response, 'text') and response.text: response_text_cleaned = response.text
        elif response.parts and hasattr(response.parts[0], 'text') and response.parts[0].text: response_text_cleaned = response.parts[0].text
        else:
            print(f"Error: Could not extract text from Gemini response. Response: {response}")
            return {"data_generation": {}, "training": {"strategy": "error_response_format", "learning_rate": 0, "num_train_epochs": 0}}

        # Clean potential markdown code block
        match = re.search(r"```json\s*(.*?)\s*```", response_text_cleaned, re.DOTALL)
        if match: config_text = match.group(1)
        else: config_text = response_text_cleaned

        print(f"Raw config text from Gemini: {config_text}")
        config = json.loads(config_text)
        return config
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from Gemini response: {e}. Response text: '{config_text}'")
        return {"data_generation": {}, "training": {"strategy": "error_json_decode", "learning_rate": 0, "num_train_epochs": 0}}
    except Exception as e:
        print(f"An unexpected error occurred with Gemini API: {e}")
        return {"data_generation": {}, "training": {"strategy": "error_api_unexpected", "learning_rate": 0, "num_train_epochs": 0}}

# Augmenter functions (from adapted_self_edit.py, ensure Augmenter base class and dummies are in Cell 3)
def get_augmenters(include_basic: bool = True, include_size: bool = True, include_chain: bool = True, include_repeat: bool = True, include_concat: bool = False) -> List[Augmenter]:
    aug_list = []
    if include_basic: aug_list.extend([Rotate(90), Flip(0)]) # Simplified list
    if include_size: aug_list.extend([IncreaseResolution(2)]) # Simplified
    # Add more if their classes are fully defined in Cell 3 and are desired
    return aug_list if aug_list else [IdentityAugmenter()] # Always return at least Identity

# _tokenize_and_process for formatting data for Qwen (learner model)
# This is different from TTT's internal _tokenize_and_process_for_ttt
def _tokenize_and_process_for_learner_data_prep(text: str, tokenizer_for_learner):
    outputs = tokenizer_for_learner(text, truncation=True, max_length=8192) # Qwen specific max_length
    input_ids = outputs["input_ids"]
    labels = input_ids.copy()

    # Logic for masking based on Qwen chat template
    # <|im_start|>system...<|im_end|>
    # <|im_start|>user...<|im_end|>
    # <|im_start|>assistant...<|im_end|>
    # We need to find the start of the final assistant part to mask labels before it.
    # Using token IDs for <|im_start|> and assistant might be fragile.
    # A robust way is to find the last occurrence of the assistant prompt structure.
    # For Qwen, this typically involves specific tokens like `tokenizer.convert_tokens_to_ids("<|im_start|>")` etc.
    # The `apply_chat_template` should format it correctly. We need to find where the *actual* response to predict begins.

    # Find all occurrences of assistant message start
    # This depends heavily on the actual chat template used by tokenizer.apply_chat_template
    # For Qwen, it might be a sequence like `\n<|im_start|>assistant\n`
    # Let's assume the template creates a string where the last assistant part is what we want to predict.
    # A common strategy: mask everything up to the point where the model should start generating.
    # This is often after the final "assistant\n" turn in the prompt.

    # Simpler heuristic: if "assistant\n" is present, mask up to the last one.
    # This is a simplified heuristic. For robust masking, one should inspect token IDs from apply_chat_template with add_generation_prompt=True
    assistant_prompt_indicator = tokenizer_for_learner.apply_chat_template([{"role": "assistant", "content": ""}], tokenize=False, add_generation_prompt=True)
    assistant_prompt_indicator = assistant_prompt_indicator.replace(tokenizer_for_learner.eos_token, "") # remove eos if added by template to empty assistant

    # Convert text to string to find last occurrence
    token_string = tokenizer_for_learner.decode(input_ids)
    last_assistant_start_char_idx = token_string.rfind(assistant_prompt_indicator)

    if last_assistant_start_char_idx != -1:
        # Find where this char index corresponds in tokens
        # This is approximate; precise way is to tokenize the prompt parts separately.
        # Here, we tokenize up to that point and count tokens.
        prompt_part_tokens = tokenizer_for_learner(token_string[:last_assistant_start_char_idx + len(assistant_prompt_indicator)], return_tensors="pt").input_ids.shape[1]
        mask_until_token_idx = prompt_part_tokens -1 # Mask tokens belonging to the prompt part
        for j in range(mask_until_token_idx):
            if j < len(labels): labels[j] = -100
    else: # Fallback: if no clear assistant prompt, predict last part (e.g. 20%)
        print("Warning: Assistant prompt indicator for masking not found in learner data prep. Using fallback masking.")
        mask_until_token_idx = int(len(labels) * 0.8)
        for j in range(mask_until_token_idx):
             if j < len(labels): labels[j] = -100

    outputs["labels"] = labels
    return outputs


# format_and_filter (from adapted_self_edit.py)
def format_and_filter(formatter_obj, tokenizer_for_learner, task_obj, train_on_input=False): # Renamed to avoid conflict
    # formatter_obj is an instance of GPTTextMessageRepresenterV2 (defined in Cell 3)
    # task_obj is an instance of Task (defined in Cell 3)

    # This `encode` is from `GPTTextMessageRepresenterV2`
    # It returns (input_messages_for_llm, output_message_for_llm_response_eval)
    input_messages, output_message = formatter_obj.encode(task_obj)

    # Combine for chat template
    full_chat = input_messages + [output_message]

    # `apply_chat_template` formats this list of dicts into a single string prompt
    task_text_for_learner = tokenizer_for_learner.apply_chat_template(
        full_chat,
        tokenize=False,
        add_generation_prompt=False # Important: False if manually adding EOS or specific turn, True if model expects it
    )
    # For SFT, typically add_generation_prompt=False, and the training data includes the assistant's full response.

    # The _tokenize_and_process_for_learner_data_prep needs to handle label masking correctly based on this task_text_for_learner
    # It should mask out the system prompt, user prompt, and the prompt part of the assistant's turn.
    tokenized_data = _tokenize_and_process_for_learner_data_prep(task_text_for_learner, tokenizer_for_learner)

    return {
        "input_ids": tokenized_data["input_ids"],
        "attention_mask": tokenized_data["attention_mask"],
        "labels": tokenized_data["labels"],
        "full_text": task_text_for_learner, # For debugging
        "total_tokens": len(tokenized_data["input_ids"])
    }

# get_test_time_train_data (from adapted_self_edit.py) - This creates sub-tasks for meta-learning style training if needed
def get_test_time_train_data(original_task: Task, augmenters: List[Augmenter], n: int = 1, permute_n: int = 1, seed: int = 0) -> List[Task]:
    rng = np.random.RandomState(seed)
    # ... (rest of the function as in adapted_self_edit.py, ensure it uses Task and Example from Cell 3)
    # This function creates new Task objects.
    # For simplicity in PoC, if not using complex augmentation strategies, this could be simplified.
    # For now, assume it's copied if complex augmentation is generated by Gemini.
    # If `get_augmenters` returns only `IdentityAugmenter`, this won't do much beyond permutations.
    train_examples = original_task.train_examples.copy()
    initial_tasks = []
    N_orig_train = len(train_examples)
    if N_orig_train == 0: return [original_task] # Cannot create leave-n-out from no examples

    for i in range(N_orig_train): # Create tasks by holding out one training example as test
        temp_train_examples = train_examples[:i] + train_examples[i+1:]
        if not temp_train_examples: continue # Need at least one train example for the new sub-task

        # The held-out example becomes the test_example for this new sub-task
        # The remaining examples form the train_examples for this new sub-task
        # This is a common setup for few-shot learning style prompts or meta-learning.
        # The original `get_test_time_train_data` had a more complex combination logic.
        # Simplifying for the notebook: use all other train examples to predict one held-out train example.
        new_task_for_ttt = Task(name=f"{original_task.name}_ttt_{i}", train_examples=temp_train_examples, test_example=train_examples[i])
        initial_tasks.append(new_task_for_ttt)

    if not initial_tasks: # If original task had only 1 train example, loop above makes initial_tasks empty
        initial_tasks.append(original_task)


    augmented_tasks = []
    for aug in augmenters:
        for task_to_aug in initial_tasks:
            try:
                aug_task = aug.apply_to_task(task_to_aug) # Assuming apply_to_task is defined for each augmenter
                if not (aug_task.max_height() <= 30 and aug_task.max_width() <= 30): continue # Constraint from original
                augmented_tasks.append(aug_task)
            except Exception as e:
                print(f"Warning: Augmenter {aug} failed for task {task_to_aug.name}: {e}")

    final_tasks = list(set(initial_tasks + augmented_tasks)) # Use set to remove duplicates
    return final_tasks if final_tasks else [original_task] # Ensure at least original task is returned


# get_formatted_data (from adapted_self_edit.py)
def get_formatted_data(task_obj: Task, list_of_augmenters: List[Augmenter], formatter_obj, tokenizer_for_learner, leave_n: int = 1, permute_n: int = 1, seed: int = 0, max_tokens: int = 8192):
    # `task_obj` is the original task. `get_test_time_train_data` will create sub-tasks from it.
    train_data_tasks = get_test_time_train_data(task_obj, list_of_augmenters, n=leave_n, permute_n=permute_n, seed=seed)

    formatted_data_list = []
    for ttt_task in train_data_tasks: # These are the tasks for the Test-Time Training (TTT) learner model
        formatted = format_and_filter(formatter_obj, tokenizer_for_learner, ttt_task)
        if formatted["total_tokens"] < max_tokens:
            formatted_data_list.append(formatted)
    return formatted_data_list

# process_task (from adapted_self_edit.py)
def process_task(task_obj: Task, list_of_augmenters: List[Augmenter], formatter_obj, tokenizer_for_learner, leave_n_list: List[int], permute_n: int = 1, Nmax: int = 250, seed: int = 0):
    rng = np.random.RandomState(seed)
    all_formatted_train_data = []
    for n_val in leave_n_list:
        data_for_n = get_formatted_data(task_obj, list_of_augmenters, formatter_obj, tokenizer_for_learner, leave_n=n_val, permute_n=permute_n, seed=seed)
        all_formatted_train_data.extend(data_for_n)

    if len(all_formatted_train_data) > Nmax:
        rng.shuffle(all_formatted_train_data)
        all_formatted_train_data = all_formatted_train_data[:Nmax]
    return all_formatted_train_data


# Inference functions (run_inference_base_model, run_inference_with_lora from adapted_self_edit.py)
# These are already defined in adapted_self_edit.py and will be part of this cell.
# Ensure PeftModel, AutoModelForCausalLM, AutoTokenizer are imported in Cell 2.
def run_inference_with_lora(base_model_name_arg: str, lora_adapter_path_arg: str, tokenizer_name_arg: str, test_input_prompt_arg: str, device_arg: str = "cuda") -> str:
    print(f"\n--- Running LoRA Inference ---")
    print(f"Base: {base_model_name_arg}, Adapter: {lora_adapter_path_arg}")
    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_name_arg)
        if tok.pad_token is None: tok.pad_token = tok.eos_token

        base_m = AutoModelForCausalLM.from_pretrained(base_model_name_arg, torch_dtype=torch.bfloat16).to(device_arg)
        lora_m = PeftModel.from_pretrained(base_m, lora_adapter_path_arg).to(device_arg)
        lora_m.eval()

        inputs = tok(test_input_prompt_arg, return_tensors="pt", truncation=True, max_length=512).to(device_arg)
        with torch.no_grad(): outputs = lora_m.generate(**inputs, max_new_tokens=150, pad_token_id=tok.pad_token_id, temperature=0.7, do_sample=True) # Added sampling params
        decoded = tok.decode(outputs[0], skip_special_tokens=True)

        del lora_m, base_m, tok; torch.cuda.empty_cache()
        print(f"LoRA Decoded: {decoded[-200:]}...") # Print last part
        return decoded
    except Exception as e:
        print(f"Error LoRA Inf: {e}"); del locals().get('lora_m', None), locals().get('base_m', None), locals().get('tok', None); torch.cuda.empty_cache(); return f"Err: {e}"

def run_inference_base_model(base_model_name_arg: str, tokenizer_name_arg: str, test_input_prompt_arg: str, device_arg: str = "cuda") -> str:
    print(f"\n--- Running Base Model Inference ---")
    print(f"Base: {base_model_name_arg}")
    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_name_arg)
        if tok.pad_token is None: tok.pad_token = tok.eos_token

        m = AutoModelForCausalLM.from_pretrained(base_model_name_arg, torch_dtype=torch.bfloat16).to(device_arg)
        m.eval()

        inputs = tok(test_input_prompt_arg, return_tensors="pt", truncation=True, max_length=512).to(device_arg)
        with torch.no_grad(): outputs = m.generate(**inputs, max_new_tokens=150, pad_token_id=tok.pad_token_id, temperature=0.7, do_sample=True)
        decoded = tok.decode(outputs[0], skip_special_tokens=True)

        del m, tok; torch.cuda.empty_cache()
        print(f"Base Decoded: {decoded[-200:]}...") # Print last part
        return decoded
    except Exception as e:
        print(f"Error Base Inf: {e}"); del locals().get('m', None), locals().get('tok', None); torch.cuda.empty_cache(); return f"Err: {e}"


# get_prompt for Gemini (from adapted_self_edit.py)
def get_prompt_for_gemini(task_obj: Task, system_msg: str, self_edit_instr: str): # Renamed to avoid conflict
    # This function prepares the prompt FOR GEMINI (the config generator)
    train_examples_str = ""
    for ex in task_obj.train_examples:
        # Using simple list representation for Gemini prompt
        input_grid_str = str(ex.input.tolist())
        output_grid_str = str(ex.output.tolist())
        train_examples_str += f"Input:\n{input_grid_str}\nOutput:\n{output_grid_str}\n\n"

    # The system_message for Gemini is not used here as per adapted_self_edit.py, self_edit_prompt contains all instructions.
    # The prompt for Gemini should be just the task examples and the self_edit_prompt which tells it what to generate.
    user_message = f"Here are examples from an ARC task:\n\n{train_examples_str.strip()}\n\n------\n\n{self_edit_instr}"
    return user_message


# Main function (adapted from adapted_self_edit.py)
def main_seal_logic(exp_name, skip_cfg, base_m_name, num_tasks_to_run, edits_per_task, gemini_m_name, gemini_key):
    # LoraConfig (from adapted_self_edit.py)
    lora_config_obj = LoraConfig(r=32, lora_alpha=16, lora_dropout=0.05, bias="none", task_type=PeftTaskType.CAUSAL_LM, target_modules="all-linear")

    # Training defaults (from adapted_self_edit.py, these can be overridden by Gemini's config)
    default_batch_size = 1 # Keep low for small datasets / large models on single GPU
    default_grad_accum = 2
    default_lr_scheduler = "cosine"

    # Representer for learner model data (from adapted_self_edit.py)
    # Ensure PythonListGridRepresenter, TextExampleRepresenter, TextTaskRepresenter, GPTTextMessageRepresenterV2 are defined in Cell 3
    learner_data_formatter = GPTTextMessageRepresenterV2(
        task_representer=TextTaskRepresenter(
            example_representer=TextExampleRepresenter(grid_representer=PythonListGridRepresenter())
        )
    )

    # Load tasks from embedded sample_arc_data
    print("Loading tasks from embedded sample_arc_data...")
    parsed_tasks_list = []
    for t_name, t_data in sample_arc_data.items():
        # Directly use Task.deserialize as per adapted_self_edit.py's simplified loading for embedded data
        # The 'test=False' allows loading of 'output' fields in test examples from our embedded data.
        try:
            # Each task_data in sample_arc_data is a dict for a single task with 'train' and 'test' keys.
            # Task.deserialize expects data for one task.
            current_task = Task.deserialize(t_data, test=False)
            current_task.name = t_name
            parsed_tasks_list.append(current_task)
        except Exception as e:
            print(f"Error deserializing task {t_name}: {e}")
            continue

    if not parsed_tasks_list:
        print("FATAL: No tasks were loaded. Exiting.")
        return
    print(f"Loaded {len(parsed_tasks_list)} tasks from embedded data: {[t.name for t in parsed_tasks_list]}")

    # Adjust num_tasks_to_run
    actual_num_tasks = min(num_tasks_to_run, len(parsed_tasks_list))
    print(f"Processing {actual_num_tasks} tasks out of {len(parsed_tasks_list)} available.")

    # Tokenizer for learner model (Qwen)
    learner_tokenizer = AutoTokenizer.from_pretrained(base_m_name)
    if learner_tokenizer.pad_token is None: learner_tokenizer.pad_token = learner_tokenizer.eos_token

    # --- Phase 1: Generate Configs with Gemini ---
    print("\n--- Phase 1: Generating Configs with Gemini ---")
    all_task_configs_generated = {} # Stores {base_task_name: [list_of_configs_from_gemini]}

    for i in range(actual_num_tasks):
        current_task_obj = parsed_tasks_list[i]
        print(f"\nProcessing task: {current_task_obj.name} ({i+1}/{actual_num_tasks})")

        # Prepare prompt for Gemini (config generator)
        # self_edit_prompt and system_message are global vars from Cell 4
        prompt_for_gemini_config_gen = get_prompt_for_gemini(current_task_obj, system_message, self_edit_prompt)

        # Use current_task_obj.name as base_task_name directly
        base_task_name_key = current_task_obj.name
        if base_task_name_key not in all_task_configs_generated:
            all_task_configs_generated[base_task_name_key] = []

        explored_config_keys_for_task = set()

        for edit_attempt in range(edits_per_task):
            print(f"Attempting to generate config {edit_attempt+1}/{edits_per_task} for task {base_task_name_key}...")
            # gemini_m_name is like "gemini-1.5-flash-latest"
            generated_config_dict = generate_config_with_gemini(prompt_for_gemini_config_gen, gemini_key, gemini_m_name)

            if not generated_config_dict or generated_config_dict.get("training", {}).get("strategy") == "error":
                print(f"Failed to get valid config from Gemini for task {base_task_name_key}, attempt {edit_attempt+1}. Skipping this attempt.")
                continue

            # Validate config structure (basic check)
            if not ("data_generation" in generated_config_dict and "training" in generated_config_dict and \
                  "strategy" in generated_config_dict["training"] and "learning_rate" in generated_config_dict["training"] and \
                  "num_train_epochs" in generated_config_dict["training"]):
                print(f"Invalid config structure from Gemini: {generated_config_dict}. Skipping.")
                continue

            config_key_tuple = (
                ("data_generation", tuple(sorted(generated_config_dict.get("data_generation", {}).items()))),
                ("training", tuple(sorted(generated_config_dict.get("training", {}).items())))
            )

            if skip_cfg and config_key_tuple in explored_config_keys_for_task:
                print(f"Skipping already explored config for task {base_task_name_key}")
                continue

            explored_config_keys_for_task.add(config_key_tuple)
            all_task_configs_generated[base_task_name_key].append({
                "config": generated_config_dict,
                "prompt_to_gemini": prompt_for_gemini_config_gen, # For debugging
                "gemini_response_json": json.dumps(generated_config_dict) # For debugging
            })
            print(f"New config for task {base_task_name_key} from Gemini: {generated_config_dict}")

    print("\n--- Phase 1 Complete: Config Generation ---")

    # --- Phase 2: Train Learner Models (Qwen with LoRA) ---
    print("\n--- Phase 2: Training Learner Models ---")

    # Initialize TTT instance (defined in Cell 3)
    # The lora_config_obj is passed here.
    # TTT handles model and tokenizer loading internally.
    try:
        learner_trainer_ttt = TTT(model_name=base_m_name, lora_config=lora_config_obj)
    except Exception as e:
        print(f"FATAL: Could not initialize TTT learner model with {base_m_name}. Error: {e}")
        print("Check if model name is correct and you have internet access for model download if it's the first time.")
        return

    final_results_and_paths = {} # To store paths to saved LoRA adapters and inference results

    for task_idx, (base_task_name_key, list_of_configs_for_task) in enumerate(all_task_configs_generated.items()):
        current_task_obj = next((t for t in parsed_tasks_list if t.name == base_task_name_key), None)
        if not current_task_obj:
            print(f"Error: Task object not found for {base_task_name_key}. Skipping.")
            continue

        print(f"\nStarting training for task: {current_task_obj.name} ({task_idx+1}/{len(all_task_configs_generated)})")

        lora_adapter_idx_for_task = 0
        final_results_and_paths[base_task_name_key] = []

        for config_data_item in list_of_configs_for_task:
            current_config_dict = config_data_item["config"]

            # Get augmenters based on Gemini's config
            # Ensure get_augmenters and Augmenter classes are defined (Cell 3 for classes, Cell 5 for get_augmenters)
            try:
                current_augmenters = get_augmenters(
                    include_basic=current_config_dict.get("data_generation", {}).get("use_basic_augmentations", False),
                    include_size=current_config_dict.get("data_generation", {}).get("use_size_augmentations", False),
                    include_chain=current_config_dict.get("data_generation", {}).get("use_chain_augmentations", False),
                    include_repeat=current_config_dict.get("data_generation", {}).get("use_repeat_augmentations", False)
                )
            except Exception as e:
                print(f"Error getting augmenters for task {base_task_name_key}: {e}. Defaulting to no augmenters.")
                current_augmenters = [IdentityAugmenter()]
                current_config_dict["training"]["num_train_epochs"] = 0 # Don't train if aug config is bad

            # Process task to get training data strings for the learner model
            # `process_task` uses `get_formatted_data` which uses `format_and_filter`
            # `format_and_filter` uses `learner_data_formatter` (GPTTextMessageRepresenterV2) and `learner_tokenizer`
            training_instances_for_learner = process_task(
                task_obj=current_task_obj,
                list_of_augmenters=current_augmenters,
                formatter_obj=learner_data_formatter,
                tokenizer_for_learner=learner_tokenizer,
                leave_n_list=[1], # For PoC, just use leave-1-out from training examples
                permute_n=1,
                Nmax=50, # Limit number of training instances for TTT for speed in PoC
                seed=42
            )

            if not training_instances_for_learner:
                print(f"No training data generated for task {base_task_name_key} with config. Skipping this LoRA.")
                continue

            # Extract full_text from each dict for TTT
            task_text_list_for_ttt = [item["full_text"] for item in training_instances_for_learner]

            training_params_from_gemini = current_config_dict.get("training", {})
            num_epochs = training_params_from_gemini.get("num_train_epochs", 1)
            learning_rate = training_params_from_gemini.get("learning_rate", 1e-4)
            strategy = training_params_from_gemini.get("strategy", "train_using_output_tokens")

            if num_epochs == 0 : # Skip if Gemini suggested 0 epochs (e.g. due to bad aug config)
                print(f"Skipping LoRA {lora_adapter_idx_for_task} for {base_task_name_key} as num_train_epochs is 0.")
                continue

            # Define output directory for this specific LoRA adapter
            lora_output_dir = f"loras/{exp_name}/{base_task_name_key}/{lora_adapter_idx_for_task}"
            os.makedirs(lora_output_dir, exist_ok=True)

            print(f"Updating learner model with LoRA adapter: {lora_output_dir}")
            # Call TTT's update_model
            # TTT's update_model will internally call its own _tokenize_and_process_for_ttt
            # and _train_model methods.
            try:
                saved_adapter_path = learner_trainer_ttt.update_model(
                    task_text_list=task_text_list_for_ttt,
                    output_dir=lora_output_dir,
                    batch_size=default_batch_size,
                    gradient_accumulation_steps=default_grad_accum,
                    learning_rate=learning_rate,
                    num_train_epochs=num_epochs,
                    lr_scheduler_type=default_lr_scheduler,
                    loss_on_all_tokens=(strategy == "train_using_all_tokens")
                )
            except Exception as e:
                print(f"Error during TTT update_model for {lora_output_dir}: {e}")
                saved_adapter_path = "" # Indicate failure


            # --- Inference Step (after each LoRA is trained) ---
            output_before_lora_str = "N/A"
            output_after_lora_str = "N/A"

            if saved_adapter_path and os.path.exists(saved_adapter_path):
                print(f"\n--- Running Inference for Task: {current_task_obj.name}, Adapter: {saved_adapter_path} ---")
                if current_task_obj.test_example:
                    test_ex_obj = current_task_obj.test_example
                    input_grid_str_list = test_ex_obj.input.tolist()
                    input_grid_formatted_str = "Input:\n" + "\n".join([" ".join(map(str, r)) for r in input_grid_str_list])
                    # This is a very simple prompt. More complex prompting might be needed.
                    test_input_prompt_for_inf = f"Solve this ARC puzzle. Input grid:\n{input_grid_formatted_str}\nOutput grid:\n"
                    print(f"Formatted Test Input Prompt for Inference:\n{test_input_prompt_for_inf}")

                    # Device for inference
                    inf_device = "cuda" if torch.cuda.is_available() else "cpu"

                    # Base Model Inference (run once per task if not already done, or always for clarity)
                    # To save time, we could do this only once per task, but for PoC it's fine here.
                    output_before_lora_str = run_inference_base_model(
                        base_model_name_arg=base_m_name, tokenizer_name_arg=base_m_name,
                        test_input_prompt_arg=test_input_prompt_for_inf, device_arg=inf_device
                    )
                    print(f"\nOutput from BASE MODEL for {current_task_obj.name} (Test Example):\n{output_before_lora_str}")

                    # LoRA Model Inference
                    output_after_lora_str = run_inference_with_lora(
                        base_model_name_arg=base_m_name, lora_adapter_path_arg=saved_adapter_path,
                        tokenizer_name_arg=base_m_name, test_input_prompt_arg=test_input_prompt_for_inf, device_arg=inf_device
                    )
                    print(f"\nOutput from LoRA MODEL for {current_task_obj.name} (Adapter {lora_adapter_idx_for_task}):\n{output_after_lora_str}")
                    print(f"Expected output for reference: \n{test_ex_obj.output.tolist()}")
                else:
                    print(f"Task {current_task_obj.name} has no test example for inference.")
            else:
                print(f"Skipping inference for {base_task_name_key}/{lora_adapter_idx_for_task} as adapter path is invalid or training failed.")

            final_results_and_paths[base_task_name_key].append({
                "lora_adapter_path": saved_adapter_path,
                "gemini_config": current_config_dict,
                "output_before_lora": output_before_lora_str,
                "output_after_lora": output_after_lora_str,
                "expected_output": current_task_obj.test_example.output.tolist() if current_task_obj.test_example else "N/A"
            })
            lora_adapter_idx_for_task += 1
            torch.cuda.empty_cache() # Clean cache after each LoRA training + inference cycle

    print("\n--- Phase 2 Complete: Learner Model Training & Inference ---")

    # Save final results (paths and inference outputs)
    results_file = os.path.join(f"loras/{exp_name}", "final_experiment_results.json")
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(final_results_and_paths, f, indent=2)
    print(f"\nFull experiment results, LoRA paths, and inference outputs saved to: {results_file}")

    del learner_trainer_ttt # Clean up TTT model
    torch.cuda.empty_cache()
    print("SEAL PoC run finished.")


# --- Configuration for the notebook execution ---
class Args:
    def __init__(self):
        self.experiment_name = "kaggle_seal_poc_v1"
        self.skip_repeated_configs = False
        self.model_name = "Qwen/Qwen1.5-0.5B" # Smaller Qwen model for Kaggle, ensure it's available
        self.n_tasks = len(sample_arc_data) # Process all tasks in the embedded data
        self.n_self_edits_per_task = 1 # Number of configs to generate from Gemini per task
        self.gemini_model_name = "gemini-1.5-flash-latest"
        # GEMINI_API_KEY should be available from Cell 2 setup

args_for_main = Args()

# --- Execution ---
# This block will run when the cell is executed in a Kaggle notebook.
if __name__ == '__main__' or 'ipykernel' in sys.modules: # Check if running in notebook like environment
    print(f"Starting SEAL PoC with experiment name: {args_for_main.experiment_name}")
    print(f"Using base learner model: {args_for_main.model_name}")
    print(f"Using Gemini model for config generation: {args_for_main.gemini_model_name}")
    print(f"Number of tasks to process: {args_for_main.n_tasks}")
    print(f"Number of configs per task: {args_for_main.n_self_edits_per_task}")

    # Ensure GEMINI_API_KEY is correctly passed from Cell 2's global scope
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        print("WARNING: GEMINI_API_KEY is set to placeholder. Gemini calls will fail.")
        print("Please set your Gemini API key in Cell 2.")
    elif not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY is not set. Gemini calls will fail.")
        print("Please set your Gemini API key in Cell 2.")
    else:
        print("GEMINI_API_KEY seems to be set. Proceeding.")

    # Call the main logic function
    # The main function signature is:
    # main_seal_logic(exp_name, skip_cfg, base_m_name, num_tasks_to_run, edits_per_task, gemini_m_name, gemini_key)
    main_seal_logic(
        exp_name=args_for_main.experiment_name,
        skip_cfg=args_for_main.skip_repeated_configs,
        base_m_name=args_for_main.model_name,
        num_tasks_to_run=args_for_main.n_tasks, # This will be len(sample_arc_data)
        edits_per_task=args_for_main.n_self_edits_per_task,
        gemini_m_name=args_for_main.gemini_model_name,
        gemini_key=GEMINI_API_KEY # Pass the globally set API key
    )

print("Done with main SEAL logic cell execution.")
# ```

# <markdown cell>
# ### End of Notebook
#
# The above cells, when run in sequence in a Kaggle notebook, should demonstrate the SEAL proof-of-concept.
#
# **Next Steps & Potential Improvements:**
# - **Error Handling:** More robust error handling, especially around API calls and model loading.
# - **Evaluation:** Implement a proper evaluation metric for ARC tasks (e.g., exact grid match).
# - **Self-Critique Loop:** Add a mechanism for Gemini (or another model) to critique the performance of the learner model based on evaluation results and suggest improvements for the next round of configuration generation. This is the core "critique and edit" part of SEAL.
# - **Richer Dataset:** Use a larger and more diverse set of ARC tasks.
# - **Advanced Augmentation:** Implement the full range of augmenters from `arclib`.
# - **Hyperparameter Tuning:** The fixed LoRA config (`r`, `alpha`) and training parameters (`batch_size`, etc.) could also be part of the search space for Gemini.
# - **Resource Management:** More aggressive memory clearing if running into OOM issues on Kaggle, especially when handling multiple models or larger batches.
# ```
