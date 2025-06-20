# seal_v2_notebook.py

# <markdown cell>
# # SEAL v2 Framework on Kaggle with Iterative Loop
#
# This notebook demonstrates an iterative version of the SEAL (Self-critique and Edit to Learn) framework.
# It uses the Gemini API for Teacher (ARC task solving) and Judge (evaluation and correction) roles,
# and a Qwen model as the Learner, fine-tuned with LoRA.
# The process focuses on iteratively refining the Learner's ability on specific ARC tasks.
# This version implements a full loop including Learner attempts, Judge feedback, data preparation based on feedback,
# and Learner fine-tuning, with detailed logging of each step.
# ```

# <markdown cell>
# ## Cell 1: Install Dependencies
#
# This cell lists the necessary Python libraries for the notebook.
# In a Kaggle environment, these should be run by uncommenting the `!pip install ...` line in a code cell.
# Key libraries include:
# - `transformers`: For Hugging Face models (Qwen).
# - `peft`: For LoRA (Parameter-Efficient Fine-Tuning).
# - `torch`: The core PyTorch library.
# - `accelerate`: For efficient model training on various hardware.
# - `datasets`: Used by the `Trainer` class.
# - `google-generativeai`: For interacting with the Gemini API (Teacher and Judge roles).
# - `bitsandbytes`: For optimizing model memory usage (e.g., 8-bit Adam optimizer).
# ```
# <code cell>
# ### Cell 1: Install Dependencies
print("Cell 1: Installing necessary libraries...")
# In a real Kaggle notebook, uncomment and run these with '!' prefix.
# print("!pip install transformers==4.40.1 peft==0.10.0 torch==2.2.1 accelerate==0.28.0 datasets==2.18.0 google-generativeai==0.5.0 bitsandbytes==0.42.0")
print("Done with installation instructions.")
# ```

# <markdown cell>
# ## Cell 2: Imports and Gemini API Key Setup
#
# This cell imports all required Python modules and sets up the Gemini API key.
# For this notebook to function with Gemini, the API key must be correctly configured.
# The recommended way in Kaggle is to use "Secrets" to store your `GEMINI_API_KEY`.
# The code will attempt to load it from Kaggle Secrets, then fall back to an environment variable,
# and finally to a placeholder string if not found (which will prevent Gemini calls from working).
# ```
# <code cell>
print("Cell 2: Setting up imports and Gemini API key...")
import os
import re
import json
import numpy as np
import torch
from tqdm import tqdm
from typing import List, Optional, Dict, Any, Callable, Tuple, Union
from datetime import datetime
from collections import Counter
import dataclasses
import sys
import ast

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType as PeftTaskType
from datasets import Dataset

import google.generativeai as genai

# Configure Gemini API Key
GEMINI_API_KEY = None
GEMINI_API_KEY_SOURCE = "Not Found"

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")
    GEMINI_API_KEY_SOURCE = "Kaggle Secrets"
    print("Successfully loaded GEMINI_API_KEY from Kaggle secrets.")
except Exception as e:
    print(f"Could not load GEMINI_API_KEY from Kaggle secrets: {e}. Trying environment variable.")
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
    if GEMINI_API_KEY:
        GEMINI_API_KEY_SOURCE = "Environment Variable"

if not GEMINI_API_KEY:
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE" # Fallback
    GEMINI_API_KEY_SOURCE = "Placeholder"
    print("GEMINI_API_KEY not found. Using placeholder.")

if GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE" and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print(f"Gemini API key configured successfully using: {GEMINI_API_KEY_SOURCE}")
    except Exception as e:
        print(f"Error configuring Gemini API (using key from {GEMINI_API_KEY_SOURCE}): {e}")
else:
    print("Please replace 'YOUR_GEMINI_API_KEY_HERE' with your actual key or set it up in Kaggle Secrets/Env for Gemini functionality.")

print(f"Gemini API Key starts with: {GEMINI_API_KEY[:5] if GEMINI_API_KEY else 'None'}... (Source: {GEMINI_API_KEY_SOURCE})")
print("Done with imports and API key setup.")
# ```

# <markdown cell>
# ## Cell 3: Embedded ARC Library Code (`arclib`)
#
# This cell contains essential helper classes for handling Abstract Reasoning Corpus (ARC) tasks.
# Since `arclib` (from the original SEAL repository) is not a standard pip-installable package, relevant components are embedded here:
# - **Data Structures (`Task`, `Example`, `Grid`):** These classes are fundamental for representing ARC tasks, including their training pairs and test cases, with grids stored as NumPy arrays.
# - **Representers & Messagers:** A minimal set of classes (`PythonListGridRepresenter`, `TextExampleRepresenter`, `TextTaskRepresenter`, `GPTTextMessageRepresenterV2`) are included. These handle the conversion of ARC task data into string formats or structured messages suitable for language models (both the Learner and the Gemini-based Teacher/Judge).
# - **`TTT` (Iterative Fine-Tuning Class):** This class manages the fine-tuning of the Learner model (Qwen) using LoRA. It has been adapted to support cumulative learning by allowing an `existing_lora_adapter_path` to be passed to its constructor. If provided, `TTT` loads this adapter and continues training it. Otherwise, it initializes a new LoRA adapter on the base model.
# - **Augmenters:** A base `Augmenter` class and a dummy `IdentityAugmenter` are provided. Data augmentation is not actively used in the main iterative loop of this PoC for simplicity but could be integrated.
# ```
# <code cell>
print("Cell 3: Defining arclib components...")

Grid = np.ndarray
def to_list(arr): return [[int(e) for e in row] for row in arr]

@dataclasses.dataclass
class Example:
    input: Grid; output: Grid; name: Optional[str] = None; meta: Optional[Dict] = dataclasses.field(default_factory=dict)
    def __hash__(self) -> int: return hash((self.input.tobytes(), self.output.tobytes()))
    def __repr__(self) -> str: return f"Ex(in:{self.input.shape}, out:{self.output.shape})"
    @classmethod
    def deserialize(cls, data: dict, test: bool = False) -> "Example":
        input_arr = np.array(data["input"], dtype=np.int8)
        if test and "output" not in data : output_arr = np.array([[]], dtype=np.int8)
        elif "output" in data: output_arr = np.array(data["output"], dtype=np.int8)
        else: output_arr = np.array([[]], dtype=np.int8)
        return cls(input_arr, output_arr)

@dataclasses.dataclass
class Task:
    train_examples: List[Example]; test_examples: List[Example]; name: str = ""; description: Optional[str] = None; meta: Optional[Dict] = dataclasses.field(default_factory=dict)
    def __hash__(self) -> int: return hash((tuple(self.train_examples), tuple(self.test_examples), self.name))
    def __repr__(self) -> str: return f"Task(name='{self.name}', tr_len={len(self.train_examples)}, te_len={len(self.test_examples)})"
    @property
    def test_example(self) -> Example: return self.test_examples[0] if self.test_examples else None
    @classmethod
    def deserialize(cls, data: dict, task_name: Optional[str] = None) -> "Task":
        train_ex = [Example.deserialize(ex_data) for ex_data in data.get("train", [])]
        test_ex = [Example.deserialize(ex_data, test=True) for ex_data in data.get("test", [])]
        return cls(train_examples=train_ex, test_examples=test_ex, name=task_name or data.get("name", ""))

class GridRepresenter(ABC): @abstractmethod
def encode(self, grid: Grid, **kwargs) -> str: pass
class PythonListGridRepresenter(GridRepresenter):
    def encode(self, grid: Grid, **kwargs) -> str: return str(grid.tolist())

class ExampleRepresenter(ABC):
    grid_representer: GridRepresenter
    @abstractmethod
    def encode(self, example: Example, **kwargs) -> Tuple[str, str]: pass

class TextExampleRepresenter(ExampleRepresenter):
    def __init__(self, grid_representer: GridRepresenter = PythonListGridRepresenter(), input_header="Input:\n", output_header="Output:\n"):
        self.grid_representer = grid_representer; self.input_header = input_header; self.output_header = output_header
    def encode(self, example: Example, **kwargs) -> Tuple[str, str]:
        input_str = self.grid_representer.encode(example.input, **kwargs)
        output_str = self.grid_representer.encode(example.output, **kwargs) if example.output.size > 0 else "[[ ]]"
        return (f"{self.input_header}{input_str}", f"{self.output_header}{output_str}")

class TaskRepresenter(ABC):
    example_representer: ExampleRepresenter
    @abstractmethod
    def encode(self, task: Task, **kwargs) -> Any: pass

class TextTaskRepresenter(TaskRepresenter):
    def __init__(self, example_representer: ExampleRepresenter = TextExampleRepresenter()):
        self.example_representer = example_representer
    def encode(self, task: Task, **kwargs) -> str:
        parts = [f"Task: {task.name}"]
        if task.description: parts.append(f"Description: {task.description}")
        parts.append("Training Examples:")
        for i, ex in enumerate(task.train_examples): q, o = self.example_representer.encode(ex, **kwargs); parts.append(f"  Ex {i+1}:\n    {q}\n    {o}")
        parts.append("Test Input:")
        for i, ex in enumerate(task.test_examples): q, _ = self.example_representer.encode(ex, **kwargs); parts.append(f"  Test {i+1}:\n    {q}")
        return "\n".join(parts)

class MessageRepresenter(ABC):
    task_representer: TaskRepresenter
    @abstractmethod
    def encode(self, task: Task, **kwargs) -> Tuple[List[Dict[str, str]], Dict[str, str]]: pass

class GPTTextMessageRepresenterV2(MessageRepresenter):
    def __init__(self, task_representer: TaskRepresenter = TextTaskRepresenter()):
        self.task_representer = task_representer
    def encode(self, task: Task, **kwargs) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
        system_prompt = "You are an expert ARC puzzle solver. Given training examples (input/output pairs), determine the transformation rule and apply it to the test input to predict the test output. Represent grids as Python lists of lists."
        user_content_parts = ["Training Examples:"]
        for i, ex in enumerate(task.train_examples): q_str, o_str = self.task_representer.example_representer.encode(ex); user_content_parts.append(f"Example {i+1}:\n{q_str}\n{o_str}")
        target_assistant_output_str = ""
        if task.test_examples:
            test_ex_input_str, _ = self.task_representer.example_representer.encode(task.test_examples[0])
            user_content_parts.append(f"\nTest Input:\n{test_ex_input_str}\n\nPredict the output for the test input.")
            _, target_assistant_output_str = self.task_representer.example_representer.encode(task.test_examples[0])
        else: user_content_parts.append("\nNo test input provided.")
        input_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": "\n".join(user_content_parts)}]
        output_message = {"role": "assistant", "content": target_assistant_output_str}
        return input_messages, output_message

class Augmenter(ABC): @abstractmethod
def apply_to_task(self, task: Task, **kwargs) -> Task: pass
class IdentityAugmenter(Augmenter):
    def apply_to_task(self, task: Task, **kwargs) -> Task: return Task(train_examples=task.train_examples[:], test_examples=task.test_examples[:], name=task.name + "_id_aug", description=task.description, meta=task.meta.copy())

class TTT:
    def __init__(self, model_name: str, lora_config: Optional[LoraConfig] = None, device: str = "cuda" if torch.cuda.is_available() else "cpu", existing_lora_adapter_path: Optional[str]=None):
        self.model_name = model_name; self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token

        model_load_args = {"torch_dtype": torch.bfloat16 if self.device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16}
        base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_load_args)

        if existing_lora_adapter_path and os.path.exists(existing_lora_adapter_path):
            print(f"TTT: Loading existing LoRA adapter from: {existing_lora_adapter_path}")
            self.model = PeftModel.from_pretrained(base_model, existing_lora_adapter_path, is_trainable=True).to(self.device)
            self.initial_lora_A = {}
            print(f"TTT: Loaded PeftModel from {existing_lora_adapter_path} and set to trainable for continued fine-tuning.")
        elif lora_config:
            print("TTT: Initializing new LoRA layers from lora_config on base model.")
            self.model = get_peft_model(base_model.to(self.device), lora_config)
            self.initial_lora_A = {name: param.data.clone().detach() for name, param in self.model.named_parameters() if "lora_A" in name}
        else:
            print("TTT: Warning - Initialized without existing adapter or new LoRA config. Using base model as is.")
            self.model = base_model.to(self.device); self.initial_lora_A = {}

    def update_model(self, training_texts: List[str], output_dir: str, batch_size: int, grad_accum: int, lr: float, epochs: int, lr_scheduler: str, loss_on_all: bool) -> str:
        # If this TTT instance was initialized with a *new* lora_config (initial_lora_A is populated),
        # then reset_lora will reset those specific LoRA layers.
        # If it was initialized with an existing_lora_adapter_path, initial_lora_A is empty, so reset_lora does nothing,
        # which is correct as we want to continue training the loaded adapter weights.
        if self.initial_lora_A : self.reset_lora()

        tokenized_data = self._tokenize_for_ttt_training(training_texts, loss_on_all)
        torch.cuda.empty_cache(); self._train_model(tokenized_data, output_dir, batch_size, grad_accum, lr, epochs, lr_scheduler)
        self.model.save_pretrained(output_dir); self.tokenizer.save_pretrained(output_dir)
        print(f"TTT: Model/adapter saved to {output_dir}"); return output_dir
    def reset_lora(self):
        if not self.initial_lora_A: print("TTT: No initial LoRA A weights to reset (e.g., when continuing training on loaded adapter or using base model)."); return
        print("TTT: Resetting newly added LoRA layers (A to initial, B to zero).")
        for name, param in self.model.named_parameters():
            if "lora_B" in name and name.replace("lora_B","lora_A") in self.initial_lora_A : param.data.fill_(0.0)
            elif "lora_A" in name and name in self.initial_lora_A: param.data.copy_(self.initial_lora_A[name])
    def _tokenize_for_ttt_training(self, texts: List[str], loss_on_all: bool):
        outputs = self.tokenizer(texts, truncation=True, max_length=1024, padding="longest", return_tensors="pt")
        input_ids, attention_mask = outputs.input_ids.to(self.device), outputs.attention_mask.to(self.device)
        labels = input_ids.clone()
        if not loss_on_all:
            for i in range(input_ids.shape[0]):
                try: mask_len = int(labels.shape[1] * 0.50) ; labels[i, :mask_len] = -100 # Simplified masking
                except Exception as e: print(f"W: Error TTT label masking sample {i}: {e}. Full loss.")
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    def _train_model(self, data: Dict, output_dir: str, batch_size: int, grad_accum: int, lr: float, epochs: int, lr_scheduler: str):
        ds = Dataset.from_dict(data)
        use_fp16 = self.device == "cuda" and not torch.cuda.is_bf16_supported()
        use_bf16 = self.device == "cuda" and torch.cuda.is_bf16_supported()
        train_args = TrainingArguments(
            output_dir=output_dir, per_device_train_batch_size=batch_size, gradient_accumulation_steps=grad_accum,
            learning_rate=lr, num_train_epochs=epochs, lr_scheduler_type=lr_scheduler,
            logging_steps=max(1, int(epochs * len(ds) / (batch_size * grad_accum * 20))),
            save_strategy="no", report_to="none", fp16=use_fp16, bf16=use_bf16, remove_unused_columns=False,
            optim="adamw_torch_fused" if self.device=="cuda" else "adamw_torch", warmup_ratio=0.1,
            dataloader_pin_memory=False if sys.platform == "darwin" else True, )
        trainer = Trainer(model=self.model, args=train_args, train_dataset=ds, tokenizer=self.tokenizer)
        print(f"TTT: Train on {len(ds)} samples. Epochs:{epochs},LR:{lr},Batch:{batch_size},Accum:{grad_accum},Dev:{self.device}")
        trainer.train(); print("TTT: Training complete.")
print("Cell 3: Done defining arclib components.")
# ```

# <markdown cell>
# ### Cell 4: Utility Prompts for Configuration Generation
#
# This cell contains prompts that were originally designed for a version of the SEAL framework where Gemini
# would generate training configurations (like learning rate, augmentation choices, etc.).
# - `self_edit_prompt`: Instructs Gemini to output a JSON with data augmentation and LoRA training parameters.
# - `system_message`: A general system message for Gemini in such a configuration generation role.
#
# **Note:** In the current iterative loop implemented in Cell 7, these specific prompts for *generating training configurations* are **not used**.
# The fine-tuning parameters for LoRA (learning rate, epochs) are currently set to fixed placeholder values within the loop.
# These prompts are retained here for reference or potential future use if dynamic configuration generation by an LLM is re-integrated.
# The prompts for the Teacher (task solving) and Judge (evaluation) roles are defined separately in Cell 6.
# ```
# <code cell>
print("Cell 4: Defining utility prompts...")
self_edit_prompt = """
You are an AI assistant tasked with configuring a model training pipeline for Abstract Reasoning Corpus (ARC) style tasks.
Your goal is to propose a configuration for data augmentation and LoRA fine-tuning parameters.

1.  **Data Generation Strategy**:
    For each data augmentation technique, decide if it should be used (true/false):
    - `use_basic_augmentations`: Basic flips, rotations.
    - `use_size_augmentations`: Operations like increasing resolution.
    - `use_chain_augmentations`: Combining multiple augmenters (e.g., rotate then increase resolution).
    - `use_repeat_augmentations`: Repeating patterns within the grid.

2.  **Training Configuration**:
    - `strategy`: Choose one training strategy:
        - "train_using_all_tokens": Calculate loss on both prompt and completion tokens.
        - "train_using_output_tokens": Calculate loss only on completion (output) tokens.
    - `learning_rate`: Suggest a learning rate (float between 1e-5 and 5e-4).
    - `num_train_epochs`: Suggest number of training epochs (integer between 1 and 3 for this PoC).

Respond with a valid JSON object containing your choices. Ensure all boolean values are lowercase.
Example JSON Output:
{
  "data_generation": { "use_basic_augmentations": true, "use_size_augmentations": false, "use_chain_augmentations": false, "use_repeat_augmentations": true },
  "training": { "strategy": "train_using_output_tokens", "learning_rate": 2e-5, "num_train_epochs": 2 }
}"""
system_message = "You are a helpful assistant designed to output JSON configurations." # For Gemini config generation
print("Cell 4: Done defining utility prompts (self_edit_prompt for config generation is currently not used in the main v2 loop).")
# ```

# <markdown cell>
# ### Cell 5: Experiment Configuration and Data Loading
#
# This cell defines the `Args` class, which centralizes all configurations for the SEAL experiment.
# This includes model names for the Learner, Teacher, and Judge, parameters for the iterative loop (e.g., number of iterations),
# LoRA settings, and output directory information.
#
# It also defines the `sample_arc_data`, a small, embedded dataset of ARC-like tasks. These tasks are parsed into
# `Task` objects (defined in Cell 3) for use throughout the notebook.
# ```
# <code cell>
print("Cell 5: Initializing configurations and expanded sample data...")
class Args:
    def __init__(self):
        self.experiment_name = "seal_v2_iterative_loop_poc"
        self.base_learner_model_name = "Qwen/Qwen2.5-1.5B"
        self.teacher_model_name = "gemini-1.5-flash-latest"
        self.judge_model_name = "gemini-1.5-flash-latest"
        self.num_seal_iterations = 2
        self.configs_per_iteration_per_task = 1
        self.max_configs_to_store_per_task = 5
        self.max_context_configs_for_teacher = 2
        self.lora_r = 16; self.lora_alpha = 32; self.lora_dropout = 0.05
        self.learner_batch_size = 1; self.learner_grad_accum = 2
        self.learner_lr_scheduler = "cosine"
        self.learner_max_train_samples_per_config = 50
        self.leave_n_out_values = [1]
        self.output_dir_base = "seal_v2_results"; os.makedirs(self.output_dir_base, exist_ok=True)
args = Args()
print(f"Config: Exp='{args.experiment_name}', Learner='{args.base_learner_model_name}', Teacher='{args.teacher_model_name}', Iterations: {args.num_seal_iterations}")
sample_arc_data = {
    "task_1_simple_pattern": {"train": [{"input": [[0,1,0],[0,1,0],[0,1,0]], "output": [[0,0,0],[1,1,1],[0,0,0]]}, {"input": [[1,0,0],[1,0,0],[1,0,0]], "output": [[0,0,0],[0,0,0],[1,1,1]]}],"test": [{"input": [[0,0,1],[0,0,1],[0,0,1]], "output": [[1,1,1],[0,0,0],[0,0,0]]}]},
    "task_2_fill_shape": {"train": [{"input": [[1,0,1],[0,0,0],[1,0,1]], "output": [[1,1,1],[1,1,1],[1,1,1]]}, {"input": [[0,0,0,0],[0,2,2,0],[0,2,2,0],[0,0,0,0]], "output": [[0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]]}],"test": [{"input": [[3,0,0,3],[0,0,0,0],[0,0,0,0],[3,0,0,3]], "output": [[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]}]},
    "task_3_draw_line": {"train": [{"input": [[0,0,0],[1,0,0],[0,0,0]], "output": [[0,0,0],[1,1,1],[0,0,0]]}, {"input": [[0,1,0],[0,0,0],[0,0,0]], "output": [[0,1,0],[0,1,0],[0,1,0]]}],"test": [{"input": [[0,0,0],[0,0,0],[0,0,1]], "output": [[0,0,0],[0,0,0],[1,1,1]]}]},
    "task_4_color_swap_conditional": {"train": [{"input": [[1,2,1],[2,1,2],[1,2,1]], "output": [[2,1,2],[1,2,1],[2,1,2]]}, {"input": [[3,0,3],[0,3,0],[3,0,3]], "output": [[3,0,3],[0,3,0],[3,0,3]]}],"test": [{"input": [[1,1,2],[3,1,0],[2,0,1]], "output": [[2,2,1],[3,2,0],[1,0,2]]}]}
}
parsed_arc_tasks = [Task.deserialize(task_data, task_name=name) for name, task_data in sample_arc_data.items()]
print(f"Parsed {len(parsed_arc_tasks)} sample tasks. First task: {parsed_arc_tasks[0].name if parsed_arc_tasks else 'None'}")
print("Cell 5: Done.")
# ```

# <markdown cell>
# ### Cell 6: Helper Functions (Grid Parsing, Teacher/Judge Interactions, Baseline Eval, Learner Inference)
#
# This cell consolidates various helper functions crucial for the SEAL framework:
# - **`compare_grids`**: A utility to compare two grids for exact equality in shape and content.
# - **`run_learner_inference`**: Executes the Learner model (Qwen). It can load the base model or apply a specified LoRA adapter for inference. It returns the raw text output and a parsed grid.
# - **`parse_grid_from_text`**: Utility to extract grid data (Python list of lists) from raw LLM text output using regular expressions and JSON/AST parsing.
# - **Teacher Components**:
#     - `ARC_TEACHER_PROMPT_TEMPLATE`: Defines the prompt structure for asking Gemini to solve an ARC task and explain its reasoning.
#     - `get_teacher_solution`: Formats a task, queries the Gemini API using the teacher prompt, and parses the response to get a thought process and a solution grid.
# - **Judge Components**:
#     - `ARC_JUDGE_PROMPT_TEMPLATE`: Defines the prompt structure for asking Gemini to evaluate a Learner's solution, provide a critique, a corrected grid, and a score.
#     - `get_judge_feedback`: Formats inputs (task, learner's solution, optional teacher's solution), queries the Gemini API using the judge prompt, and parses the response.
# - **`evaluate_baseline_performance`**: Function to assess the base Learner model's performance on all sample tasks before any fine-tuning iterations. This provides a reference point.
# ```

# <code cell>
print("Cell 6: Defining SEAL components and helper functions...")

def compare_grids(grid1: Optional[List[List[int]]], grid2: Optional[List[List[int]]]) -> bool:
    if grid1 is None or grid2 is None: return False
    if not isinstance(grid1, list) or not isinstance(grid2, list): return False
    try:
        arr1 = np.array(grid1); arr2 = np.array(grid2)
        return (arr1.shape == arr2.shape) and (arr1 == arr2).all()
    except Exception: return False

def run_learner_inference( base_model_name: str, tokenizer_to_use: AutoTokenizer, prompt_text: str, device_to_use: str, lora_adapter_path: Optional[str] = None, max_new_tokens_inf: int = 256, temperature_inf: float = 0.1, do_sample_inf: bool = False) -> Dict[str, Any]:
    print(f"\n--- Running Learner Inference ---"); print(f"Model: {base_model_name}, LoRA: {lora_adapter_path if lora_adapter_path else 'Base Model'}")
    model_to_run = None; raw_output = "Error: Model loading/inference failed."; parsed_grid = None; error_msg = None
    try:
        base_model_obj = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16 if device_to_use == "cuda" and torch.cuda.is_bf16_supported() else torch.float16)
        if lora_adapter_path and os.path.exists(lora_adapter_path):
            print(f"Loading LoRA adapter from: {lora_adapter_path}")
            model_to_run = PeftModel.from_pretrained(base_model_obj, lora_adapter_path).to(device_to_use)
        else:
            if lora_adapter_path: print(f"W: LoRA path '{lora_adapter_path}' not found. Using base model.")
            model_to_run = base_model_obj.to(device_to_use)
        model_to_run.eval()
        inputs = tokenizer_to_use(prompt_text, return_tensors="pt", truncation=True, max_length=1024).to(device_to_use)
        with torch.no_grad(): outputs = model_to_run.generate(**inputs, max_new_tokens=max_new_tokens_inf, pad_token_id=tokenizer_to_use.pad_token_id, temperature=temperature_inf, do_sample=do_sample_inf)
        prompt_num_tokens = inputs.input_ids.shape[1]; generated_tokens = outputs[0][prompt_num_tokens:]
        raw_output = tokenizer_to_use.decode(generated_tokens, skip_special_tokens=True)
        print(f"Learner Raw Output (first 150 chars): {raw_output[:150]}...")
        parsed_grid = parse_grid_from_text(raw_output)
    except Exception as e: print(f"E during learner inference: {e}"); raw_output = f"E during learner inference: {str(e)}"; error_msg = str(e)
    finally:
        if model_to_run: del model_to_run
        if 'base_model_obj' in locals(): del base_model_obj
        torch.cuda.empty_cache()
    return {"raw_output": raw_output, "parsed_grid": parsed_grid, "error_message": error_msg}

def parse_grid_from_text(text_output: str) -> Optional[List[List[int]]]:
    if not text_output or not isinstance(text_output, str): return None
    pattern = r"(\[\[.*?\]\])"
    matches = re.findall(pattern, text_output, re.DOTALL)
    if not matches:
        simple_list_pattern = r"(\[.*?\])"
        matches = re.findall(simple_list_pattern, text_output, re.DOTALL)
        if not matches: print("W: No grid-like pattern found in text_output."); return None
    best_match_str = max(matches, key=len)
    grid_str_for_parse = best_match_str.replace("\n", "").replace(" ", "")
    grid_str_for_parse = grid_str_for_parse.replace("],[", "], [")
    try: parsed_grid = json.loads(grid_str_for_parse)
    except json.JSONDecodeError:
        try: parsed_grid = ast.literal_eval(grid_str_for_parse)
        except (SyntaxError, ValueError, TypeError) as e: print(f"E parsing grid with ast: '{grid_str_for_parse}'. Err: {e}"); return None
    if isinstance(parsed_grid, list) and all(isinstance(row, list) for row in parsed_grid):
        final_grid = []
        for r_idx, row in enumerate(parsed_grid):
            new_row = []
            for c_idx, item in enumerate(row):
                if isinstance(item, int): new_row.append(item)
                elif isinstance(item, (float, str)) and str(item).replace('.', '', 1).replace('-', '', 1).isdigit():
                    try: new_row.append(int(float(item)))
                    except ValueError: print(f"W: Non-int '{item}' at [{r_idx}][{c_idx}]. Ret None."); return None
                else: print(f"W: Non-int '{item}' at [{r_idx}][{c_idx}]. Ret None."); return None
            final_grid.append(new_row)
        return final_grid
    else: print(f"Parsed structure not list of lists: {parsed_grid}"); return None

ARC_TEACHER_PROMPT_TEMPLATE = """You are an expert at solving Abstract Reasoning Corpus (ARC) puzzles.
You will be given a puzzle with a few training examples (input -> output pairs) and a final test input grid.
Your goal is to understand the transformation rule from the training examples and apply it to the test input grid.
Please provide your answer in two parts:
1. Thought Process: Explain step-by-step how you identified the pattern from the training examples and how this pattern applies to the test input. Be concise but clear.
2. Solution Grid: Provide only the solution grid for the test input, formatted as a Python list of lists. For example: [[1,2,3],[4,5,6]]
Here is the puzzle:
---
{task_description_string}
---
Remember to output the Solution Grid clearly and accurately in the specified list of lists format."""

def get_teacher_solution(task_obj: Task, teacher_gemini_model_name: str, task_desc_formatter: TextTaskRepresenter, prompt_template: str) -> Tuple[Optional[str], Optional[List[List[int]]]]:
    print(f"\n--- Requesting Teacher Solution for Task: {task_obj.name} ---")
    task_description_str = task_desc_formatter.encode(task_obj)
    full_prompt_for_teacher = prompt_template.format(task_description_string=task_description_str)
    if not genai.API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE": print("E: Gemini API key not configured."); return "E: Gemini API key not configured.", None
    try:
        model = genai.GenerativeModel(teacher_gemini_model_name); response = model.generate_content(full_prompt_for_teacher)
        raw_teacher_output = response.text if hasattr(response, 'text') and response.text else (response.parts[0].text if response.parts and hasattr(response.parts[0], 'text') else "")
        if not raw_teacher_output: print(f"E: Could not extract text from Teacher response. Resp: {response}"); return "E: No text in Gemini resp.", None
        thought_process, solution_grid_str = None, None; solution_grid_heading = "Solution Grid:"
        if solution_grid_heading in raw_teacher_output:
            parts = raw_teacher_output.split(solution_grid_heading, 1); thought_process = parts[0].replace("Thought Process:", "").strip(); solution_grid_str = parts[1].strip()
        else: thought_process = raw_teacher_output
        parsed_solution_grid = parse_grid_from_text(solution_grid_str if solution_grid_str else raw_teacher_output)
        if not parsed_solution_grid and not solution_grid_str : thought_process = "Could not sep thought. Raw output may contain grid."
        elif not parsed_solution_grid: print(f"W: Failed to parse sol grid from Teacher. Grid str: '{solution_grid_str}'")
        return thought_process, parsed_solution_grid
    except Exception as e: print(f"E getting Teacher solution for {task_obj.name}: {e}"); return f"E: API call failed: {e}", None

ARC_JUDGE_PROMPT_TEMPLATE = """You are an expert ARC Puzzle Judge. You will be given an ARC puzzle, a Learner's attempted solution grid, and optionally, an Expert Teacher's solution grid for reference.
Your tasks are:
1. Critique: Analyze the Learner's solution. Is it correct? If not, what specific errors did the Learner make in applying the pattern from the training examples? If correct, acknowledge it.
2. Corrected Solution Grid: Provide the correct solution grid. If the Learner's solution was already correct, you can re-state it. This must be formatted as a Python list of lists (e.g., [[1,0],[0,1]]). This is the most crucial part of your output.
3. Score: Give a score from 0 (completely wrong) to 10 (perfectly correct) for the Learner's attempt based on its closeness to the true solution.
Puzzle Description:
---
{task_description_string}
---
Learner's Attempted Solution Grid:
{learner_solution_grid_string}
---
Expert Teacher's Solution Grid (for your reference, if available, otherwise this section might be empty):
{teacher_solution_grid_string}
---
Please provide your evaluation clearly. Ensure the 'Corrected Solution Grid' is accurately formatted.
Output Format Example:
Critique: The learner correctly identified the pattern of swapping colors but missed the conditional aspect on the third row.
Corrected Solution Grid: [[0,1,1],[1,1,0],[0,0,1]]
Score: 7"""
def get_judge_feedback(task_obj: Task, learner_solution_grid: List[List[int]], judge_gemini_model_name: str, task_desc_formatter: TextTaskRepresenter, parse_grid_from_text_func: callable, prompt_template: str, teacher_solution_grid: Optional[List[List[int]]] = None) -> Tuple[Optional[str], Optional[List[List[int]]], Optional[int]]:
    print(f"\n--- Requesting Judge Feedback for Task: {task_obj.name} ---")
    task_description_str = task_desc_formatter.encode(task_obj)
    learner_solution_grid_str = str(learner_solution_grid) if learner_solution_grid else "None"; teacher_solution_grid_str = str(teacher_solution_grid) if teacher_solution_grid else "Not available"
    full_prompt_for_judge = prompt_template.format(task_description_string=task_description_str, learner_solution_grid_string=learner_solution_grid_str, teacher_solution_grid_string=teacher_solution_grid_str)
    if not genai.API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE": print("E: Gemini API key not configured."); return "E: Gemini API key not configured.", None, None
    try:
        model = genai.GenerativeModel(judge_gemini_model_name); response = model.generate_content(full_prompt_for_judge)
        raw_judge_output = response.text if hasattr(response, 'text') and response.text else (response.parts[0].text if response.parts and hasattr(response.parts[0], 'text') else "")
        if not raw_judge_output: print(f"E: Could not extract text from Judge response. Resp: {response}"); return "E: No text in Gemini resp.", None, None
        critique, corrected_grid, score = None, None, None
        critique_match = re.search(r"Critique:(.*?)(Corrected Solution Grid:|Score:|$)", raw_judge_output, re.DOTALL | re.IGNORECASE);
        if critique_match: critique = critique_match.group(1).strip()
        corrected_grid_match = re.search(r"Corrected Solution Grid:(.*?)(Critique:|Score:|$)", raw_judge_output, re.DOTALL | re.IGNORECASE)
        if corrected_grid_match: grid_str = corrected_grid_match.group(1).strip(); corrected_grid = parse_grid_from_text_func(grid_str)
        score_match = re.search(r"Score:\s*(\d+)", raw_judge_output, re.IGNORECASE)
        if score_match:
            try: score = int(score_match.group(1).strip())
            except ValueError: print(f"W: Could not parse score '{score_match.group(1)}'.")
        if not critique and not corrected_grid and not score: # Fallback parsing
            if not corrected_grid: corrected_grid = parse_grid_from_text_func(raw_judge_output)
            if not critique and raw_judge_output and not corrected_grid: critique = raw_judge_output
            elif not critique and raw_judge_output and corrected_grid : critique = raw_judge_output.split(str(corrected_grid))[0].strip()
        return critique, corrected_grid, score
    except Exception as e: print(f"E getting Judge feedback for {task_obj.name}: {e}"); return f"E: Judge API call failed: {e}", None, None

def evaluate_baseline_performance(current_args: Args, tasks: List[Task], tokenizer, model_formatter: MessageRepresenter):
    print("\n--- Evaluating Baseline Performance ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"; baseline_results = []
    for i, task_obj in enumerate(tasks):
        print(f"\nEvaluating Task {i+1}/{len(tasks)}: {task_obj.name}")
        if not task_obj.test_examples: print(f"Skipping task {task_obj.name}: No test examples."); continue
        test_example = task_obj.test_examples[0]
        input_messages, _ = model_formatter.encode(Task(train_examples=task_obj.train_examples, test_examples=[test_example], name=task_obj.name))
        inference_prompt = tokenizer.apply_chat_template(input_messages, tokenize=False, add_generation_prompt=True)
        learner_result_dict = run_learner_inference(current_args.base_learner_model_name, tokenizer, inference_prompt, device, None)
        raw_text_output, parsed_pred_grid = learner_result_dict["raw_output"], learner_result_dict["parsed_grid"]
        match_status = False; ground_truth_grid = test_example.output.tolist()
        if parsed_pred_grid: match_status = compare_grids(parsed_pred_grid, ground_truth_grid)
        print(f"  Match Status (Baseline): {'Exact Match' if match_status else 'No Match'}")
        baseline_results.append({"task_name": task_obj.name, "parsed_predicted_grid": parsed_pred_grid, "ground_truth_grid": ground_truth_grid, "match_status": match_status, "raw_output": raw_text_output})
        torch.cuda.empty_cache()
    print("\n--- Baseline Performance Evaluation Complete ---")
    return baseline_results
print("Cell 6: Done with component definitions.")
# ```

# <markdown cell>
# ### Cell 7: Main SEAL Iterative Loop and Execution
#
# This cell defines and runs the main SEAL iterative loop.
# The loop focuses on a single task from `parsed_arc_tasks` for a set number of iterations.
# In each iteration:
# 1.  **Learner Attempt:** The Qwen model (with the current LoRA adapter, or base if first iteration) attempts to solve the task.
# 2.  **Judge Step:** Gemini evaluates the Learner's attempt, providing a critique, a corrected solution, and a score.
# 3.  **Learning Data Prep:** If the Judge provides a valid correction and the learner's attempt was not perfect, this corrected solution is formatted as a new training instance.
# 4.  **Learner Fine-tuning:** The Qwen model is fine-tuned. If an existing LoRA adapter path is provided from a previous iteration, `TTT` loads it and continues training. Otherwise, it creates a new LoRA adapter from the base model.
# 5.  **Evaluation (Post-Iteration):** The newly fine-tuned Learner attempts the task again to measure improvement within the iteration.
# All results from each step of the iteration are logged and then printed in a structured format.
# ```

# <code cell>
print("Cell 7: Defining and running the main SEAL iterative loop...")

def run_seal_iterative_loop(
    args_config: Args, tasks_list: List[Task],
    ext_learner_tokenizer: AutoTokenizer, ext_learner_model_formatter: MessageRepresenter,
    ext_teacher_judge_task_formatter: TextTaskRepresenter, ext_parse_grid_from_text_func: callable,
    ext_gemini_api_key: str
):
    print(f"\n===== Starting SEAL Iterative Loop =====")
    current_lora_adapter_path: Optional[str] = None
    iteration_results_log = []

    lora_config_for_new_adapters = LoraConfig(r=args_config.lora_r, lora_alpha=args_config.lora_alpha, lora_dropout=args_config.lora_dropout, bias="none", task_type=PeftTaskType.CAUSAL_LM, target_modules="all-linear")

    if not tasks_list: print("No tasks for iterative loop."); return []
    target_task = tasks_list[0]
    print(f"Target task for iterative improvement: {target_task.name}")
    ground_truth_grid = target_task.test_examples[0].output.tolist()

    for iteration_num in range(args_config.num_seal_iterations):
        print(f"\n--- SEAL Iteration {iteration_num + 1}/{args_config.num_seal_iterations} for Task: {target_task.name} ---")

        learner_trainer = TTT(model_name=args_config.base_learner_model_name,
                              lora_config=lora_config_for_new_adapters if not current_lora_adapter_path else None,
                              device="cuda" if torch.cuda.is_available() else "cpu",
                              existing_lora_adapter_path=current_lora_adapter_path)

        log_entry = {
            "iteration_number": iteration_num + 1, "task_name": target_task.name,
            "lora_used_for_attempt": str(current_lora_adapter_path) if current_lora_adapter_path else "Base Model",
            "ground_truth_grid": ground_truth_grid,
        }

        # 1. Learner Attempt
        print(f"Step 1: Learner Attempt (Adapter: {log_entry['lora_used_for_attempt']})")
        learner_prompt_messages, _ = ext_learner_model_formatter.encode(Task(train_examples=target_task.train_examples, test_examples=[target_task.test_examples[0]], name=target_task.name))
        learner_inference_prompt = ext_learner_tokenizer.apply_chat_template(learner_prompt_messages, tokenize=False, add_generation_prompt=True)

        attempt_result = run_learner_inference(
            base_model_name=args_config.base_learner_model_name, tokenizer_to_use=ext_learner_tokenizer,
            prompt_text=learner_inference_prompt, device_to_use="cuda" if torch.cuda.is_available() else "cpu",
            lora_adapter_path=current_lora_adapter_path
        )
        log_entry["learner_attempt_raw_output"] = attempt_result["raw_output"]
        log_entry["learner_attempt_parsed_grid"] = attempt_result["parsed_grid"]
        log_entry["match_with_ground_truth_before_iteration"] = compare_grids(attempt_result["parsed_grid"], ground_truth_grid)

        if not attempt_result["parsed_grid"]:
            print("W: Learner failed to produce parsable grid in attempt.")
            log_entry.update({
                "judge_critique": "N/A (Learner failed to parse)", "judge_corrected_grid": None, "judge_score": 0,
                "match_with_judge_correction_before_iteration": False, "is_correction_used_for_training": False,
                "finetuning_config": None, "new_lora_adapter_path": current_lora_adapter_path,
                "learner_post_iteration_raw_output": "N/A", "learner_post_iteration_parsed_grid": None,
                "match_with_ground_truth_after_iteration": False, "match_with_judge_correction_after_iteration": False
            })
            iteration_results_log.append(log_entry)
            continue

        # 2. Judge Step
        print(f"Step 2: Judge Evaluation")
        critique, judge_corrected_grid, score = get_judge_feedback(
            target_task, attempt_result["parsed_grid"], args_config.judge_model_name,
            ext_teacher_judge_task_formatter, ext_parse_grid_from_text_func,
            ARC_JUDGE_PROMPT_TEMPLATE, teacher_solution_grid=ground_truth_grid
        )
        log_entry.update({"judge_critique": critique, "judge_corrected_grid": judge_corrected_grid, "judge_score": score})
        log_entry["match_with_judge_correction_before_iteration"] = compare_grids(attempt_result["parsed_grid"], judge_corrected_grid)

        is_correction_useful = False
        if judge_corrected_grid and not compare_grids(attempt_result["parsed_grid"], judge_corrected_grid) and (score is None or score < 10):
            is_correction_useful = True
        log_entry["is_correction_used_for_training"] = is_correction_useful

        # 3. Learning Data Prep & 4. Learner Fine-tuning
        if is_correction_useful:
            print(f"Step 3 & 4: Preparing Data from Judge & Fine-tuning Learner")
            test_ex_for_learning = Example(input=target_task.test_examples[0].input, output=np.array(judge_corrected_grid, dtype=np.int8))
            learning_task = Task(train_examples=target_task.train_examples, test_examples=[test_ex_for_learning], name=f"{target_task.name}_iter{iteration_num+1}_ft")
            learning_input_msgs, learning_output_msg = ext_learner_model_formatter.encode(learning_task)
            training_text = ext_learner_tokenizer.apply_chat_template(learning_input_msgs + [learning_output_msg], tokenize=False, add_generation_prompt=False)
            training_instances_for_finetuning = [training_text]
            log_entry["learning_data_example_snippet"] = training_instances_for_finetuning[0][:200] + "..."

            ft_lr, ft_epochs, ft_strat = 2e-5, 1, "train_using_output_tokens"
            log_entry["finetuning_config"] = {"lr": ft_lr, "epochs": ft_epochs, "strategy": ft_strat}
            lora_output_dir_iter = os.path.join(args_config.output_dir_base, args_config.experiment_name, target_task.name.replace("/","_"), f"iteration_{iteration_num + 1}")
            os.makedirs(lora_output_dir_iter, exist_ok=True)
            new_adapter_path = learner_trainer.update_model(
                training_instances_for_finetuning, lora_output_dir_iter, args_config.learner_batch_size, args_config.learner_grad_accum,
                ft_lr, ft_epochs, args_config.learner_lr_scheduler, loss_on_all=(ft_strat == "train_using_all_tokens")
            )
            if new_adapter_path and os.path.exists(new_adapter_path): current_lora_adapter_path = new_adapter_path
            else: print(f"W: Fine-tuning failed or adapter not saved for iter {iteration_num+1}. Current LoRA path remains: {current_lora_adapter_path}")
        else:
            print("Step 3 & 4: No useful correction from Judge or learner was already perfect according to Judge. Skipping fine-tuning.")
            log_entry["finetuning_config"] = None
        log_entry["new_lora_adapter_path"] = current_lora_adapter_path

        # 5. Evaluation (Post-Iteration)
        print(f"Step 5: Evaluating Learner Post-Iteration (Adapter: {current_lora_adapter_path})")
        post_iter_result = run_learner_inference(
            args_config.base_learner_model_name, ext_learner_tokenizer, learner_inference_prompt,
            "cuda" if torch.cuda.is_available() else "cpu", current_lora_adapter_path
        )
        log_entry["learner_post_iteration_raw_output"] = post_iter_result["raw_output"]
        log_entry["learner_post_iteration_parsed_grid"] = post_iter_result["parsed_grid"]
        log_entry["match_with_ground_truth_after_iteration"] = compare_grids(post_iter_result["parsed_grid"], ground_truth_grid)
        log_entry["match_with_judge_correction_after_iteration"] = compare_grids(post_iter_result["parsed_grid"], judge_corrected_grid) # Compare with this iteration's judge correction

        iteration_results_log.append(log_entry)
        del learner_trainer
        torch.cuda.empty_cache()

    print(f"\n===== SEAL Iterative Loop Finished for Task: {target_task.name} =====")
    return iteration_results_log

# --- Main Execution Block (Cell 7 continued) ---
if __name__ == '__main__' or 'ipykernel' in sys.modules:
    if 'current_args' not in locals(): current_args = Args()
    if 'learner_tokenizer' not in locals():
        try:
            learner_tokenizer = AutoTokenizer.from_pretrained(current_args.base_learner_model_name)
            if learner_tokenizer.pad_token is None: learner_tokenizer.pad_token = learner_tokenizer.eos_token
        except Exception as e: sys.exit(f"FATAL in Cell 7: Learner tokenizer failed: {e}")
    if 'learner_model_formatter' not in locals(): learner_model_formatter = GPTTextMessageRepresenterV2()
    if 'teacher_judge_task_formatter' not in locals(): teacher_judge_task_formatter = TextTaskRepresenter()
    if 'parsed_arc_tasks' not in globals() or not parsed_arc_tasks : parsed_arc_tasks = [Task.deserialize(data, name) for name, data in sample_arc_data.items()]

    if 'baseline_results' not in globals():
        baseline_results = evaluate_baseline_performance(current_args, parsed_arc_tasks, learner_tokenizer, learner_model_formatter)
        print("\n--- Baseline Evaluation Summary (from Cell 7 execution) ---")
        if baseline_results: num_matches = sum(1 for r in baseline_results if r["match_status"]); acc = (num_matches / len(baseline_results) * 100) if baseline_results else 0; print(f"Tasks: {len(baseline_results)}, Matches: {num_matches}, Acc: {acc:.2f}%")
        else: print("No baseline results from this run.")

    print("\n\n--- Attempting to Run Main SEAL Iterative Loop ---")
    iteration_log = run_seal_iterative_loop(current_args, parsed_arc_tasks, learner_tokenizer, learner_model_formatter, teacher_judge_task_formatter, parse_grid_from_text, GEMINI_API_KEY)

    print("\n\n" + "="*70); print("SEAL ITERATIVE LOOP - DETAILED LOGS"); print("="*70)
    for entry in iteration_log:
        print(f"\n--- Iteration: {entry['iteration_number']} | Task: {entry['task_name']} ---")
        print(f"LoRA Used for Initial Attempt: {entry.get('lora_used_for_attempt', 'N/A')}")
        print("\n[Initial Learner Attempt]")
        print(f"  Raw Output: {entry.get('learner_attempt_raw_output', 'N/A')[:200]}...")
        print(f"  Parsed Grid: {entry.get('learner_attempt_parsed_grid', 'N/A')}")
        print(f"  Matches Ground Truth: {entry.get('match_with_ground_truth_before_iteration', 'N/A')}")
        print(f"  Matches Judge's Correction (this iter): {entry.get('match_with_judge_correction_before_iteration', 'N/A')}")
        print("\n[Judge Feedback]")
        print(f"  Critique: {entry.get('judge_critique', 'N/A')}")
        print(f"  Corrected Grid by Judge: {entry.get('judge_corrected_grid', 'N/A')}")
        print(f"  Score by Judge: {entry.get('judge_score', 'N/A')}")
        print("\n[Learning & Fine-tuning]")
        print(f"  Correction used for training: {entry.get('is_correction_used_for_training', 'N/A')}")
        print(f"  Fine-tuning Config: {entry.get('finetuning_config', 'N/A')}")
        print(f"  New LoRA Adapter Path: {entry.get('new_lora_adapter_path', 'N/A')}")
        print("\n[Post-Iteration Learner Attempt]")
        print(f"  Raw Output: {entry.get('learner_post_iteration_raw_output', 'N/A')[:200]}...")
        print(f"  Parsed Grid: {entry.get('learner_post_iteration_parsed_grid', 'N/A')}")
        print(f"  Matches Ground Truth: {entry.get('match_with_ground_truth_after_iteration', 'N/A')}")
        print(f"  Matches Judge's Correction (this iter): {entry.get('match_with_judge_correction_after_iteration', 'N/A')}")
        print("-"*(70))

    print("\n" + "="*70); print("SEAL ITERATIVE LOOP - OVERALL SUMMARY"); print("="*70)
    if iteration_log:
        task_summary = {}
        for entry in iteration_log:
            task_name = entry['task_name']
            if task_name not in task_summary:
                # For initial_match_gt, use the first iteration's "before_iteration" status
                task_summary[task_name] = {"initial_match_gt": iteration_log[0]['match_with_ground_truth_before_iteration'] if iteration_log[0]['task_name'] == task_name else "N/A", "iterations": []}
            task_summary[task_name]["iterations"].append({
                "iter": entry['iteration_number'],
                "judge_score": entry.get('judge_score', 'N/A'),
                "post_tune_match_gt": entry.get('match_with_ground_truth_after_iteration', 'N/A'),
                "final_lora": entry.get('new_lora_adapter_path', 'N/A')
            })
        for task_name, summary_data in task_summary.items():
            print(f"Task Focus: {task_name}")
            print(f"  Initial Match with Ground Truth (Base Model or Initial LoRA): {summary_data['initial_match_gt']}")
            final_iter_data = summary_data['iterations'][-1]
            print(f"  Match with Ground Truth after {len(summary_data['iterations'])} SEAL Iterations: {final_iter_data['post_tune_match_gt']}")
            print(f"  Final LoRA adapter path: {final_iter_data['final_lora']}")
            for iter_data in summary_data['iterations']:
                print(f"    Iter {iter_data['iter']}: Judge Score: {iter_data['judge_score']}, Post-Tune Match GT: {iter_data['post_tune_match_gt']}")
    else: print("No iterations were logged.")
print("\nCell 7: Full execution finished.")
# ```

print("seal_v2_notebook.py setup complete with iterative loop structure and main execution in Cell 7.")

[end of seal_v2_notebook.py]

[end of seal_v2_notebook.py]
