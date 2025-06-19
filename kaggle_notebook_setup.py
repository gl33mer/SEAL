# This is a Kaggle notebook setup script

# Cell 1: Install necessary packages
!pip install transformers
!pip install peft
!pip install torch
!pip install accelerate
!pip install datasets
!pip install google-generativeai
# !pip install vllm # Optional, uncomment if needed and if installation is straightforward

# Cell 2: Import libraries
import torch
import transformers
from google.colab import userdata
import google.generativeai as genai
import os

# Cell 3: Configure Gemini API Key
# Attempt to get the API key from userdata (for Kaggle/Colab environments)
try:
    GEMINI_API_KEY = userdata.get('GEMINI_API_KEY')
except Exception as e:
    print(f"Could not retrieve GEMINI_API_KEY from userdata: {e}")
    GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE" # Fallback or for local execution

# Set the API key as an environment variable if it's not the placeholder
if GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
    os.environ['GEMINI_API_KEY'] = GEMINI_API_KEY
else:
    print("GEMINI_API_KEY is set to placeholder. Please replace it with your actual key for the script to work.")

# Configure the genai client if the key is available
if os.environ.get('GEMINI_API_KEY'):
    genai.configure(api_key=os.environ['GEMINI_API_KEY'])
    print("Gemini API key configured.")
else:
    print("Gemini API key not configured. Please set the GEMINI_API_KEY environment variable or replace the placeholder.")

print("Kaggle notebook setup script cells generated.")
