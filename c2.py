import os
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "/workspace/hytorch",
    trust_remote_code=True
)
model.save_pretrained(
    "/workspace/hy-hf",
    safe_serialization=True 
)