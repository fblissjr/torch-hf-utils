from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "/workspace/HunyuanVideo-PromptRewrite",
    trust_remote_code=True
)
model.save_pretrained(
    "/workspace/hyconverted",
    safe_serialization=True 
)