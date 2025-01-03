import json
import re

def transform_parameter_name(param_name):
    """Transforms the parameter name to a potentially compatible format."""

    moe_pattern = r"model\.layers\.(\d+)\.mlp\.mlp\.experts\.(\d+)\.(up_proj|down_proj|gate_proj)\.(weight|bias)"
    shared_moe_pattern = r"model\.layers\.(\d+)\.mlp\.shared_mlp\.(gate_proj|down_proj)\.(weight|bias)"
    self_attn_pattern = r"model\.layers\.(\d+)\.self_attn\.(query_proj|key_layernorm|value_proj|o_proj|query_layernorm|key_proj|v_proj|q_proj)\.(weight|bias)"

    match = re.match(moe_pattern, param_name)
    if match:
        layer, expert, proj_type, param_type = match.groups()
        proj_type = proj_type.replace("_proj", "")  # Remove "_proj" suffix
        return f"model.layers.{layer}.mlp.experts.{expert}.{proj_type}.{param_type}"

    match = re.match(shared_moe_pattern, param_name)
    if match:
        layer, proj_type, param_type = match.groups()
        proj_type = proj_type.replace("_proj", "")
        return f"model.layers.{layer}.mlp.{proj_type}.{param_type}"

    match = re.match(self_attn_pattern, param_name)
    if match:
        layer, attn_type, param_type = match.groups()
        return f"model.layers.{layer}.self_attn.{attn_type}.{param_type}"

    return param_name  # Return original if no pattern matches

def create_transformed_weight_map(original_weight_map):
    """Transforms the weight map keys based on identified patterns."""
    transformed_map = {}
    for param_name, value in original_weight_map.items():
        transformed_name = transform_parameter_name(param_name)
        transformed_map[transformed_name] = value
    return transformed_map

def create_transformed_index_file(original_index_file_path, output_file_path):
  """Creates a new safetensors index JSON file with transformed parameter names."""

  try:
      with open(original_index_file_path, "r") as f:
          original_index = json.load(f)
  except FileNotFoundError:
      print(f"Error: File not found at {original_index_file_path}")
      return
  except json.JSONDecodeError:
      print(f"Error: Invalid JSON format in {original_index_file_path}")
      return

  transformed_weight_map = create_transformed_weight_map(original_index["weight_map"])

  transformed_index = {
      "metadata": original_index["metadata"],
      "weight_map": transformed_weight_map
  }

  with open(output_file_path, "w") as f:
      json.dump(transformed_index, f, indent=2)

  print(f"Transformed index file created at {output_file_path}")

# --- Example usage ---
original_file = "/Users/fredbliss/Storage/HunyuanVideo-PromptRewrite-HF/model.safetensors.index.json"  # Replace with your file
output_file = "transformed_index.json"

create_transformed_index_file(original_file, output_file)