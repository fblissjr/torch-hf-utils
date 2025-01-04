import json
import re
import mlx.core as mx
import argparse
import os
import shutil
from datetime import datetime

def combine_weights(weights, layer_prefix, layer_name, num_experts):
    """
    Combines the weights and biases for a given layer across all experts.

    Args:
        weights (dict): The dictionary containing the model weights.
        layer_prefix (str): The prefix of the layer name (e.g., "model.layers.0").
        layer_name (str): The name of the layer (e.g., "up", "down").
        num_experts (int): The number of experts.
    """
    to_combine = []
    for e in range(num_experts):
        weight_key = f"{layer_prefix}.mlp.experts.{e}.{layer_name}.weight"
        if weight_key in weights:
            to_combine.append(weights.pop(weight_key))

    if to_combine:
        combined_weight = mx.concatenate(to_combine, axis=0)
        weights[f"{layer_prefix}.mlp.switch_mlp.{layer_name}.weight"] = combined_weight

    # Check for biases
    to_combine_bias = []
    for e in range(num_experts):
        bias_key = f"{layer_prefix}.mlp.experts.{e}.{layer_name}.bias"
        if bias_key in weights:
            to_combine_bias.append(weights.pop(bias_key))
    if to_combine_bias:
        combined_bias = mx.concatenate(to_combine_bias, axis=0)
        weights[f"{layer_prefix}.mlp.switch_mlp.{layer_name}.bias"] = combined_bias

def transform_parameter_name(param_name):
    """
    Transforms parameter names to match the MLX Hunyuan model structure.

    Args:
        param_name (str): The original parameter name.

    Returns:
        str: The transformed parameter name.
    """

    # Pattern for MoE expert parameters
    moe_pattern = r"model\.layers\.(\d+)\.model\.layers\.\d+\.mlp\.mlp\.experts\.(\d+)\.(up_proj|down_proj|gate_proj)\.(weight|bias)"
    match = re.match(moe_pattern, param_name)
    if match:
        layer, expert, proj_type, param_type = match.groups()
        # Modify the structure to match mlp.experts.0.up.weight
        if proj_type == 'gate_proj':
            proj_type = 'gate'
        else:
            proj_type = proj_type.replace("_proj", "")
        return f"model.layers.{layer}.mlp.experts.{expert}.{proj_type}.{param_type}"

    # Pattern for shared MLP parameters
    shared_moe_pattern = r"model\.layers\.(\d+)\.model\.layers\.\d+\.mlp\.shared_mlp\.(gate_proj|down_proj|up_proj)\.(weight|bias)"
    match = re.match(shared_moe_pattern, param_name)
    if match:
        layer, proj_type, param_type = match.groups()
        # Modify the structure to match mlp.gate.weight, etc.
        proj_type = proj_type.replace("_proj", "")
        # The shared mlp should map to experts.0
        return f"model.layers.{layer}.mlp.experts.0.{proj_type}.{param_type}" 

    # Pattern for gate parameters
    gate_pattern = r"model\.layers\.(\d+)\.model\.layers\.\d+\.mlp\.gate\.wg\.(weight|bias)"
    match = re.match(gate_pattern, param_name)
    if match:
        layer, param_type = match.groups()
        # Modify the structure to match mlp.gate.wg.weight or mlp.gate.wg.bias
        return f"model.layers.{layer}.mlp.gate.wg.{param_type}"

    # Pattern for self-attention parameters
    self_attn_pattern = r"model\.layers\.(\d+)\.model\.layers\.\d+\.self_attn\.(query_proj|key_layernorm|value_proj|o_proj|query_layernorm|key_proj|v_proj|q_proj)\.(weight|bias)"
    match = re.match(self_attn_pattern, param_name)
    if match:
        layer, attn_type, _, param_type = match.groups()
        # Ensure attn_type is not captured as _ in the return value.
        return f"model.layers.{layer}.self_attn.{attn_type}.{param_type}"

    # Pattern for input layernorm, post attention layernorm weights
    layer_norm_pattern = r"model\.layers\.(\d+)\.(input_layernorm|post_attention_layernorm)\.weight"
    match = re.match(layer_norm_pattern, param_name)
    if match:
        layer, norm_type = match.groups()
        return f"model.layers.{layer}.{norm_type}.weight"

    # Pattern for attention bias parameters
    attn_bias_pattern = r"model\.layers\.(\d+)\.(self_attn\.(?:q|k|v)_proj)\.bias"
    match = re.match(attn_bias_pattern, param_name)
    if match:
        layer, proj_type = match.groups()
        return f"model.layers.{layer}.{proj_type}.bias"

    # Pattern for embedding and norm weights
    non_layer_pattern = r"model\.(embed_tokens|norm)\.(weight)"
    match = re.match(non_layer_pattern, param_name)
    if match:
        component, weight_type = match.groups()
        return f"model.{component}.{weight_type}"

    # Pattern for shared expert gate parameters
    shared_expert_gate_pattern = r"model\.layers\.(\d+)\.model\.layers\.\d+\.mlp\.shared_expert_gate\.(weight|bias)"
    match = re.match(shared_expert_gate_pattern, param_name)
    if match:
        layer, param_type = match.groups()
        return f"model.layers.{layer}.mlp.shared_expert_gate.{param_type}"

    return param_name  # Return original if no pattern matches

def create_transformed_weight_map(original_weight_map):
    """
    Transforms the weight map keys based on identified patterns.

    Args:
        original_weight_map (dict): The original weight map.

    Returns:
        dict: The transformed weight map.
    """
    transformed_map = {}
    for param_name, value in original_weight_map.items():
        transformed_name = transform_parameter_name(param_name)
        transformed_map[transformed_name] = value
    return transformed_map

def create_transformed_index_file(original_index_file_path, output_file_path):
    """
    Creates a new safetensors index JSON file with transformed parameter names.

    Args:
        original_index_file_path (str): Path to the original index file.
        output_file_path (str): Path to save the transformed index file.
    """

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

def main():
    parser = argparse.ArgumentParser(description="Transform a safetensors index file for MLX Hunyuan model.")
    parser.add_argument("original_file", type=str, help="Path to the original safetensors index file.")
    args = parser.parse_args()

    original_file = args.original_file
    model_dir = os.path.dirname(original_file)
    versions_dir = os.path.join(model_dir, "versions")
    output_file = original_file

    # Create versions directory if it doesn't exist
    os.makedirs(versions_dir, exist_ok=True)

    # Create a timestamped backup of the original file
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_file = os.path.join(versions_dir, f"backup_model.safetensors.index_{timestamp}.json")
    shutil.copy2(original_file, backup_file)
    print(f"Backup created: {backup_file}")

    # Create the transformed index file, overwriting the original
    create_transformed_index_file(original_file, output_file)

if __name__ == "__main__":
    main()