import torch
import os
import json
import re
from tqdm import tqdm
import argparse

def create_pytorch_index(model_dir, output_file="pytorch_model.bin.index.json"):
    """
    Creates a pytorch_model.bin.index.json file for a sharded PyTorch model.

    Args:
        model_dir: The directory containing the .bin files.
        output_file: The path to save the index JSON file.
    """

    # 1. Comprehensive Parameter Discovery
    param_inventory = {}  # Store parameter names, shapes, data types, and originating files
    bin_files = [f for f in os.listdir(model_dir) if f.endswith(".bin")]
    num_files = len(bin_files)
    total_size = 0

    print(f"Found {num_files} .bin files in {model_dir}")

    with tqdm(total=num_files, desc="Scanning for unique parameter names, shapes, and data types", unit="file") as progress_bar:
        for filename in bin_files:
            filepath = os.path.join(model_dir, filename)
            try:
                checkpoint = torch.load(filepath, map_location='cpu')
                file_size = os.path.getsize(filepath)
                total_size += file_size

                for key, value in checkpoint.items():
                    if key not in param_inventory:
                        param_inventory[key] = {
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                            "files": [filename],
                            "all_zeros": all(v == 0 for v in value.flatten().tolist()) if "bias" in key else False # Check if bias is all zeros
                        }
                    else:
                        # Check for consistency
                        if param_inventory[key]["shape"] != list(value.shape):
                            print(f"  WARNING: Shape mismatch for {key}: {param_inventory[key]['shape']} vs. {list(value.shape)} in {filename}")
                        if param_inventory[key]["dtype"] != str(value.dtype):
                            print(f"  WARNING: Data type mismatch for {key}: {param_inventory[key]['dtype']} vs. {str(value.dtype)} in {filename}")
                        param_inventory[key]["files"].append(filename)
                progress_bar.update(1)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

    # Categorize parameters, including the new 'zero_bias' category
    categories = {
        "experts": [],
        "shared_expert": [],
        "self_attn": [],
        "layernorm": [],
        "embed_norm": [],
        "gate_wg": [],
        "zero_bias": [],
        "other": [],
    }

    for name, info in param_inventory.items():
        if "mlp.experts" in name:
            categories["experts"].append((name, info))
        elif "mlp.shared_mlp" in name:
            categories["shared_expert"].append((name, info))
        elif "self_attn" in name:
            categories["self_attn"].append((name, info))
        elif "layernorm" in name:
            categories["layernorm"].append((name, info))
        elif "embed" in name or "norm" in name:
            categories["embed_norm"].append((name, info))
        elif "mlp.gate.wg" in name:
            categories["gate_wg"].append((name, info))
        elif "bias" in name and info["all_zeros"]:
             categories["zero_bias"].append((name, info))
        else:
            categories["other"].append((name, info))

    # Print the categorized parameters for inspection
    print("\nParameter Categories:")
    for category, param_list in categories.items():
        print(f"\n{category.upper()}:")
        for name, info in param_list:
           print(f"  {name} (Shape: {info['shape']}, Dtype: {info['dtype']}, All Zeros: {info['all_zeros']}, Files: {', '.join(info['files'])})")

    # Create weight map from param_inventory
    weight_map = {}
    for param_name, info in param_inventory.items():
        # Use the first file that contains this parameter
        weight_map[param_name] = info["files"][0]

    # Generate pytorch_model.bin.index.json
    index = {
        "metadata": {
            "total_size": total_size,
        },
        "weight_map": weight_map
    }

    with open(output_file, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nIndex file created: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the model parameters.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the directory with the PyTorch .bin model files.")
    args = parser.parse_args()

    create_pytorch_index(args.model_dir)