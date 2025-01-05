import torch
import os
import json
import re
from tqdm import tqdm
import argparse

def discover_parameters(model_dir, output_file="param_info.json"):
    """
    Discovers parameters in the original PyTorch model.

    Args:
        model_dir: Directory containing the PyTorch .bin files.
        output_file: File to save the parameter information to (JSON).
    """

    param_info = {}
    bin_files = [f for f in os.listdir(model_dir) if f.endswith(".bin")]
    num_files = len(bin_files)

    print(f"Found {num_files} .bin files in {model_dir}")

    with tqdm(total=num_files, desc="Scanning parameters", unit="file") as progress_bar:
        for filename in bin_files:
            filepath = os.path.join(model_dir, filename)
            try:
                checkpoint = torch.load(filepath, map_location="cpu")
                for key, value in checkpoint.items():
                    if key not in param_info:
                        param_info[key] = {
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                            "files": [filename],
                        }
                    else:
                        if list(value.shape) != param_info[key]["shape"]:
                            print(
                                f"  WARNING: Shape mismatch for {key}: {param_info[key]['shape']} vs. {list(value.shape)} in {filename}"
                            )
                        if str(value.dtype) != param_info[key]["dtype"]:
                            print(
                                f"  WARNING: Data type mismatch for {key}: {param_info[key]['dtype']} vs. {str(value.dtype)} in {filename}"
                            )
                        param_info[key]["files"].append(filename)
                progress_bar.update(1)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

    # Categorize parameters (add more categories as needed)
    categories = {
        "experts": [],
        "shared_expert": [],
        "self_attn": [],
        "layernorm": [],
        "embed_norm": [],
        "gate_wg": [],
        "shared_expert_gate": [],
        "other": [],
    }
    for name, info in param_info.items():
        if "mlp.mlp.experts" in name:
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
        elif "mlp.shared_expert_gate" in name:
            categories["shared_expert_gate"].append((name, info))
        else:
            categories["other"].append((name, info))

    # Print the categorized parameters for inspection
    print("\nParameter Categories:")
    for category, param_list in categories.items():
        print(f"\n{category.upper()}:")
        for name, info in param_list:
            print(
                f"  {name} (Shape: {info['shape']}, Dtype: {info['dtype']}) (Files: {', '.join(info['files'])})"
            )

    # Save parameter info to a JSON file
    with open(output_file, "w") as f:
        json.dump(param_info, f, indent=2)

    print(f"\nParameter information saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Discover and analyze parameters in a PyTorch model."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing the PyTorch .bin files.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="param_info.json",
        help="Output JSON file to store parameter information.",
    )
    args = parser.parse_args()
    discover_parameters(args.model_dir, args.output_file)