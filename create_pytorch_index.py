import json
import os
import torch
from tqdm import tqdm

def create_pytorch_index(model_dir, output_file="pytorch_model.bin.index.json"):
    """
    Creates a pytorch_model.bin.index.json file for a sharded PyTorch model.

    Args:
        model_dir: The directory containing the .bin files.
        output_file: The path to save the index JSON file.
    """

    weight_map = {}
    total_size = 0
    bin_files = [f for f in os.listdir(model_dir) if f.endswith(".bin")]
    num_files = len(bin_files)

    print(f"Found {num_files} .bin files in {model_dir}")

    with tqdm(total=num_files, desc="Processing .bin files", unit="file") as progress_bar:
        for filename in bin_files:
            filepath = os.path.join(model_dir, filename)
            try:
                # Use 'cpu' to ensure tensors are loaded to CPU memory
                checkpoint = torch.load(filepath, map_location='cpu')
                file_size = os.path.getsize(filepath)
                total_size += file_size
                print(f"\nProcessing: {filename} ({file_size} bytes)")

                for key, value in checkpoint.items():
                    # Handle gate.wg and shared_expert_gate parameters
                    original_key = key
                    if "mlp.mlp.experts" in key:
                        key = key.replace("mlp.mlp.experts", "mlp.switch_mlp")
                    if "mlp.shared_mlp." in key:
                        key = key.replace("mlp.shared_mlp", "mlp.shared_expert")
                    if "mlp.gate.wg" in key:
                        key = key.replace("mlp.gate.wg", "mlp.gate")
                    if "mlp.shared_expert_gate" in key:
                        key = key.replace("mlp.shared_expert_gate", "mlp.shared_expert_gate")

                    if key != original_key:
                        print(f"  Mapped: {original_key} -> {key}")
                    else:
                        print(f"  Loaded: {key}")

                    weight_map[key] = filename

                progress_bar.update(1)

            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

    # Add extra debugging for gate.wg and shared_expert_gate
    print("\nWeight Map Debug:")
    for key in weight_map:
        if "mlp.gate" in key or "mlp.shared_expert_gate" in key:
            print(f"  {key}: {weight_map[key]}")

    index = {
        "metadata": {
            "total_size": total_size,
        },
        "weight_map": weight_map
    }

    with open(output_file, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nIndex file created: {output_file}")

# Example usage:
model_directory = "/Users/fbliss/Storage/HunyuanVideo-PromptRewrite-bin"  # Replace with your model directory
create_pytorch_index(model_directory)