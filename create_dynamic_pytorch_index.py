import json
import os
import re
import torch
from tqdm import tqdm

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

    print(f"Found {num_files} .bin files in {model_dir}")

    with tqdm(total=num_files, desc="Scanning for unique parameter names, shapes, and data types", unit="file") as progress_bar:
        for filename in bin_files:
            filepath = os.path.join(model_dir, filename)
            try:
                checkpoint = torch.load(filepath, map_location='cpu')
                for key, value in checkpoint.items():
                    if key not in param_inventory:
                        param_inventory[key] = {
                            "shape": value.shape,
                            "dtype": str(value.dtype),
                            "files": [filename]
                        }
                    else:
                        # Check for consistency
                        if param_inventory[key]["shape"] != value.shape:
                            print(f"  WARNING: Shape mismatch for {key}: {param_inventory[key]['shape']} vs. {value.shape} in {filename}")
                        if param_inventory[key]["dtype"] != str(value.dtype):
                            print(f"  WARNING: Data type mismatch for {key}: {param_inventory[key]['dtype']} vs. {str(value.dtype)} in {filename}")
                        param_inventory[key]["files"].append(filename)
                progress_bar.update(1)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue

    # 2. Analyze Parameter Structure (Programmatic Approach)
    # This is a basic example; you might need more sophisticated logic based on your model
    def infer_structure(param_names):
        patterns = {}
        for name in sorted(param_names):
            parts = name.split(".")
            for i in range(1, len(parts) + 1):
                prefix = ".".join(parts[:i])
                if prefix not in patterns:
                    patterns[prefix] = 0
                patterns[prefix] += 1
        return patterns

    structure = infer_structure(param_inventory.keys())

    print("\nInferred Parameter Structure (Prefixes and Counts):")
    for prefix, count in sorted(structure.items(), key=lambda item: item[1], reverse=True):
        print(f"  {prefix}: {count}")

    # 3. Develop Flexible Renaming Logic
    def rename_parameter(key, structure_info):
        """
        Renames a parameter key based on the inferred structure.

        This is a basic example. You'll likely need more complex logic here.
        """
        new_key = key

        # Example: Remove redundant "model.layers" if it exists
        new_key = re.sub(r"model\.layers\.(\d+)\.model\.layers\.\d+", r"model.layers.\1", new_key)

        # Example: Combine mlp.mlp.experts into mlp.switch_mlp
        new_key = new_key.replace("mlp.mlp.experts", "mlp.switch_mlp")

        # Example: Handle shared_mlp
        new_key = new_key.replace("mlp.shared_mlp", "mlp.shared_expert")
        
        # Example: Handle mlp.gate.wg
        new_key = new_key.replace("mlp.gate.wg", "mlp.gate")

        # Add more renaming rules as needed based on your model's structure

        return new_key

    # 4. weight_map Creation with Shape and Data Type Validation
    weight_map = {}
    total_size = 0

    with tqdm(total=num_files, desc="Processing .bin files, renaming, and validating", unit="file") as progress_bar:
        for filename in bin_files:
            filepath = os.path.join(model_dir, filename)
            try:
                checkpoint = torch.load(filepath, map_location='cpu')
                file_size = os.path.getsize(filepath)
                total_size += file_size
                print(f"\nProcessing: {filename} ({file_size} bytes)")

                for old_key, value in checkpoint.items():
                    new_key = rename_parameter(old_key, structure)

                    # Validation
                    if new_key in param_inventory:
                        expected_shape = param_inventory[new_key]["shape"]
                        expected_dtype = param_inventory[new_key]["dtype"]
                        if value.shape != expected_shape:
                            print(f"  WARNING: Shape mismatch for {new_key}: Expected {expected_shape}, found {value.shape} in {filename}")
                        if str(value.dtype) != expected_dtype:
                            print(f"  WARNING: Data type mismatch for {new_key}: Expected {expected_dtype}, found {str(value.dtype)} in {filename}")
                    else:
                        print(f"  WARNING: {new_key} not found in initial scan.")

                    if new_key != old_key:
                        print(f"  Mapped: {old_key} -> {new_key} (Shape: {value.shape}, Dtype: {value.dtype})")
                    else:
                        print(f"  Loaded: {new_key} (Shape: {value.shape}, Dtype: {value.dtype})")

                    weight_map[new_key] = filename
                progress_bar.update(1)

            except Exception as e:
                print(f"Error loading {filename}: {e}")
                continue
        
        # 5. Generate pytorch_model.bin.index.json
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
model_directory = "/home/fbliss/Storage/HunyuanVideo-PromptRewrite-bin"
create_pytorch_index(model_directory)