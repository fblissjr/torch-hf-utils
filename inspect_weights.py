import os
import json
import argparse
from collections import defaultdict
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def load_index_file(model_path):
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    try:
        with open(index_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(
            f"Index file not found at {index_file}. "
            "Proceeding without it, but this may not be a sharded model."
        )
        return None

def inspect_model_weights(model_path):
    index_data = load_index_file(model_path)
    all_weights = {}

    if index_data:
        # Sharded model
        weight_map = index_data["weight_map"]
        for weight_name, file_name in weight_map.items():
            file_path = os.path.join(model_path, file_name)
            try:
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    if weight_name in f.keys():
                        all_weights[weight_name] = f.get_tensor(weight_name)
                    else:
                        print(f"Weight {weight_name} not found in file {file_name}")
            except Exception as e:
                print(f"Error opening or reading file {file_path}: {e}")

        # Check for additional safetensor files not in the index
        for file_name in os.listdir(model_path):
            if file_name.endswith(".safetensors") and file_name not in set(
                weight_map.values()
            ):
                file_path = os.path.join(model_path, file_name)
                try:
                    with safe_open(file_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            if key not in all_weights:  # Avoid overwriting
                                all_weights[key] = f.get_tensor(key)
                                print(f"Loaded additional weight from {file_name}: {key}")
                except Exception as e:
                    print(f"Error opening or reading file {file_path}: {e}")

    else:
        # Non-sharded model (single .safetensors file)
        for file_name in os.listdir(model_path):
            if file_name.endswith(".safetensors"):
                file_path = os.path.join(model_path, file_name)
                try:
                    with safe_open(file_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            all_weights[key] = f.get_tensor(key)
                except Exception as e:
                    print(f"Error opening or reading file {file_path}: {e}")

    return all_weights

def summarize_model_structure(weights):
    layer_counts = defaultdict(int)
    other_keys = []
    layer_details = defaultdict(list)

    for key in weights.keys():
        if key.startswith("model.layers."):
            parts = key.split(".")
            if len(parts) > 2:
                layer_num = int(parts[2])
                layer_counts[layer_num] += 1
                layer_details[layer_num].append(key)
        else:
            other_keys.append(key)

    return dict(layer_counts), other_keys, layer_details

def tie_weights(weights, output_path):
    if "lm_head.weight" not in weights and "model.embed_tokens.weight" in weights:
        print("Adding lm_head.weight by tying it to model.embed_tokens.weight")
        weights["lm_head.weight"] = weights["model.embed_tokens.weight"]

        index_file_path = os.path.join(output_path, "model.safetensors.index.json")
        if os.path.exists(index_file_path):
            # Update the index file if it exists
            index_data = load_index_file(output_path)
            weight_map = index_data["weight_map"]

            # Find a safetensors file to add the new weight to
            if weight_map:
                last_file = max(set(weight_map.values()))
                weight_map["lm_head.weight"] = last_file

                # Save updated index file
                with open(index_file_path, "w") as f:
                    json.dump(index_data, f, indent=2)
            else:
                print(
                    "Warning: model.safetensors.index.json is not found. Cannot update weight map."
                )
        else:
            print(
                "Warning: model.safetensors.index.json not found. Cannot update weight map."
            )

        # Save updated weights - this will overwrite existing files if not careful
        save_file(weights, os.path.join(output_path, "tied_model.safetensors"))

    return weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect and summarize Hugging Face Transformers model weights"
    )
    parser.add_argument(
        "model_folder",
        type=str,
        help="Path to the model weights folder (or HF cache directory)",
    )
    parser.add_argument(
        "--tie_weights",
        action="store_true",
        help="Tie weights if lm_head is missing",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for tied weights",
        default=None
    )
    parser.add_argument(
        "--layer_num",
        type=int,
        default=None,
        help="Layer number to inspect",
    )
    args = parser.parse_args()

    model_folder = args.model_folder

    try:
        weights = inspect_model_weights(model_folder)

        if args.tie_weights:
            if args.output is None:
                raise ValueError(
                    "Please specify an output path when using --tie_weights"
                )
            weights = tie_weights(weights, args.output)

        print(f"Total number of keys: {len(weights)}")

        layer_counts, other_keys, layer_details = summarize_model_structure(weights)

        print("\nLayer structure:")
        for layer, count in sorted(layer_counts.items()):
            print(f"Layer {layer}: {count} keys")

        if args.layer_num is not None:
            layer_num = args.layer_num
            print(f"\nDetails for layer {layer_num}:")
            if layer_num in layer_details:
                for key in sorted(layer_details[layer_num]):
                    print(f"- {key}: {weights[key].shape}")
            else:
                print(f"Layer {layer_num} not found in the model.")

        print(f"\nNumber of non-layer keys: {len(other_keys)}")
        print("Non-layer keys:")
        for key in other_keys:
            print(f"- {key}: {weights[key].shape}")

        print("\nSample of layer keys:")
        layer_keys = [key for key in weights.keys() if key.startswith("model.layers.")]
        for key in sorted(layer_keys)[:10]:  # Print first 10 layer keys as a sample
            print(f"- {key}: {weights[key].shape}")

        # Print all parameters
        print("\nAll model parameters:")
        for key, value in weights.items():
            print(f"- {key}: {value.shape}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")