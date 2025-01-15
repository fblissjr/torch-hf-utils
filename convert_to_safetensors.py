import json
import os
import re
import torch
import concurrent.futures
from safetensors.torch import save_file
from tqdm import tqdm
from pathlib import Path
import logging
import ijson
import gc
import argparse

def setup_logging(output_dir):
    """Sets up logging to file and console."""
    log_file = os.path.join(output_dir, "conversion.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def get_weight_map(index_file_path):
    """Load and return the weight mapping from index file."""
    try:
        with open(index_file_path, 'r') as f:
            index_data = json.load(f)
            return index_data.get('weight_map', {})
    except Exception as e:
        logging.error(f"Failed to load index file: {e}")
        return {}

def filter_keys(d: dict):
  """
  Filters a dictionary so that any keys with the substring "bias"
  are removed entirely
  """
  return {k: v for k, v in d.items() if "bias" not in k}

def convert_single_shard(bin_file_path, output_safetensors_dir, shard_index, num_shards):
    """Convert a single .bin shard to .safetensors format."""
    try:
        safetensors_file_name = f"model-{shard_index:05d}-of-{num_shards:05d}.safetensors"
        safetensors_file_path = os.path.join(output_safetensors_dir, safetensors_file_name)

        # Skip if already converted
        if os.path.exists(safetensors_file_path):
            logging.info(f"Skipping existing file: {safetensors_file_name}")
            return safetensors_file_name, None

        # Load checkpoint with mmap
        loading_kwargs = {
            'map_location': 'cpu',
            'mmap': True
        }
        checkpoint = torch.load(bin_file_path, **loading_kwargs)

        # Convert tensors directly without moving to GPU
        converted_dict = {}
        for key, tensor in checkpoint.items():
            # Convert to bfloat16 only for appropriate tensor types
            if tensor.dtype in [torch.float32, torch.float16]:
                tensor = tensor.to(torch.bfloat16)
            converted_dict[key] = tensor.cpu().detach()

        # Save to safetensors
        save_file(filter_keys(converted_dict), safetensors_file_path, metadata={"format": "pt"})

        # Clean up
        del checkpoint
        del converted_dict
        gc.collect()
        torch.cuda.empty_cache()

        logging.info(f"[SUCCESS] Converted {bin_file_path} to {safetensors_file_path}")
        return safetensors_file_name, None

    except Exception as e:
        logging.error(f"[ERROR] Failed to convert {bin_file_path}: {str(e)}")
        return None, str(e)

def convert_to_safetensors(pytorch_model_dir, output_safetensors_dir):
    """Convert sharded PyTorch model to SafeTensors format."""
    setup_logging(output_safetensors_dir)
    logging.info("Starting conversion process...")

    # Load weight map
    index_file_path = os.path.join(pytorch_model_dir, "pytorch_model.bin.index.json")
    weight_map = get_weight_map(index_file_path)

    if not weight_map:
        logging.error("Failed to load weight map. Exiting.")
        return

    # Get list of bin files
    bin_files = sorted([
        f for f in os.listdir(pytorch_model_dir)
        if f.endswith(".bin") and not f.endswith("index.json")
    ], key=lambda x: int(re.search(r'(\d+)', x).group(1)))

    # Use a thread pool for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(convert_single_shard, os.path.join(pytorch_model_dir, bin_file), output_safetensors_dir, i + 1, len(bin_files)) for i, bin_file in enumerate(bin_files)]
        for future in concurrent.futures.as_completed(futures):
            result, error = future.result()
            if error:
                logging.error(f"Error during conversion: {error}")

    # Create the model index file
    with open(index_file_path, 'r') as f:
        index_data = json.load(f)

    weight_map = {}
    for k, v in index_data["weight_map"].items():
      # the weight map in the pt index file is
      # structured in a way we cannot use it
      # so let's just remove it
      if 'bias' not in k:
        shard_file_name = re.sub(r"-(\d+).bin", r"-{\g<1>}-of-" + str(len(bin_files)).zfill(5) + ".safetensors", v)
        weight_map[k] = shard_file_name

    # add the metadata
    metadata = {
        "format": "pt+mlx",
        "total_size": 0,
        "num_shards": len(bin_files)
    }
    for filename in os.listdir(output_safetensors_dir):
      if ".safetensors" in filename:
        metadata["total_size"] = metadata["total_size"] + os.path.getsize(
                os.path.join(output_safetensors_dir, filename)
        )
    
    output_index_data = {
        "metadata": metadata,
        "weight_map": weight_map
    }

    output_index_file = os.path.join(output_safetensors_dir, "model.safetensors.index.json")

    with open(output_index_file, "w") as f:
        json.dump(output_index_data, f, indent=4)
    logging.info(f"Saved model index to {output_index_file}")

    logging.info("Conversion process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert sharded PyTorch model weights to SafeTensors format."
    )
    parser.add_argument(
        "--pytorch-path",
        type=str,
        required=True,
        help="Path to the directory containing PyTorch .bin files."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the output SafeTensors files."
    )
    args = parser.parse_args()

    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    convert_to_safetensors(args.pytorch_path, str(output_dir))