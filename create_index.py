#!/usr/bin/env python3

import os
import json
import torch
import argparse
import gc
from pathlib import Path
from tqdm import tqdm

def load_progress(model_dir):
    progress_file = Path(model_dir) / "index_progress.json"
    if progress_file.exists():
        return json.loads(progress_file.read_text())
    return {"weight_map": {}, "metadata": {"total_size": 0}, "processed_shards": []}

def save_progress(model_dir, progress_data):
    progress_file = Path(model_dir) / "index_progress.json"
    progress_file.write_text(json.dumps(progress_data, indent=2))

def process_shard(shard_path):
    state_dict = torch.load(shard_path, map_location="cuda")
    param_info = {}
    total_size = 0
    
    for param_name, tensor in state_dict.items():
        param_info[param_name] = {
            'size': tensor.numel() * tensor.element_size()
        }
        total_size += param_info[param_name]['size']
    
    del state_dict
    torch.cuda.empty_cache()
    gc.collect()
    
    return param_info, total_size

def create_index(model_dir, output_dir=None, batch_size=2):
    model_dir = Path(model_dir)
    if output_dir is None:
        output_dir = model_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load progress
    progress = load_progress(model_dir)
    weight_map = progress["weight_map"]
    metadata = progress["metadata"]
    processed_shards = set(progress["processed_shards"])
    
    # Find shards
    shard_files = sorted([f for f in model_dir.glob("pytorch_model-*.bin")])
    print(f"Found {len(shard_files)} shards")

    # Process remaining shards
    remaining_shards = [f for f in shard_files if f.name not in processed_shards]
    
    with tqdm(total=len(remaining_shards)) as pbar:
        for shard_file in remaining_shards:
            try:
                param_info, total_size = process_shard(shard_file)
                
                # Update index
                for param_name in param_info:
                    weight_map[param_name] = shard_file.name
                metadata["total_size"] += total_size
                
                # Update progress
                processed_shards.add(shard_file.name)
                progress.update({
                    "weight_map": weight_map,
                    "metadata": metadata,
                    "processed_shards": list(processed_shards)
                })
                save_progress(model_dir, progress)
                
                pbar.update(1)
                
            except Exception as e:
                print(f"\nError processing {shard_file}: {e}")
                print("Progress saved - you can resume from this point")
                raise

    # Create final index
    index = {
        "metadata": metadata,
        "weight_map": weight_map
    }
    
    # Save index
    index_path = output_dir / "pytorch_model.bin.index.json"
    index_path.write_text(json.dumps(index, indent=2))
    
    # Clean up progress file
    progress_file = model_dir / "index_progress.json"
    if progress_file.exists():
        progress_file.unlink()
    
    print(f"\nCreated index file with {len(weight_map)} parameters at {index_path}")
    return index

def verify_model(model_dir):
    print("\nVerifying model loads correctly...")
    from transformers import AutoModelForCausalLM
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            device_map="auto"
        )
        print("Model verification successful!")
        return True
    except Exception as e:
        print(f"Model verification failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Create index for sharded HuggingFace model')
    parser.add_argument('model_dir', help='Directory containing model shards')
    parser.add_argument('--output-dir', help='Output directory (defaults to model_dir)')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size for processing')
    parser.add_argument('--skip-verify', action='store_true', help='Skip model verification')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("Warning: No GPU detected - processing will be slow")
    else:
        print(f"GPU detected with {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB memory")

    create_index(args.model_dir, args.output_dir, args.batch_size)

    if not args.skip_verify:
        verify_model(args.output_dir or args.model_dir)

if __name__ == "__main__":
    main()