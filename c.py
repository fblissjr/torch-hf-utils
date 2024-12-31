#!/usr/bin/env python3

import os
import json
import torch
import argparse
import safetensors
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file
from typing import Dict, Any

class ResumableSafetensorConverter:
    def __init__(self, input_dir: str, output_dir: str, cache_dir: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        
        # Create all necessary directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup offload directories within cache directory
        self.offload_folder = self.cache_dir / "model_offload"
        self.offload_folder.mkdir(exist_ok=True)
        
        self.progress_file = self.output_dir / "conversion_progress.json"
        
    def load_progress(self) -> Dict[str, Any]:
        """Load conversion progress if it exists."""
        if self.progress_file.exists():
            return json.loads(self.progress_file.read_text())
        return {
            "converted_tensors": [],
            "total_tensors": 0,
            "current_shard": 0
        }
    
    def save_progress(self, progress: Dict[str, Any]):
        """Save conversion progress."""
        self.progress_file.write_text(json.dumps(progress, indent=2))

    def convert_to_safetensors(self, shard_size: int = 8_000_000_000):
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.input_dir,
            trust_remote_code=True,
            device_map="auto",
            offload_folder=str(self.offload_folder),
            cache_dir=str(self.cache_dir),
            offload_buffers=True,
            torch_dtype=torch.bfloat16  # Use FP16 to reduce memory usage
        )
        
        # Load existing progress
        progress = self.load_progress()
        converted_tensors = set(progress["converted_tensors"])
        
        # Get state dict
        state_dict = model.state_dict()
        
        if progress["total_tensors"] == 0:
            progress["total_tensors"] = len(state_dict)
            self.save_progress(progress)
        
        print(f"Total tensors to convert: {progress['total_tensors']}")
        print(f"Already converted: {len(converted_tensors)}")
        
        # Group parameters into shards
        current_shard = {}
        current_shard_size = 0
        shard_idx = progress["current_shard"]
        
        # Create iterator over remaining tensors
        remaining_tensors = {k: v for k, v in state_dict.items() if k not in converted_tensors}
        
        try:
            with tqdm(total=len(remaining_tensors)) as pbar:
                for key, tensor in remaining_tensors.items():
                    tensor_size = tensor.numel() * tensor.element_size()
                    
                    # If adding this tensor would exceed shard size, save current shard
                    if current_shard_size + tensor_size > shard_size and current_shard:
                        shard_file = self.output_dir / f"model-{shard_idx:05d}-of-{len(state_dict):05d}.safetensors"
                        save_file(current_shard, str(shard_file))
                        shard_idx += 1
                        current_shard = {}
                        current_shard_size = 0
                        
                        # Clear CUDA cache after saving each shard
                        torch.cuda.empty_cache()
                    
                    # Add tensor to current shard
                    current_shard[key] = tensor.cpu()
                    current_shard_size += tensor_size
                    
                    # Update progress
                    converted_tensors.add(key)
                    progress.update({
                        "converted_tensors": list(converted_tensors),
                        "current_shard": shard_idx
                    })
                    self.save_progress(progress)
                    
                    pbar.update(1)
                
                # Save final shard if anything remains
                if current_shard:
                    shard_file = self.output_dir / f"model-{shard_idx:05d}-of-{len(state_dict):05d}.safetensors"
                    save_file(current_shard, str(shard_file))
            
            # Clean up progress file after successful conversion
            if self.progress_file.exists():
                self.progress_file.unlink()
            
            print("\nConversion completed successfully!")
            
        except Exception as e:
            print(f"\nError during conversion: {e}")
            print("Progress has been saved - you can resume from this point")
            raise
        finally:
            # Clean up cache and offload directories
            if self.offload_folder.exists():
                import shutil
                shutil.rmtree(str(self.offload_folder))
                
def convert_model(input_dir: str, output_dir: str, cache_dir: str):
    print("Starting conversion process...")
    
    # Setup directories
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir)
    offload_folder = cache_dir / "offload"
    offload_folder.mkdir(parents=True, exist_ok=True)
    
    # Load model with optimized settings for large models
    model = AutoModelForCausalLM.from_pretrained(
        input_dir,
        trust_remote_code=True,
        device_map="auto",
        offload_buffers=True,
        offload_folder=str(offload_folder),
        max_memory={0: "20GB", "cpu": "50GB"},
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    
    # Get state dict and prepare for conversion
    print("Getting state dict...")
    state_dict = model.state_dict()
    num_tensors = len(state_dict)
    
    print(f"Converting {num_tensors} tensors to safetensors format...")
    out_file = output_dir / "model.safetensors"
    
    try:
        tensors_tqdm = tqdm(state_dict.items(), total=num_tensors)
        state_dict = {k: v.cpu() for k, v in tensors_tqdm}
        print("\nSaving safetensors file...")
        save_file(state_dict, str(out_file))
        print(f"Successfully saved to {out_file}")
    except Exception as e:
        print(f"Error during conversion: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Convert model to safetensors with resume capability')
    parser.add_argument('input_dir', help='Directory containing the input model')
    parser.add_argument('output_dir', help='Directory to save converted safetensors')
    parser.add_argument('--cache-dir', default='/workspace/model_cache',
                        help='Directory for cache and offloading (default: /workspace/model_cache)')
    parser.add_argument('--shard-size', type=int, default=8_000_000_000, 
                        help='Maximum size of each shard in bytes (default: 8GB)')
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        print(f"GPU detected with {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB memory")
    else:
        print("Warning: No GPU detected - conversion may be slow")
    
    converter = ResumableSafetensorConverter(args.input_dir, args.output_dir, args.cache_dir)
    converter.convert_to_safetensors(args.shard_size)

if __name__ == "__main__":
    main()