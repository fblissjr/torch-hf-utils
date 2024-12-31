import os
import json
import torch
from collections import OrderedDict

def load_progress(model_dir):
    """Load existing progress from temporary file if it exists."""
    progress_file = os.path.join(model_dir, "index_progress.json")
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return json.load(f)
    return {"weight_map": {}, "metadata": {"total_size": 0}, "processed_shards": []}

def save_progress(model_dir, progress_data):
    """Save current progress to temporary file."""
    progress_file = os.path.join(model_dir, "index_progress.json")
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)

def create_model_index(model_dir):
    """Create an index mapping parameters to their weight shards with resume capability."""
    
    # Load existing progress if any
    progress = load_progress(model_dir)
    weight_map = progress["weight_map"]
    metadata = progress["metadata"]
    processed_shards = set(progress["processed_shards"])
    
    # Find all shard files
    shard_files = sorted([f for f in os.listdir(model_dir) 
                         if f.startswith('pytorch_model-') and f.endswith('.bin')])
    
    try:
        for shard_file in shard_files:
            # Skip already processed shards
            if shard_file in processed_shards:
                print(f"Skipping already processed {shard_file}")
                continue
                
            print(f"Processing {shard_file}...")
            shard_path = os.path.join(model_dir, shard_file)
            state_dict = torch.load(shard_path, map_location='cpu')
            
            # Record parameters in this shard
            for param_name in state_dict.keys():
                weight_map[param_name] = shard_file
                metadata["total_size"] += state_dict[param_name].numel() * state_dict[param_name].element_size()
            
            # Mark shard as processed and save progress
            processed_shards.add(shard_file)
            progress["weight_map"] = weight_map
            progress["metadata"] = metadata
            progress["processed_shards"] = list(processed_shards)
            save_progress(model_dir, progress)
            
            # Free memory
            del state_dict
            torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print("Progress saved. You can resume from the last successful shard.")
        raise
    
    # Create final index file
    index = {
        "metadata": metadata,
        "weight_map": weight_map
    }
    
    # Save the final index file
    with open(os.path.join(model_dir, "pytorch_model.bin.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    
    # Clean up progress file
    progress_file = os.path.join(model_dir, "index_progress.json")
    if os.path.exists(progress_file):
        os.remove(progress_file)
    
    print(f"Created index file mapping {len(weight_map)} parameters")
    return index

if __name__ == "__main__":
    model_dir = "/workspace/hytorch"
    index = create_model_index(model_dir)
    
    print("Loading model to verify index...")
    from transformers import AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True
    )
    
    print("Saving converted model...")
    model.save_pretrained(
        "/workspace/hyconverted", 
        safe_serialization=True
    )