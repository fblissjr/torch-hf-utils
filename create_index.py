import os
import json
import torch
from collections import OrderedDict

def create_model_index(model_dir):
    """Create an index mapping parameters to their weight shards."""
    
    # Find all the model shard files
    shard_files = [f for f in os.listdir(model_dir) if f.startswith('pytorch_model-') and f.endswith('.bin')]
    
    # Create weight map
    weight_map = {}
    metadata = {"total_size": 0}
    
    for shard_file in sorted(shard_files):
        print(f"Processing {shard_file}...")
        shard_path = os.path.join(model_dir, shard_file)
        state_dict = torch.load(shard_path, map_location='cpu')
        
        # Record which parameters are in this shard
        for param_name in state_dict.keys():
            weight_map[param_name] = shard_file
            metadata["total_size"] += state_dict[param_name].numel() * state_dict[param_name].element_size()
            
    # Create the index dictionary
    index = {
        "metadata": metadata,
        "weight_map": weight_map
    }
    
    # Save the index file
    with open(os.path.join(model_dir, "pytorch_model.bin.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    
    print(f"Created index file mapping {len(weight_map)} parameters")
    return index

# Use the model directory containing the sharded .bin files
model_dir = "/workspace/hytorch"
index = create_model_index(model_dir)