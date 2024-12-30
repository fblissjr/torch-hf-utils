import argparse
import os
import logging

import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def convert_and_upload(repo_id, max_shard_size, upload_to_hub, local_dir=None, cleanup=False):
    """
    Converts a PyTorch .bin model to .safetensors format.

    Args:
        repo_id (str): The ID of the Hugging Face repository.
        max_shard_size (str): The maximum size of the .safetensors shards.
        upload_to_hub (bool): Whether to upload to the Hugging Face Hub.
        local_dir (str): The local directory where the .bin files are located.
        cleanup (bool): Whether to delete the original .bin files after conversion.
    """

    safetensors_dir = os.path.join(os.getcwd(), "safetensors_files")
    os.makedirs(safetensors_dir, exist_ok=True)

    if local_dir:
        repo_path = local_dir
        bin_files = [f for f in os.listdir(repo_path) if f.endswith(".bin")]
    else:
        repo_path = os.path.join(os.getcwd(), "bin_files")
        os.makedirs(repo_path, exist_ok=True)
        logging.info(f"Downloading .bin files from {repo_id} to {repo_path}...")
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc="Downloading") as progress_bar:
            snapshot_download(
                repo_id,
                local_dir=repo_path,
                allow_patterns=["*.bin"],
                local_dir_use_symlinks=False,
                tqdm_class=lambda total, desc: progress_bar,
            )

        api = HfApi()
        repo_files = api.list_repo_files(repo_id)
        bin_files = [f for f in repo_files if f.endswith(".bin")]

    for bin_file in tqdm(bin_files, desc="Converting .bin to .safetensors"):
        try:
            state_dict = {}
            with safe_open(os.path.join(repo_path, bin_file), framework="pt", device="cpu") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)

            safe_filename = bin_file.replace(".bin", ".safetensors")
            safe_filepath = os.path.join(safetensors_dir, safe_filename)
            save_file(state_dict, safe_filepath)

            if upload_to_hub:
                with tqdm(
                    total=os.path.getsize(safe_filepath),
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Uploading {safe_filename}",
                ) as progress_bar:
                    api.upload_file(
                        path_or_fileobj=safe_filepath,
                        path_in_repo=safe_filename,
                        repo_id=repo_id,
                        repo_type="model",
                        tqdm_class=lambda total, desc: progress_bar,
                    )
                logging.info(f"Uploaded {safe_filename} to {repo_id}")

            if cleanup:
                try:
                    os.remove(os.path.join(repo_path, bin_file))
                    logging.info(f"Removed original .bin file: {bin_file}")
                except Exception as e:
                    logging.error(f"Error removing {bin_file}: {e}")

        except Exception as e:
            logging.error(f"Error converting {bin_file}: {e}")

    logging.info("Conversion complete!")

def main():
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch .bin model to .safetensors format."
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="The ID of the Hugging Face repository.",
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="5GB",
        help="The maximum size of the .safetensors shards.",
    )
    parser.add_argument(
        "--no_upload",
        action="store_false",
        dest="upload_to_hub",
        help="Skip uploading the converted files to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="The local directory where the .bin files are located.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete the original .bin files after conversion.",
    )

    args = parser.parse_args()

    convert_and_upload(args.repo_id, args.max_shard_size, args.upload_to_hub, args.local_dir, args.cleanup)

if __name__ == "__main__":
    main()
