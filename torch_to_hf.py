import argparse
import os
import logging
import zipfile

import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def extract_data_from_zip(zip_filepath, extract_path):
    """Extracts data.pkl from a .bin (zip) file."""
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            for member in zip_ref.namelist():
                if member.endswith('data.pkl'):
                    zip_ref.extract(member, path=extract_path)
                    return os.path.join(extract_path, member)
        logging.warning(f"'data.pkl' not found in {zip_filepath}")
        return None
    except zipfile.BadZipFile:
        logging.error(f"'{zip_filepath}' does not appear to be a valid zip file.")
        return None
    except Exception as e:
        logging.error(f"Error extracting data from {zip_filepath}: {e}")
        return None

def convert_to_safetensors(max_shard_size, local_dir, cleanup):
    """
    Converts a PyTorch .bin model to .safetensors format.

    Args:
        max_shard_size (str): The maximum size of the .safetensors shards.
        local_dir (str): The local directory where the .bin files are located.
        cleanup (bool): Whether to delete the original .bin files after conversion.
    """

    safetensors_dir = os.path.join(os.getcwd(), "safetensors_files")
    extracted_data_dir = os.path.join(os.getcwd(), "extracted_data")  # New directory for extracted data.pkl files
    os.makedirs(safetensors_dir, exist_ok=True)
    os.makedirs(extracted_data_dir, exist_ok=True)

    repo_path = local_dir
    bin_files = [f for f in os.listdir(repo_path) if f.endswith(".bin")]

    for bin_file in tqdm(bin_files, desc="Converting .bin to .safetensors"):
        try:
            bin_filepath = os.path.join(repo_path, bin_file)

            # Extract data.pkl from the .bin (zip) file
            extracted_data_path = extract_data_from_zip(bin_filepath, extracted_data_dir)
            if not extracted_data_path:
                logging.warning(f"Skipping {bin_file} due to missing data.pkl")
                continue

            state_dict = {}
            with safe_open(extracted_data_path, framework="pt", device="cpu") as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)

            safe_filename = bin_file.replace(".bin", ".safetensors")
            safe_filepath = os.path.join(safetensors_dir, safe_filename)
            save_file(state_dict, safe_filepath)

            if cleanup:
                try:
                    os.remove(os.path.join(repo_path, bin_file))
                    os.remove(extracted_data_path)
                    logging.info(f"Removed original .bin file and extracted data: {bin_file}")
                except Exception as e:
                    logging.error(f"Error removing {bin_file} or its extracted data: {e}")

        except Exception as e:
            logging.error(f"Error converting {bin_file}: {e}")

    logging.info("Conversion complete!")

def main():
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch .bin model to .safetensors format."
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="5GB",
        help="The maximum size of the .safetensors shards.",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        required=True,
        help="The local directory where the .bin files are located.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete the original .bin files after conversion.",
    )

    args = parser.parse_args()

    convert_to_safetensors(args.max_shard_size, args.local_dir, args.cleanup)

if __name__ == "__main__":
    main()