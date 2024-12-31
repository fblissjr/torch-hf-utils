import argparse
import os
import logging
import zipfile
import json

import typing Optional
import torch

from huggingface_hub import CommitOperationAdd, HfApi
from safetensors.torch import _find_shared_tensors, _is_complete, save_file
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

COMMIT_DESCRIPTION = """
This is an automated PR created with a modified script

This new file is equivalent to `pytorch_model.bin` but safe in the sense that
no arbitrary code can be put into it.

These files also happen to load much faster than their pytorch counterpart:
https://colab.research.google.com/github/huggingface/notebooks/blob/main/safetensors_doc/en/speed.ipynb

The widgets on your model page will run using this model even if this is not merged
making sure the file actually works.

If you find any issues: please report here: https://github.com/anotheruser/hf-conversion-utils

Feel free to ignore this PR.
"""

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

def _remove_duplicate_names(
    state_dict: Dict[str, torch.Tensor],
    *,
    preferred_names: List[str] = None,
    discard_names: List[str] = None,
) -> Dict[str, List[str]]:
    if preferred_names is None:
        preferred_names = []
    preferred_names = set(preferred_names)
    if discard_names is None:
        discard_names = []
    discard_names = set(discard_names)

    shareds = _find_shared_tensors(state_dict)
    to_remove = defaultdict(list)
    for shared in shareds:
        complete_names = set([name for name in shared if _is_complete(state_dict[name])])
        if not complete_names:
            if len(shared) == 1:
                # Force contiguous
                name = list(shared)[0]
                state_dict[name] = state_dict[name].clone()
                complete_names = {name}
            else:
                raise RuntimeError(
                    f"Error while trying to find names to remove to save state dict, but found no suitable name to keep for saving amongst: {shared}. None is covering the entire storage.Refusing to save/load the model since you could be storing much more memory than needed. Please refer to https://huggingface.co/docs/safetensors/torch_shared_tensors for more information. Or open an issue."
                )

        keep_name = sorted(list(complete_names))[0]

        # Mecanism to preferentially select keys to keep
        # coming from the on-disk file to allow
        # loading models saved with a different choice
        # of keep_name
        preferred = complete_names.difference(discard_names)
        if preferred:
            keep_name = sorted(list(preferred))[0]

        if preferred_names:
            preferred = preferred_names.intersection(complete_names)
            if preferred:
                keep_name = sorted(list(preferred))[0]
        for name in sorted(shared):
            if name != keep_name:
                to_remove[keep_name].append(name)
    return to_remove

def convert_file(
    extracted_data_path: str,
    sf_filename: str,
    discard_names: List[str],
):
    try:
        state_dict = torch.load(extracted_data_path, map_location="cpu", weights_only=True)
    except Exception as e:
        logging.error(f"Error loading {extracted_data_path}: {e}")
        return

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    to_removes = _remove_duplicate_names(state_dict, discard_names=discard_names)

    metadata = {"format": "pt"}
    for kept_name, to_remove_group in to_removes.items():
        for to_remove in to_remove_group:
            if to_remove not in metadata:
                metadata[to_remove] = kept_name
            del state_dict[to_remove]
    # Force tensors to be contiguous
    state_dict = {k: v.contiguous() for k, v in state_dict.items()}

    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    save_file(state_dict, sf_filename, metadata=metadata)

    print(f"Converted {extracted_data_path} to {sf_filename}")

def convert_from_local(
    model_id: str,
    local_dir: str,
    *,
    revision: Optional[str] = None,
    upload_to_hub: bool = False,
    token: Optional[str] = None,
    max_shard_size: str = "5GB",
):
    api = HfApi(token=token)
    pr_title = "Adding `safetensors` variant of this model"
    logging.info(f"Converting model from {local_dir} to {model_id}")

    extracted_data_dir = os.path.join(local_dir, "extracted_data")
    os.makedirs(extracted_data_dir, exist_ok=True)

    bin_files = [f for f in os.listdir(local_dir) if f.endswith(".bin")]

    operations = []
    for bin_file in tqdm(bin_files, desc="Converting .bin to .safetensors"):
        bin_filepath = os.path.join(local_dir, bin_file)
        extracted_data_path = extract_data_from_zip(bin_filepath, extracted_data_dir)

        if not extracted_data_path:
            continue

        sf_filename = os.path.join(local_dir, bin_file.replace(".bin", ".safetensors"))
        convert_file(
            extracted_data_path,
            sf_filename,
            discard_names=[],
        )

        if upload_to_hub:
            operations.append(
                CommitOperationAdd(
                    path_in_repo=bin_file.replace(".bin", ".safetensors"),
                    path_or_fileobj=sf_filename,
                )
            )
        else:
            logging.info(f"Converted {bin_file} to {sf_filename}. Skipping upload.")

    if upload_to_hub and operations:
        try:
            logging.info(f"Uploading files to {model_id}...")
            api.create_commit(
                repo_id=model_id,
                operations=operations,
                commit_message=pr_title,
                revision=revision,
                create_pr=True,
            )
            logging.info(f"Successfully created PR for {model_id}")
        except Exception as e:
            logging.error(f"Error uploading to Hugging Face Hub: {e}")

    logging.info("Conversion complete!")

def main():
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch .bin model to .safetensors format."
    )
    parser.add_argument(
        "model_id",
        type=str,
        help="The ID of the Hugging Face repository (e.g., 'your_username/your_repo_name').",
        nargs='?',
        default=None,
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        required=True,
        help="The local directory where the .bin files are located.",
    )
    parser.add_argument(
        "--upload_to_hub",
        action="store_true",
        help="Upload the converted files to the Hugging Face Hub. Requires a valid token.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token. Required if uploading to the Hub.",
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="5GB",
        help="The maximum size of the .safetensors shards (e.g., '5GB'). Default is '5GB'.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The branch to create a PR to (e.g. 'main').",
    )

    args = parser.parse_args()

    convert_from_local(
        model_id=args.model_id,
        local_dir=args.local_dir,
        upload_to_hub=args.upload_to_hub,
        token=args.token,
        max_shard_size=args.max_shard_size,
        revision=args.revision,
    )

if __name__ == "__main__":
    main()