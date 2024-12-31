import argparse
import json
import os
import shutil
import logging
from collections import defaultdict
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Set, Tuple
import zipfile

import torch

from huggingface_hub import CommitInfo, CommitOperationAdd, Discussion, HfApi, hf_hub_download
from huggingface_hub.file_download import repo_folder_name
from safetensors.torch import _find_shared_tensors, _is_complete, load_file, save_file

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

COMMIT_DESCRIPTION = """
This is an automated PR created with https://huggingface.co/spaces/safetensors/convert

This new file is equivalent to `pytorch_model.bin` but safe in the sense that
no arbitrary code can be put into it.

These files also happen to load much faster than their pytorch counterpart:
https://colab.research.google.com/github/huggingface/notebooks/blob/main/safetensors_doc/en/speed.ipynb

The widgets on your model page will run using this model even if this is not merged
making sure the file actually works.

If you find any issues: please report here: https://huggingface.co/spaces/safetensors/convert/discussions

Feel free to ignore this PR.
"""

ConversionResult = Tuple[List["CommitOperationAdd"], List[Tuple[str, "Exception"]]]

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

def get_discard_names(model_id: str, revision: Optional[str], folder: str, token: Optional[str]) -> List[str]:
    try:
        import json

        import transformers

        config_filename = hf_hub_download(
            model_id, revision=revision, filename="config.json", token=token, cache_dir=folder
        )
        with open(config_filename, "r") as f:
            config = json.load(f)
        architecture = config["architectures"][0]

        class_ = getattr(transformers, architecture)

        # Name for this varible depends on transformers version.
        discard_names = getattr(class_, "_tied_weights_keys", [])

    except Exception:
        discard_names = []
    return discard_names

class AlreadyExists(Exception):
    pass

def check_file_size(sf_filename: str, pt_filename: str):
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size

    if (sf_size - pt_size) / pt_size > 0.01:
        raise RuntimeError(
            f"""The file size different is more than 1%:
         - {sf_filename}: {sf_size}
         - {pt_filename}: {pt_size}
         """
        )

def rename(pt_filename: str) -> str:
    filename, ext = os.path.splitext(pt_filename)
    local = f"{filename}.safetensors"
    local = local.replace("pytorch_model", "model")
    return local

def convert_multi(
    folder: str, *, revision: Optional[str], token: Optional[str], discard_names: List[str], model_id: Optional[str]
) -> ConversionResult:
    index_filename = os.path.join(folder, "pytorch_model.bin.index.json")
    with open(index_filename, "r") as f:
        data = json.load(f)

    filenames = set(data["weight_map"].values())
    local_filenames = []
    errors = []

    for filename in filenames:
        pt_filename = os.path.join(folder, filename)

        sf_filename = rename(filename)
        sf_filename = os.path.join(folder, sf_filename)

        try:
            convert_file(pt_filename, sf_filename, discard_names=discard_names)
            local_filenames.append(sf_filename)
        except Exception as e:
            errors.append((pt_filename, e))

    index = os.path.join(folder, "model.safetensors.index.json")
    with open(index, "w") as f:
        newdata = {k: v for k, v in data.items()}
        newmap = {k: rename(v) for k, v in data["weight_map"].items()}
        newdata["weight_map"] = newmap
        json.dump(newdata, f, indent=4)
    local_filenames.append(index)

    operations = [
        CommitOperationAdd(path_in_repo=local.split("/")[-1], path_or_fileobj=local) for local in local_filenames
    ]

    return operations, errors

def convert_single(
    folder: str, *, revision: Optional[str], token: Optional[str], discard_names: List[str], model_id: Optional[str]
) -> ConversionResult:
    pt_filename = os.path.join(folder, "pytorch_model.bin")

    sf_name = "model.safetensors"
    sf_filename = os.path.join(folder, sf_name)

    try:
        convert_file(pt_filename, sf_filename, discard_names)
        operations = [CommitOperationAdd(path_in_repo=sf_name, path_or_fileobj=sf_filename)]
        errors: List[Tuple[str, "Exception"]] = []
    except Exception as e:
        operations = []
        errors = [(pt_filename, e)]

    return operations, errors

def convert_file(
    pt_filename: str,
    sf_filename: str,
    discard_names: List[str],
):
    extracted_data_dir = os.path.join(os.path.dirname(sf_filename), "extracted_data")
    os.makedirs(extracted_data_dir, exist_ok=True)

    extracted_data_path = extract_data_from_zip(pt_filename, extracted_data_dir)
    if not extracted_data_path:
        raise Exception(f"Could not extract data.pkl from {pt_filename}")

    state_dict = torch.load(extracted_data_path, map_location="cpu", weights_only=True)

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
    check_file_size(sf_filename, pt_filename)
    reloaded = load_file(sf_filename)
    for k in state_dict:
        pt_tensor = state_dict[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")

def create_diff(pt_infos: Dict[str, List[str]], sf_infos: Dict[str, List[str]]) -> str:
    errors = []
    for key in ["missing_keys", "mismatched_keys", "unexpected_keys"]:
        pt_set = set(pt_infos[key])
        sf_set = set(sf_infos[key])

        pt_only = pt_set - sf_set
        sf_only = sf_set - pt_set

        if pt_only:
            errors.append(f"{key} : PT warnings contain {pt_only} which are not present in SF warnings")
        if sf_only:
            errors.append(f"{key} : SF warnings contain {sf_only} which are not present in PT warnings")
    return "\n".join(errors)

def previous_pr(api: "HfApi", model_id: str, pr_title: str, revision: Optional[str]) -> Optional["Discussion"]:
    try:
        revision_commit = api.model_info(model_id, revision=revision).sha
        discussions = api.get_repo_discussions(repo_id=model_id)
    except Exception:
        return None
    for discussion in discussions:
        if discussion.status in {"open", "closed"} and discussion.is_pull_request and discussion.title == pr_title:
            commits = api.list_repo_commits(model_id, revision=discussion.git_reference)

            if revision_commit == commits[1].commit_id:
                return discussion
    return None

def convert_from_local(
    model_id: str,
    local_dir: str,
    *,
    revision: Optional[str] = None,
    force: bool = False,
    token: Optional[str] = None,
) -> Tuple[Optional["CommitInfo"], List[Tuple[str, "Exception"]]]:
    pr_title = "Adding `safetensors` variant of this model"
    api = HfApi(token=token)
    try:
        operations = None
        pr = previous_pr(api, model_id, pr_title, revision=revision)

        filenames = os.listdir(local_dir)
        discard_names = get_discard_names(model_id, revision=revision, folder=local_dir, token=token)
        if "pytorch_model.bin" in filenames:
            operations, errors = convert_single(
                local_dir, revision=revision, token=token, discard_names=discard_names, model_id=model_id
            )
        elif "pytorch_model.bin.index.json" in filenames:
            operations, errors = convert_multi(
                local_dir, revision=revision, token=token, discard_names=discard_names, model_id=model_id
            )
        else:
            raise RuntimeError(f"No valid pytorch model found in {local_dir}. Cannot convert")

        new_pr = None
        if operations:
            new_pr = api.create_commit(
                repo_id=model_id,
                revision=revision,
                operations=operations,
                commit_message=pr_title,
                commit_description=COMMIT_DESCRIPTION,
                create_pr=True,
            )
            print(f"PR created at {new_pr.pr_url}")
        else:
            print("No files to convert")
        return new_pr, errors

    except Exception as e:
        logging.error(f"Error during conversion: {e}")
        return None, [(local_dir, e)]

def convert(
    api: "HfApi", model_id: str, revision: Optional[str] = None, force: bool = False
) -> Tuple["CommitInfo", List[Tuple[str, "Exception"]]]:
    pr_title = "Adding `safetensors` variant of this model"
    info = api.model_info(model_id, revision=revision)
    filenames = set(s.rfilename for s in info.siblings)

    with TemporaryDirectory() as d:
        folder = os.path.join(d, repo_folder_name(repo_id=model_id, repo_type="models"))
        os.makedirs(folder)
        new_pr = None
        try:
            operations = None
            pr = previous_pr(api, model_id, pr_title, revision=revision)

            library_name = getattr(info, "library_name", None)
            if any(filename.endswith(".safetensors") for filename in filenames) and not force:
                raise AlreadyExists(f"Model {model_id} is already converted, skipping..")
            elif pr is not None and not force:
                url = f"https://huggingface.co/{model_id}/discussions/{pr.num}"
                new_pr = pr
                raise AlreadyExists(f"Model {model_id} already has an open PR check out {url}")
            elif library_name == "transformers":

                discard_names = get_discard_names(model_id, revision=revision, folder=folder, token=api.token)
                if "pytorch_model.bin" in filenames:
                    operations, errors = convert_single(
                        model_id, revision=revision, folder=folder, token=api.token, discard_names=discard_names
                    )
                elif "pytorch_model.bin.index.json" in filenames:
                    operations, errors = convert_multi(
                        model_id, revision=revision, folder=folder, token=api.token, discard_names=discard_names
                    )
                else:
                    raise RuntimeError(f"Model {model_id} doesn't seem to be a valid pytorch model. Cannot convert")
            else:
                operations, errors = convert_generic(
                    model_id, revision=revision, folder=folder, filenames=filenames, token=api.token
                )

            if operations:
                new_pr = api.create_commit(
                    repo_id=model_id,
                    revision=revision,
                    operations=operations,
                    commit_message=pr_title,
                    commit_description=COMMIT_DESCRIPTION,
                    create_pr=True,
                )
                print(f"Pr created at {new_pr.pr_url}")
            else:
                print("No files to convert")
        finally:
            shutil.rmtree(folder)
        return new_pr, errors

if __name__ == "__main__":
    DESCRIPTION = """
    Simple utility tool to convert automatically some weights on the hub to `safetensors` format.
    It is PyTorch exclusive for now.
    It works by downloading the weights (PT), converting them locally, and uploading them back
    as a PR on the hub.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        "model_id",
        type=str,
        help="The name of the model on the hub to convert. E.g. `gpt2` or `facebook/wav2vec2-base-960h`. Alternatively, the local directory if combined with --local_dir",
        nargs='?',
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        help="The revision to convert",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Create the PR even if it already exists of if the model was already converted.",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        help="Path to local directory containing PyTorch .bin files.",
    )
    parser.add_argument(
        "--max_shard_size",
        type=str,
        default="5GB",
        help="Maximum shard size for the converted .safetensors files.",
    )
    parser.add_argument(
        "--no_upload",
        action="store_false",
        dest="upload_to_hub",
        help="Skip uploading the converted files to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete the original .bin files after conversion.",
    )
    parser.add_argument(
        "-y",
        action="store_true",
        help="Skip the safety prompt",
    )
    args = parser.parse_args()

    if args.local_dir:
        if args.model_id is None:
            raise ValueError("When using --local_dir, you must specify a model_id for the target Hugging Face repository.")
        commit_info, errors = convert_from_local(
            model_id=args.model_id,
            local_dir=args.local_dir,
            revision=args.revision,
            force=args.force,
        )
    else:
        api = HfApi()
        if args.y:
            txt = "y"
        else:
            txt = input(
                "This conversion script will unpickle a pickled file, which is inherently unsafe. If you do not trust this file, we invite you to use"
                " https://huggingface.co/spaces/safetensors/convert or google colab or other hosted solution to avoid potential issues with this file."
                " Continue [Y/n] ?"
            )
        if txt.lower() in {"", "y"}:
            commit_info, errors = convert(api, args.model_id, revision=args.revision, force=args.force)
        else:
            print(f"Answer was `{txt}` aborting.")
            exit()

    if commit_info:
        string = f"""
    ### Success 🔥
    Yay! This model was successfully converted and a PR was open using your token, here:
    [{commit_info.pr_url}]({commit_info.pr_url})
            """
        if errors:
            string += "\nErrors during conversion:\n"
            string += "\n".join(
                f"Error while converting {filename}: {e}, skipped conversion" for filename, e in errors
            )
        print(string)
    else:
        print("Conversion failed or was skipped.")
        if errors:
            print("\nErrors during conversion:")
            for filename, e in errors:
                print(f"Error while converting {filename}: {e}")