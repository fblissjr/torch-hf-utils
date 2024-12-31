import os
import zipfile
import pickle
import json
from collections import defaultdict
from tqdm import tqdm

def inspect_bin_file(filepath):
    """
    Inspects a .bin (zip) file, extracts the data.pkl if present,
    and analyzes its header.

    Args:
        filepath: Path to the .bin file.

    Returns:
        A dictionary containing:
        - filename: Name of the .bin file.
        - is_zip: True if the file is a zip archive, False otherwise.
        - zip_files: List of files inside the zip archive (if applicable).
        - data_pkl_header_size: Size of the header of data.pkl (if applicable).
        - data_pkl_header: First 1024 bytes of the data.pkl header (if applicable).
        - error: Any error encountered during inspection.
        - metadata: Decoded metadata from the header, if applicable and possible.
    """
    result = {
        "filename": os.path.basename(filepath),
        "is_zip": False,
        "zip_files": [],
        "data_pkl_header_size": None,
        "data_pkl_header": None,
        "error": None,
        "metadata": None
    }

    try:
        # Check if it's a zip file
        if zipfile.is_zipfile(filepath):
            result["is_zip"] = True
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                result["zip_files"] = zip_ref.namelist()

                # Find data.pkl
                for member in result["zip_files"]:
                    if member.endswith('data.pkl'):
                        # Extract data.pkl
                        with zip_ref.open(member) as data_pkl:
                            header = data_pkl.read(1024)
                            result["data_pkl_header_size"] = len(header) # This is not precise (doesn't account for full header size), but gives an idea
                            result["data_pkl_header"] = header

                            # Attempt to decode metadata
                            try:
                                pickle_bytes = result["data_pkl_header"].split(b"\x80", 1)[-1] #grab everything before \x80 in case there is junk before it.
                                if pickle_bytes:
                                    metadata = pickle.loads(b"\x80" + pickle_bytes)
                                    if isinstance(metadata, dict):
                                        result["metadata"] = metadata
                                    else:
                                        result["error"] = "Metadata did not decode to a dictionary."
                                else:
                                    result["error"] = "Could not parse metadata"
                            except Exception as e:
                                result["error"] = f"Metadata decoding error: {e}"
                        break
        else:
            result["error"] = "Not a zip file"

    except Exception as e:
        result["error"] = str(e)

    return result

def generate_summary_report(directory):
    """
    Generates a summary report of .bin files in a directory.

    Args:
        directory: Path to the directory containing .bin files.

    Returns:
        A list of dictionaries, where each dictionary is the result of
        inspect_bin_file() for a .bin file.
    """
    bin_files = [f for f in os.listdir(directory) if f.endswith(".bin")]
    report = []

    for filename in tqdm(bin_files, desc="Inspecting .bin files"):
        filepath = os.path.join(directory, filename)
        report.append(inspect_bin_file(filepath))

    return report

def save_report_to_json(report, output_filename="inspection_report.json"):
    """Saves the inspection report to a JSON file."""
    with open(output_filename, "w") as f:
        json.dump(report, f, indent=4)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect .bin files and generate a summary report.")
    parser.add_argument(
        "directory",
        type=str,
        help="The directory containing the .bin files."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inspection_report.json",
        help="Output JSON filename for the report. Defaults to 'inspection_report.json'."
    )

    args = parser.parse_args()
    report = generate_summary_report(args.directory)
    save_report_to_json(report, args.output)

    print(f"Inspection report saved to {args.output}")