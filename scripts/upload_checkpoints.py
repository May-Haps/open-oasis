#!/venv/open-oasis/bin/python3
"""
Upload checkpoints to HuggingFace Hub.

Usage:
    python upload_checkpoints.py --token hf_xxx --repo your-username/coinrunm
    python upload_checkpoints.py --token hf_xxx --repo your-username/coinrunm --folder baseline
"""

import argparse
import glob
from pathlib import Path
from huggingface_hub import HfApi


def main(args):
    api = HfApi(token=args.token)

    print(f"Creating repo {args.repo} (if not exists) ...")
    api.create_repo(args.repo, repo_type="model", exist_ok=True)

    ckpts = sorted(Path(args.run_dir).glob("*.pt"))
    if not ckpts:
        print(f"No .pt files found in {args.run_dir}")
        return

    print(f"Found {len(ckpts)} checkpoints → uploading to {args.repo}/{args.folder}/\n")
    for path in ckpts:
        dest = f"{args.folder}/{path.name}"
        print(f"  {path.name} ...", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=dest,
            repo_id=args.repo,
            repo_type="model",
        )
        print("done")

    print(f"\nAll uploaded → https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token",   required=True, help="HuggingFace token (hf_...)")
    parser.add_argument("--repo",    required=True, help="HF repo, e.g. your-username/coinrunm")
    parser.add_argument("--folder",  default="baseline", help="Subfolder in repo (default: baseline)")
    parser.add_argument("--run-dir", default="runs/coinrun_small_lin", help="Local dir with .pt files")
    main(parser.parse_args())
