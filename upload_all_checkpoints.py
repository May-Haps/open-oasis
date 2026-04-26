#!/venv/open-oasis/bin/python3
"""
Upload all model checkpoints to JoshuaYang/coinrun on HuggingFace.
Renames baseline/ → 57M/, uploads 17M/ and 31M/.

Usage:
    python upload_all_checkpoints.py --token hf_xxx
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi


def main(args):
    api = HfApi(token=args.token)
    repo = "JoshuaYang/coinrun"

    api.create_repo(repo, repo_type="model", exist_ok=True)

    # 1. Move baseline/ → 57M/ (HF has no rename, so download+reupload+delete)
    print("Moving baseline/ → 57M/ ...")
    try:
        files = list(api.list_repo_files(repo, repo_type="model"))
        baseline_files = [f for f in files if f.startswith("baseline/")]
        if baseline_files:
            for f in baseline_files:
                new_name = "57M/" + f[len("baseline/"):]
                print(f"  {f} → {new_name} ...", end=" ", flush=True)
                local = api.hf_hub_download(repo_id=repo, filename=f, repo_type="model")
                api.upload_file(
                    path_or_fileobj=local,
                    path_in_repo=new_name,
                    repo_id=repo,
                    repo_type="model",
                )
                api.delete_file(path_in_repo=f, repo_id=repo, repo_type="model")
                print("done")
        else:
            print("  No baseline/ files found (already renamed or not yet uploaded)")
    except Exception as e:
        print(f"  Warning: {e}")

    # 2. Upload new 57M checkpoints from local run (ckpt_step_*.pt)
    ckpts_57M = sorted(Path("runs/coinrun_v1").glob("ckpt_step_*.pt"))
    print(f"\nUploading {len(ckpts_57M)} new 57M checkpoints ...")
    for p in ckpts_57M:
        dest = f"57M/{p.name}"
        print(f"  {p.name} ...", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=dest,
            repo_id=repo,
            repo_type="model",
        )
        print("done")

    # 3. Upload 17M checkpoints (ckpt_step_*.pt only)
    ckpts_17M = sorted(Path("runs/coinrun_17M").glob("ckpt_step_*.pt"))
    print(f"\nUploading {len(ckpts_17M)} 17M checkpoints ...")
    for p in ckpts_17M:
        dest = f"17M/{p.name}"
        print(f"  {p.name} ...", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=dest,
            repo_id=repo,
            repo_type="model",
        )
        print("done")

    # 4. Upload 31M checkpoints (ckpt_step_*.pt only — skip synthetic/epoch files)
    ckpts_31M = sorted(Path("runs/coinrun_31M").glob("ckpt_step_*.pt"))
    print(f"\nUploading {len(ckpts_31M)} 31M checkpoints ...")
    for p in ckpts_31M:
        dest = f"31M/{p.name}"
        print(f"  {p.name} ...", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=dest,
            repo_id=repo,
            repo_type="model",
        )
        print("done")

    print(f"\nDone → https://huggingface.co/{repo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="HuggingFace write token (hf_...)")
    main(parser.parse_args())
