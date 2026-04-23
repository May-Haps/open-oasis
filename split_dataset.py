import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

def split_data(source_dir, train_dir, val_dir, val_ratio=0.1, seed=42):
    random.seed(seed)
    source_path = Path(source_dir)
    train_path = Path(train_dir)
    val_path = Path(val_dir)

    # 1. Create directories
    for p in [train_path, val_path]:
        if p.exists():
            print(f"Warning: {p} already exists. Deleting and recreating...")
            shutil.rmtree(p)
        p.mkdir(parents=True)

    # 2. Group frames by Episode ID
    # Assuming your filenames look like 'ep12_s123_f456.png' 
    # or are stored in episode folders. 
    # We will find all PNGs and group them by their parent directory or prefix.
    episodes = defaultdict(list)
    
    print("Scanning frames and grouping by episode...")
    all_pngs = list(source_path.glob("**/*.png"))
    
    for png in all_pngs:
        # Strategy: Use the parent folder name as the episode ID
        # If your data is flat, change this to png.name.split('_')[0]
        ep_id = png.parent.name 
        episodes[ep_id].append(png)

    ep_ids = list(episodes.keys())
    random.shuffle(ep_ids)

    # 3. Calculate split
    n_val = max(1, int(len(ep_ids) * val_ratio))
    val_episodes = set(ep_ids[:n_val])
    train_episodes = set(ep_ids[n_val:])

    print(f"Total Episodes: {len(ep_ids)}")
    print(f"Training Episodes: {len(train_episodes)}")
    print(f"Validation Episodes: {len(val_episodes)}")

    # 4. Create Symlinks
    def create_links(ep_list, target_base_path):
        count = 0
        for ep in ep_list:
            target_ep_dir = target_base_path / ep
            target_ep_dir.mkdir(exist_ok=True)
            
            for src_png in episodes[ep]:
                target_file = target_ep_dir / src_png.name
                
                if not target_file.exists():
                    try:
                        # Try Symlink first
                        os.symlink(src_png.absolute(), target_file)
                    except OSError:
                        try:
                            # Fallback to Hard Link (Works without Admin on Windows)
                            os.link(src_png.absolute(), target_file)
                        except OSError:
                            # Final Fallback: Copy (Uses more disk space)
                            shutil.copy2(src_png.absolute(), target_file)
                count += 1
        return count

    print("Linking training frames...")
    t_count = create_links(train_episodes, train_path)
    print(f"Linked {t_count} training frames.")

    print("Linking validation frames...")
    v_count = create_links(val_episodes, val_path)
    print(f"Linked {v_count} validation frames.")

if __name__ == "__main__":
    # Update these paths to your actual data locations
    split_data(
        source_dir="../data/raw_smb_subset", 
        train_dir="../data/vae_dataset/train", 
        val_dir="../data/vae_dataset/val",
        val_ratio=0.1 # 10% of episodes for validation
    )