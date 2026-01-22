import pandas as pd
import os
from datetime import datetime
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
STATS_DIR = os.path.join(ROOT_DIR, "outputs", "statistics")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATS_DIR, exist_ok=True)

TASK_PATHS = {
    "task_a": "task_a/task_a_training_set_1.parquet",
    "task_b": "task_b/task_b_training_set.parquet"
}

# Config
MAX_CODE_LENGTH = 10000  # Cap outliers (CodeBERT uses 512 tokens anyway)
TEST_SIZE = 0.2  # 80/20 train/val split


def get_device():
    """Get best available device (MPS for Apple Silicon, CUDA, or CPU)."""
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_stats_to_file(stats_list):
    """Save all task statistics to a markdown file."""
    stats_path = os.path.join(STATS_DIR, "dataset_splits.md")
    
    with open(stats_path, 'w') as f:
        f.write(f"# Dataset Split Statistics\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"---\n\n")
        
        for stats in stats_list:
            f.write(f"## {stats['task_name']}\n\n")
            f.write(f"- **Train samples**: {stats['train_size']}\n")
            f.write(f"- **Val samples**: {stats['val_size']}\n")
            f.write(f"- **Code capped**: {stats['capped_count']} samples (>{MAX_CODE_LENGTH} chars)\n")
            f.write(f"- **Stratification**: {stats['stratification']}\n\n")
            
            f.write(f"### Label Distribution\n\n")
            f.write(f"| Split | Distribution |\n")
            f.write(f"|-------|-------------|\n")
            f.write(f"| Train | {stats['train_labels']} |\n")
            f.write(f"| Val   | {stats['val_labels']} |\n\n")
            
            f.write(f"### Language Distribution\n\n")
            f.write(f"| Split | Distribution |\n")
            f.write(f"|-------|-------------|\n")
            f.write(f"| Train | {stats['train_langs']} |\n")
            f.write(f"| Val   | {stats['val_langs']} |\n\n")
            
            f.write(f"---\n\n")
    
    print(f"\n✓ Stats saved to {stats_path}")
    return stats_path


def create_full_metadata_subset(task_name, subset_size=20000): 
    print(f"\n{'='*50}")
    print(f"Fetching {task_name} with metadata...")
    
    cols = ["code", "generator", "label", "language"]
    url = f"hf://datasets/DaniilOr/SemEval-2026-Task13/{TASK_PATHS[task_name]}"
    
    df = pd.read_parquet(url, columns=cols)
    df['generator'] = df['generator'].str.capitalize()
    
    # Cap code length to handle outliers
    original_lengths = df['code'].str.len()
    df['code'] = df['code'].str[:MAX_CODE_LENGTH]
    capped_count = int((original_lengths > MAX_CODE_LENGTH).sum())
    print(f"Capped {capped_count} samples exceeding {MAX_CODE_LENGTH} chars")

    # Stratify by BOTH label AND language (preferred for task_a)
    # For task_b with many sparse label+language combos, fall back to label-only
    df['stratify_col'] = df['label'].astype(str) + "_" + df['language']
    
    # Check if any stratify group has < 2 samples (will cause error)
    group_counts = df['stratify_col'].value_counts()
    sparse_groups = group_counts[group_counts < 2]
    
    if len(sparse_groups) > 0:
        print(f"⚠️  Found {len(sparse_groups)} sparse label+language groups, falling back to label-only stratification")
        stratify_col = df['label']
        stratify_col_subset = 'label'
        stratification_method = "label-only (fallback)"
    else:
        stratify_col = df['stratify_col']
        stratify_col_subset = 'stratify_col'
        stratification_method = "label + language"
    
    # First: sample subset from full dataset
    _, subset_df = train_test_split(
        df,
        test_size=subset_size,
        stratify=stratify_col, 
        random_state=42
    )
    
    # Second: split subset into train/val
    train_df, val_df = train_test_split(
        subset_df,
        test_size=TEST_SIZE,
        stratify=subset_df[stratify_col_subset],
        random_state=42
    )
    
    # Drop the temporary stratify column
    train_df = train_df.drop(columns=['stratify_col'])
    val_df = val_df.drop(columns=['stratify_col'])
    
    # Save train and val separately
    train_path = os.path.join(DATA_DIR, f"{task_name}_train.parquet")
    val_path = os.path.join(DATA_DIR, f"{task_name}_val.parquet")
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    
    # Also save combined subset for backward compatibility
    subset_path = os.path.join(DATA_DIR, f"{task_name}_subset.parquet")
    combined = pd.concat([train_df, val_df], ignore_index=True)
    combined.to_parquet(subset_path, index=False)
    
    # Collect stats
    train_labels = train_df['label'].value_counts().to_dict()
    val_labels = val_df['label'].value_counts().to_dict()
    train_langs = train_df['language'].value_counts(normalize=True).round(3).to_dict()
    val_langs = val_df['language'].value_counts(normalize=True).round(3).to_dict()
    
    # Print verification info
    print(f"\n✓ Saved {len(train_df)} train samples to {train_path}")
    print(f"✓ Saved {len(val_df)} val samples to {val_path}")
    
    print(f"\n--- Label Distribution ---")
    print(f"Train: {train_labels}")
    print(f"Val:   {val_labels}")
    
    print(f"\n--- Language Distribution ---")
    print(f"Train: {train_langs}")
    print(f"Val:   {val_langs}")
    
    stats = {
        'task_name': task_name.upper(),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'capped_count': capped_count,
        'stratification': stratification_method,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'train_langs': train_langs,
        'val_langs': val_langs,
    }
    
    return train_df, val_df, stats


if __name__ == "__main__":
    print(f"Device available: {get_device()}")
    
    all_stats = []
    
    train_a, val_a, stats_a = create_full_metadata_subset("task_a")
    all_stats.append(stats_a)
    
    train_b, val_b, stats_b = create_full_metadata_subset("task_b")
    all_stats.append(stats_b)
    
    save_stats_to_file(all_stats)