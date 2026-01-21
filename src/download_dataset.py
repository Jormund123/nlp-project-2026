import pandas as pd
import os
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

TASK_PATHS = {
    "task_a": "task_a/task_a_training_set_1.parquet",
    "task_b": "task_b/task_b_training_set.parquet"
}

def create_full_metadata_subset(task_name, subset_size=20000): 
    print(f"Fetching {task_name} with metadata...")
    
    cols = ["code", "generator", "label", "language"]
    url = f"hf://datasets/DaniilOr/SemEval-2026-Task13/{TASK_PATHS[task_name]}"
    
    df = pd.read_parquet(url, columns=cols)
    df['generator'] = df['generator'].str.capitalize()

    _, subset_df = train_test_split(
        df,
        test_size=subset_size,
        stratify=df['label'],
        random_state=42
    )
    
    output_path = os.path.join(DATA_DIR, f"{task_name}_subset.parquet")
    subset_df.to_parquet(output_path, index=False)
    
    print(f"Success! Saved {subset_size} samples to {output_path}")
    print(f"Label distribution:\n{subset_df['label'].value_counts()}")
    return subset_df

if __name__ == "__main__":
    df_a = create_full_metadata_subset("task_a")
    df_b = create_full_metadata_subset("task_b")