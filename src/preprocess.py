"""
Preprocessing module for Machine-Generated Code Detection.

This module applies preprocessing to the already-downloaded parquet files:
- Removes comments (single-line only for safety)
- Normalizes whitespace

Usage:
    python src/preprocess.py
"""

import pandas as pd
import os
import re

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")


def remove_comments(code: str, language: str) -> str:
    """
    Remove single-line comments based on language.
    
    NOTE: Only removes unambiguous single-line comments to avoid
    incorrectly removing triple-quoted strings in Python.
    
    Args:
        code: Raw code string
        language: Programming language (Python, Java, C++, etc.)
        
    Returns:
        Code with single-line comments removed
    """
    if not isinstance(code, str):
        return code
    
    lines = code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Python: # comments (but not inside strings)
        if language.lower() == 'python':
            # Simple approach: remove # and everything after if not in a string
            # Find # that's not inside quotes
            in_string = False
            quote_char = None
            result = []
            i = 0
            while i < len(line):
                char = line[i]
                
                # Handle string detection
                if char in ('"', "'") and (i == 0 or line[i-1] != '\\'):
                    if not in_string:
                        in_string = True
                        quote_char = char
                    elif char == quote_char:
                        in_string = False
                        quote_char = None
                
                # Handle comment
                if char == '#' and not in_string:
                    break  # Rest of line is comment
                
                result.append(char)
                i += 1
            
            cleaned_lines.append(''.join(result))
        
        # Java, C++, C#, JavaScript, Go, PHP, C: // comments
        elif language.lower() in ('java', 'c++', 'c#', 'javascript', 'go', 'php', 'c'):
            # Remove // comments (simple approach)
            comment_idx = line.find('//')
            if comment_idx != -1:
                # Check it's not inside a string (rough check)
                before = line[:comment_idx]
                if before.count('"') % 2 == 0 and before.count("'") % 2 == 0:
                    line = line[:comment_idx]
            cleaned_lines.append(line)
        
        else:
            # Unknown language, keep as-is
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def normalize_whitespace(code: str) -> str:
    """
    Normalize whitespace in code.
    
    - Replace tabs with 4 spaces
    - Remove trailing whitespace from each line
    - Collapse multiple blank lines into one
    - Strip leading/trailing whitespace from entire code
    
    Args:
        code: Code string
        
    Returns:
        Code with normalized whitespace
    """
    if not isinstance(code, str):
        return code
    
    # Replace tabs with 4 spaces
    code = code.replace('\t', '    ')
    
    # Remove trailing whitespace from each line
    lines = [line.rstrip() for line in code.split('\n')]
    
    # Collapse multiple blank lines into one
    result_lines = []
    prev_blank = False
    for line in lines:
        is_blank = len(line.strip()) == 0
        if is_blank and prev_blank:
            continue  # Skip consecutive blank lines
        result_lines.append(line)
        prev_blank = is_blank
    
    # Join and strip
    return '\n'.join(result_lines).strip()


def preprocess_code(code: str, language: str = 'python') -> str:
    """
    Full preprocessing pipeline: remove comments + normalize whitespace.
    
    Args:
        code: Raw code string
        language: Programming language
        
    Returns:
        Preprocessed code
    """
    code = remove_comments(code, language)
    code = normalize_whitespace(code)
    return code


def add_preprocessed_column(input_path: str, output_path: str = None) -> dict:
    """
    Load a parquet file, add 'code_preprocessed' column, save back.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to save (defaults to overwriting input)
        
    Returns:
        Stats dict with preprocessing metrics
    """
    if output_path is None:
        output_path = input_path
    
    print(f"Loading {input_path}...")
    df = pd.read_parquet(input_path)
    
    print(f"Preprocessing {len(df)} samples...")
    df['code_preprocessed'] = df.apply(
        lambda row: preprocess_code(row['code'], row['language']),
        axis=1
    )
    
    # Stats
    original_lens = df['code'].str.len()
    preprocessed_lens = df['code_preprocessed'].str.len()
    avg_reduction = (1 - preprocessed_lens.mean() / original_lens.mean()) * 100
    
    stats = {
        'file': os.path.basename(input_path),
        'samples': len(df),
        'original_avg': int(original_lens.mean()),
        'preprocessed_avg': int(preprocessed_lens.mean()),
        'reduction_pct': round(avg_reduction, 1),
    }
    
    print(f"  Avg length reduction: {avg_reduction:.1f}%")
    print(f"  Original avg: {original_lens.mean():.0f} chars")
    print(f"  Preprocessed avg: {preprocessed_lens.mean():.0f} chars")
    
    df.to_parquet(output_path, index=False)
    print(f"✓ Saved to {output_path}")
    
    return stats


def save_preprocessing_stats(all_stats: list):
    """Save preprocessing stats to outputs/statistics/preprocessing_stats.md"""
    from datetime import datetime
    
    stats_dir = os.path.join(ROOT_DIR, "outputs", "statistics")
    os.makedirs(stats_dir, exist_ok=True)
    stats_path = os.path.join(stats_dir, "preprocessing_stats.md")
    
    with open(stats_path, 'w') as f:
        f.write("# Preprocessing Statistics\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## Summary\n\n")
        f.write("| File | Samples | Original Avg | Preprocessed Avg | Reduction |\n")
        f.write("|------|---------|--------------|------------------|----------|\n")
        
        for s in all_stats:
            f.write(f"| {s['file']} | {s['samples']} | {s['original_avg']} chars | {s['preprocessed_avg']} chars | {s['reduction_pct']}% |\n")
        
        f.write("\n---\n\n")
        f.write("## Methodology\n\n")
        f.write("- **Comment removal**: Single-line only (`#` for Python, `//` for C-family)\n")
        f.write("- **Whitespace normalization**: Tabs → 4 spaces, trailing whitespace removed\n")
        f.write("- **Conservative approach**: No multiline/docstring removal to avoid data corruption\n")
        f.write("\nSee [docs/preprocess_docs.md](../docs/preprocess_docs.md) for detailed rationale.\n")
    
    print(f"\n✓ Stats saved to {stats_path}")
    return stats_path


def test_preprocessing(n_samples: int = 100):
    """
    Test preprocessing on a sample of data to verify no crashes.
    """
    print(f"\n{'='*50}")
    print(f"Testing preprocessing on {n_samples} samples...")
    
    train_path = os.path.join(DATA_DIR, "task_a_train.parquet")
    df = pd.read_parquet(train_path).head(n_samples)
    
    errors = []
    for idx, row in df.iterrows():
        try:
            result = preprocess_code(row['code'], row['language'])
            assert isinstance(result, str), f"Expected string, got {type(result)}"
        except Exception as e:
            errors.append((idx, str(e)))
    
    if errors:
        print(f"❌ {len(errors)} errors encountered:")
        for idx, err in errors[:5]:
            print(f"  Sample {idx}: {err}")
    else:
        print(f"✓ All {n_samples} samples processed without errors")
    
    # Show example
    sample = df.iloc[0]
    original = sample['code'][:200]
    processed = preprocess_code(sample['code'], sample['language'])[:200]
    
    print(f"\n--- Example (first 200 chars) ---")
    print(f"Language: {sample['language']}")
    print(f"Original:\n{original}")
    print(f"\nPreprocessed:\n{processed}")
    
    return len(errors) == 0


if __name__ == "__main__":
    # First, test preprocessing
    if test_preprocessing(100):
        print("\n" + "="*50)
        print("Preprocessing all data files...")
        
        all_stats = []
        
        # Process all task_a files
        for split in ['train', 'val']:
            path = os.path.join(DATA_DIR, f"task_a_{split}.parquet")
            if os.path.exists(path):
                stats = add_preprocessed_column(path)
                all_stats.append(stats)
        
        # Also process the combined subset
        subset_path = os.path.join(DATA_DIR, "task_a_subset.parquet")
        if os.path.exists(subset_path):
            stats = add_preprocessed_column(subset_path)
            all_stats.append(stats)
        
        # Save stats to outputs folder
        save_preprocessing_stats(all_stats)
        
        print("\n✓ All preprocessing complete!")
    else:
        print("\n❌ Fix preprocessing errors before continuing")

