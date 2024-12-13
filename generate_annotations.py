import pandas as pd
from pathlib import Path

root = Path.home() / "Downloads/mmlu_output/"

files_list = []
for f in root.glob('*/*/attentions/*_attentions.pt'):
    if f.is_file() and f.stat().st_size > 1e6:  # Exclude the empty files
        files_list.append(f.relative_to(root))

df = pd.DataFrame()
df['filename'] = files_list
df.to_csv(Path.home() / "Downloads/mmlu_attention_files_list.txt", index=False, header=False)