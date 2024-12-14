import json
import pandas as pd
from pathlib import Path

root = Path.home() / "Downloads/mmlu_output/"

choices = ('A', 'B', 'C', 'D')
files_list = []
predictions_list = []
correct_list = []
for f in root.glob('*/*/attentions/*_attentions.pt'):
    if f.is_file() and f.stat().st_size > 2e6:  # Exclude the empty files
        files_list.append(f.relative_to(root))
        q_id = str(f.stem).split('_')[0]
        ans_file = next(iter(f.parent.parent.glob("*_answers.json")))
        with open(ans_file, "r") as infile:
            ans_json = json.load(infile)
            full_ans = ans_json[q_id]["Full Answer"]
            pred = ans_json[q_id]["Predicted Answer"]
            correct = ans_json[q_id]["Correct Answer"]
            if pred not in choices and full_ans[0] in choices:
                pred = full_ans[0]
        predictions_list.append(pred)
        correct_list.append(correct)

df = pd.DataFrame()
df['filename'] = files_list
df['prediction'] = predictions_list
df['correct'] = correct_list
accuracy = (df['prediction'] == df['correct']).sum() / len(df)
print(len(df))
print(accuracy)

df.to_csv(Path.home() / "Downloads/mmlu_attention_files_list.txt", index=False)