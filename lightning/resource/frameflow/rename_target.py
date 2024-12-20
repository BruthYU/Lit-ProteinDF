import pandas as pd
task_df = pd.read_csv("benchmark.csv")
task_df['target'] = task_df['target'].map(lambda x : x.lower())
task_df.reset_index(drop=True)
task_df.to_csv("new_benchmark.csv", index=False)

pass