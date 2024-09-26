import pandas as pd


pdb_csv = pd.read_csv('./pkl/mmcif/metadata.csv')
print(pdb_csv.columns.__contains__("oligomeric_detail"))
pass