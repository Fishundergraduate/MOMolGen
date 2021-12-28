import pandas as pd

bindingBD = pd.read_csv('BindingDB_All.tsv', sep = '\t', error_bad_lines=False)

for row in bindingBD:
    print(row.size)