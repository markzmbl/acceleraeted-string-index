import pandas as pd




with open("output.txt", 'r') as fin:
    lines = fin.readlines()

df = pd.DataFrame()

i = -1
for line in lines:
    if "[GROUP]" in line:
        i = int(line[line.find('\t') : line.find('\n')])
        df.append(dict(), ignore_index=True)
    elif line[0] == '\t':
        line = line[1 : -1]
        (col, val) = line.split('\t')
        col = col[ : -1]
        if i == 0:
            df[col] = 0
        if col in ("avg", "min", "max"):
            df.at[i, col] = float(val)
        else:
            df.at[i, col] = int(val)

df.to_csv("output.csv")