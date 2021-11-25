import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np




def read_comments(filename):
    # comments are in format
    # #key: value
    comments = dict()
    with open(filename, "r") as fin:
        for line in fin:
            if line[0] == "#":
                comment = line[1:-1].split(":")
                comments[comment[0]] = float(comment[1])
            else:
                break
    return comments

def main(file, start, end):
    csv_columns = {"mins", "group_n", "root_n", "time", "avgsize"}
    columns = {"avg", "size", "mins", "group_n", "root_n", "time", "avgsize"}
    path0 = "./csv/query/"
    path1 = "./csv/grouping/"
    _, _, filenames = next(os.walk(path1))
    datasets = []
    for filename in filenames:
        if file in filename:
            datasets.append(filename)

    comments = dict()
    for dataset in datasets:
        key = dataset.split("_")[2]
        if (start.split("e")[1] == key.split("e")[1] == end.split("e")[1] and
                int(start.split("e")[0]) <= int(key.split("e")[0]) <= int(end.split("e")[0])):
            comment = read_comments(path1+dataset)
            #comment.update(read_comments(path0+dataset))
            # average group size 
            df = pd.read_csv(path1+dataset, comment="#")
            df = df[df["type"] == "group"]
            comment["avgsize"] = df["m"].mean()

            if csv_columns <= set(comment.keys()):
                comments[float(key)] = comment

    
    df = pd.DataFrame(
        index=comments.keys(),
        columns=columns
    )

    for key, val in comments.items():
        df.loc[key] = val
    
    path2 = "./csv/benchmark/"
    df.to_csv(path2+file+".csv")

    path3 = "./plots/"+file+"/"
    Path(path3).mkdir(parents=True, exist_ok=True)

    df = df.sort_index()
    x = df.index.values.tolist() 

    for col in columns:
        y = df[col].values.tolist()

        plt.plot(x, y)
        plt.savefig(path3+col+".png")
        plt.clf()

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file')
    parser.add_argument('-s', '--start')
    parser.add_argument('-e', '--end')

    args = parser.parse_args()


    main(args.file, args.start, args.end)