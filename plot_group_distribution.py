#/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import math
import multiprocessing as mp
import argparse
from bisect import bisect_left


#https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value/12141511#12141511
def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before

def gene_to_num(l):
    l = l[ : -1]
    num = 0
    for i, c in enumerate(reversed(l)):
        num += (ord(c) - ord("A")) * 4 ** i
    return num

def protein_to_num(l):
    l = l[ : -1]
    num = 0
    for i, c in enumerate(reversed(l)):
        num += (ord(c) - ord("A")) * 20 ** i
    return num

def isbn_to_num(l):
    return int(l[ : -1])

#https://stackoverflow.com/questions/477486/how-to-use-a-decimal-range-step-value?rq=1
def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step
        
def read_comments(filename):
    # comments are in format
    # #key: value
    comments = dict()
    with open(filename, "r") as fin:
        for line in fin:
            if line[0] == "#":
                comment = line[1:-1].split(":")
                comments[comment[0]] = comment[1]
            else:
                break
    return comments


def main(dataset, filename, step):

    step = int(step)
    n = 800000000 / step
    resolution = 100000
    chunksize = int((n + resolution - 1) / resolution )

    x = np.arange(0, n, chunksize)

    b = 0
    to_num = None
    if "gene" in dataset:
        b = 4
        to_num = gene_to_num
    elif "protein" in dataset:
        b = 20
        to_num = protein_to_num
    elif "isbn" in dataset:
        to_num = isbn_to_num

    k = 0
    if "16" in dataset:
        k = 16
    elif "32" in dataset:
        k = 32
    elif "7" in dataset:
        k = 7

    y = []
    with open("./data/"+dataset+".txt", 'r') as f:
        y = list(map(to_num, f.readlines()[ : : chunksize * step]))
   
    plt.plot(x, y)

    comments = read_comments("./csv/query/"+filename+".csv")
    comments.update(read_comments("./csv/grouping/"+filename+".csv"))
    comments["avg"] = str(round(float(comments["avg"])))
    df = pd.read_csv("./csv/grouping/"+filename+".csv", comment="#")
    df = df[df["type"] == "group"]

    starts = [take_closest(x, s) for s in list(df["start"])]

    pivots = [y[np.where(x==s)[0][0]] for s in starts]

    width = b ** k / 8
    plt.vlines(
        starts,
        [p - width for p in pivots],
        [p + width for p in pivots],
        "red"
    )
    plt.text(
        .05, .6,
        "\n".join("{}: {}".format(k, v) for k, v in comments.items()),
        transform=plt.gca().transAxes,
        fontsize=10,
        color="white",
        bbox=dict(
            boxstyle="round",
            facecolor="royalblue",
            edgecolor='none'
        )
    )

    plt.savefig("./plots/groups/"+filename+".png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file')
    parser.add_argument('-s', '--step')
    parser.add_argument('-e', '--ethresh')
    parser.add_argument('-pt', '--pthresh')
    parser.add_argument('-fs', '--fstep')
    parser.add_argument('-bs', '--bstep')
    parser.add_argument('-ms', '--minsiz')

    args = parser.parse_args()

    #filename = "./grouping/csv/"
    filename = "_".join([
        args.file,
        args.step,
        args.ethresh,
        args.fstep,
        args.bstep,
        args.minsiz
    ])
    #filename += ".csv"

    main(args.file, filename, args.step)