#/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import math
import multiprocessing as mp

def to_num(l):
    return int(l[ : -1])

def main():

    n = 200000000
    x = np.linspace(0, n, 1000)
    pool = mp.Pool(10)
    step = 1000

    df = pd.read_csv("output.csv")
    starts = df["start"]
    
    with open("./gene/gene200normal.csv", 'r') as f:
        lines = f.readlines()[ : : step]

    lines = pool.map(to_num, lines)
    x = np.arange(0, n, step)
    plt.plot(x, lines)
#
#
#    width = 4 ** 16 / 8
#
#    plt.vlines(
#        starts,
#        [p - width for p in pivots],
#        [p + width for p in pivots],
#        "red"
#    )


    plt.show()

if __name__ == "__main__":
    main()