import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import multiprocessing as mp


def gene_string_to_num(s):
    global code
    num = 0
    for i, c in enumerate(reversed(s)):
        num += (code.index(c)) * 4 ** i
    return num

#parser = argparse.ArgumentParser(description='Plot Data Set')
#parser.add_argument('-f','--file', help='Input File', required=True)
#args = vars(parser.parse_args())
#
#file = args['file']
filename = '/home/markus/Documents/uni/bachelorarbeit/accelerated-string-index/gene/gene200normal.txt'
batch = -1

code = ['A', 'B', 'C', 'D']
f = plt.figure()


with open(filename, 'r') as fin:
    with open(filename[ : filename.find('.') ] + ".csv", 'w') as fout:
        count = 0
        lines = fin.readlines(batch)
        for i in range(0, 200000000, 500000):
            out = lines[i : i + 500000]
            out = [str(gene_string_to_num(g[ : -1]))+'\n' for g in out]
            fout.writelines(out)
            print(i)
