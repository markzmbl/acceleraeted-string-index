import matplotlib.pyplot as plt
import numpy as np
import argparse


def gene_string_to_num(s):
    num = 0
    for i, c in enumerate(reversed(s)):
        num += (ord(c) - ord('A')) * 4 ** i
    return num

parser = argparse.ArgumentParser(description='Plot Data Set')
parser.add_argument('-f','--file', help='Input File', required=True)
args = vars(parser.parse_args())

file = args['file']

with open(file, 'r') as f:
    for i, line in enumerate(f):
        plt.plot(i, gene_string_to_num(line), 'ro')
        break

plt.savefig(file[ : file.find('.') ] + ".png")
