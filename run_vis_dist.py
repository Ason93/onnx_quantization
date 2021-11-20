import sys

from matplotlib.pyplot import get
from vision.visualize import *
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True, help="numpy data to show distribution")
    parser.add_argument("--axis", default=0, help = "which axis is to show")
    parser.add_argument("--output_path", required=True, help="picture to show data distirbution ")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    data = np.load(args.input_path)
    get_dist(data, args.axis, args.output_path)

if __name__ == '__main__':
    main()