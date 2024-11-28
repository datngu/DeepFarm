#!/usr/bin/env python

import argparse
import os, sys
import shutil

parser = argparse.ArgumentParser(description = "Rename peak files by remove some parterns)", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--files", nargs='+', help = "input peaks ", required = True)
parser.add_argument("--strings", nargs='+', help = "string parterns to remove from the names", required = True)
parser.add_argument("--ext", default = ".bed", help = "output file extension")


args = parser.parse_args()

files = args.files
strings = args.strings
ext = args.ext

# files = ['AtlanticSalmon_ATAC_Brain_Immature_Female_R3.mLb.clN_peaks.broadPeak', 'AtlanticSalmon_ATAC_Brain_Immature_Female_R2.mLb.clN_peaks.broadPeak']
# strings = ['AtlanticSalmon_', '.mLb.clN_peaks.broadPeak']
# ext = '.bed'

for file in files:
    old = file
    for s in strings:
        file = file.replace(s, '')
    
    new = file + ext
    # print(old)
    # print(new)
    # shutil.copyfile(old, new)
    os.rename(old, new)
    


# ./bin/rename.py --files /Users/datn/DATA_ANALYSES/OCR_prediction/data/salmon/*broadPeak --strings AtlanticSalmon_ .mLb.clN_peaks.broadPeak _peaks.broadPeak