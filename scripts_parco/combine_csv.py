#!/usr/bin/env python
import argparse
import pandas as pd
import os

def combine(folder, file):
    import glob
    # sorted for lexicographic ordering
    files = sorted(glob.glob(os.path.join(folder, "chunk-*.csv")))
    
    data = pd.DataFrame()

    for filename in files:
        data_new = pd.read_csv(filename, index_col=0)
        
        data = pd.concat([data, data_new], axis=0).reset_index(drop=True)
    data.to_csv(file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Combine CSV output", epilog="PARCO")
    parser.add_argument("input_folder") 
    parser.add_argument("output_file")
    args = parser.parse_args()
    combine(args.input_folder, args.output_file)