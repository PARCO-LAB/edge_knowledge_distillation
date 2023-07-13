#!/usr/bin/python
# This file generates the index for the sampling

import random
import argparse
import math


def random_sampling(start, end, chunk):
    return sorted(random.sample(range(start, end+1), chunk))

def main():
    parser = argparse.ArgumentParser(description="Generate index for sampling", epilog="PARCO")
    parser.add_argument("--total-size",
                        default=20000,
                        type=int) 
    parser.add_argument("--window-size",
                        default=2000,
                        type=int)
    parser.add_argument("--chunk-size",
                        default=100,
                        type=int) 
    parser.add_argument("--output-folder-train",
                        default='random') 
    parser.add_argument("--output-folder-test",
                        default='window-2000-chunk-100') 
    
    args = parser.parse_args()

    # gather sample for testing. To add other form of sampling
    trains = []
    for train_id in range( math.ceil(args.total_size / args.window_size) - 1):
        window_start = train_id * args.window_size
        window_end = min(args.total_size, (train_id + 1) * args.window_size)

        sampling = random_sampling(window_start, window_end, args.chunk_size)
        with open(args.output_folder_train + '/chunk-' + str(train_id).zfill(3) + '.txt', 'w') as f:
            for s in sampling:
                f.write(str(s) + '\n')
        trains.append(sampling)
    
    # generate all the ids for test
    tests = []
    for test_id in range( math.ceil(args.total_size / args.window_size)):
        window_start = test_id * args.window_size
        window_end = min(args.total_size, (test_id + 1) * args.window_size)

        t = [i for i in range(window_start, window_end)]
        with open(args.output_folder_test + '/chunk-' + str(test_id).zfill(3) + '.txt', 'w') as f:
            for s in t:
                f.write(str(s) + '\n')
        tests.append(t)

if __name__ == '__main__':
    main()