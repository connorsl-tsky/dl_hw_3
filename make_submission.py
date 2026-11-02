"""Create submission.csv from prediction masks named 0.png through 19.png.

Usage: python make_submission.py /path/to/predicted_masks --output submission.csv
Pixels >=128 are interpreted as vessel. Supports PNG gray/RGB input.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image

def encode_rle(binary):
    flat = binary.astype(np.uint8).reshape(-1)  # C order
    idx = np.flatnonzero(np.diff(np.r_[0,flat,0]))
    starts = idx[::2] + 1
    lens = idx[1::2] - idx[::2]
    return ' '.join(str(x) for pair in zip(starts,lens) for x in pair)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('predicted_masks', type=Path)
    p.add_argument('--output', default='submission.csv',type=Path)
    args=p.parse_args()
    with args.output.open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['Id','RLE'])
        for idx in range(20):
            path=args.predicted_masks / f'{idx}.png'
            with Image.open(path) as im:
                mask=np.asarray(im.convert('L'))
            if mask.shape != (512,512):
                raise ValueError(f'{path} has shape {mask.shape}, expected 512x512')
            writer.writerow([idx,encode_rle(mask >= 128)])
    print(f'Saved {args.output}')

if __name__ == '__main__':main()
