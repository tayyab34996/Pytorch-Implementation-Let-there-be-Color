"""Convert test images to grayscale, colorize them with the trained model,
and write comparisons + MSE/PSNR metrics.

Usage:
  conda activate colorize-fresh
  python SOURCE/test_gray_and_colorize.py

This script creates the following directories (if not present):
- `DATASET/Gland/test_gray/`     : grayscale copies of test images
- `RESULT/Gland/colorized_from_gray/` : colorized outputs
- `RESULT/Gland/comparisons_gray/` : side-by-side original | colorized images
- `LOGS/Gland/resnet34_unet_metrics_gray_vs_colorized.txt` : per-image MSE/PSNR
"""
import os
import glob
import cv2
import numpy as np
import math
import config

from colorize_image import colorize_image


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def convert_test_to_gray():
    src_dir = os.path.join('DATASET', config.DATASET, config.TEST_DIR)
    gray_dir = os.path.join('DATASET', config.DATASET, 'test_gray')
    ensure_dir(gray_dir)
    patterns = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(src_dir, pat)))
    files = sorted(files)
    for p in files:
        fname = os.path.basename(p)
        img = cv2.imread(p)
        if img is None:
            print('Failed to read', p)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        outp = os.path.join(gray_dir, fname)
        cv2.imwrite(outp, gray)
    return gray_dir, files


def make_comparisons_and_metrics(orig_files, color_dir, save_comparisons=False):
    comp_dir = os.path.join('RESULT', config.DATASET, 'comparisons_gray')
    metrics_file = os.path.join('LOGS', config.DATASET, 'resnet34_unet_metrics_gray_vs_colorized.txt')
    ensure_dir(os.path.dirname(metrics_file))
    rows = []
    for p in orig_files:
        fname = os.path.basename(p)
        orig = cv2.imread(p)
        colp = os.path.join(color_dir, fname)
        col = cv2.imread(colp)
        if orig is None or col is None:
            print('Missing pair for', fname)
            continue
        if orig.shape != col.shape:
            col = cv2.resize(col, (orig.shape[1], orig.shape[0]))
        mse = float(np.mean((orig.astype(np.float32) - col.astype(np.float32)) ** 2))
        psnr = 10.0 * math.log10((255.0 ** 2) / mse) if mse > 0 else 100.0
        rows.append((fname, p, mse, psnr))
        # write comparison image (side-by-side)
        h, w = orig.shape[:2]
        # resize long side to 512 for clearer grids
        if w >= h:
            target = (512, int(512 * h / w))
        else:
            target = (int(512 * w / h), 512)
        orig_r = cv2.resize(orig, target)
        col_r = cv2.resize(col, target)
        comp = cv2.hconcat([orig_r, col_r])
        if save_comparisons:
            ensure_dir(comp_dir)
            comp_path = os.path.join(comp_dir, fname.rsplit('.', 1)[0] + '_comp.jpg')
            cv2.imwrite(comp_path, comp)

    with open(metrics_file, 'w') as g:
        g.write('file,orig_path,MSE,PSNR\n')
        for r in rows:
            g.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")
    print('Wrote metrics to', metrics_file)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert test images to grayscale and colorize.')
    parser.add_argument('--select', '-s', help='Filename (in DATASET test) to select and process')
    parser.add_argument('--random', '-r', action='store_true', help='Select a random file from test set')
    parser.add_argument('--all', '-a', action='store_true', help='Process all test images (default)')
    parser.add_argument('--save-comparisons', action='store_true', help='Save side-by-side comparison images to RESULTS (disabled by default)')
    args = parser.parse_args()

    print('Converting test set to grayscale...')
    gray_dir, orig_files = convert_test_to_gray()
    print('Grayscale images saved to', gray_dir)

    out_dir = os.path.join('RESULT', config.DATASET, 'colorized_from_gray')
    ensure_dir(out_dir)

    if args.select or args.random:
        # pick one file
        import random
        if args.select:
            sel = args.select
            # accept either basename or full path
            candidates = [p for p in orig_files if os.path.basename(p) == sel or p == sel]
            if not candidates:
                print('Selected file not found in test set:', sel)
                return
            orig_path = candidates[0]
        else:
            orig_path = random.choice(orig_files)
        fname = os.path.basename(orig_path)
        gray_path = os.path.join(gray_dir, fname)
        out_path = os.path.join(out_dir, fname)
        print('Colorizing single file:', fname)
        colorize_image(gray_path, out_path, checkpoint=None, batch=False)
        print('Generating metric for', fname)
        make_comparisons_and_metrics([orig_path], out_dir, save_comparisons=args.save_comparisons)
    else:
        # default: process all
        print('Running batch colorization...')
        # use the colorize_image batch mode to process the grayscale folder
        colorize_image(gray_dir, out_dir, checkpoint=None, batch=True)
        print('Generating metrics...')
        make_comparisons_and_metrics(orig_files, out_dir, save_comparisons=args.save_comparisons)
    print('Done.')


if __name__ == '__main__':
    main()
