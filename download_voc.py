"""
Download PASCAL VOC 2007 and 2012 datasets.

Creates the following directory structure under DATA_ROOT:
    data/
      VOCdevkit/
        VOC2007/
          Annotations/
          ImageSets/
          JPEGImages/
        VOC2012/
          ...

Usage
-----
    python download_voc.py [--data_root ./data]
"""

import argparse
import os
import tarfile
import urllib.request
import sys

import config

URLS = {
    "VOC2007_trainval": (
        "https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar",
        "VOCtrainval_06-Nov-2007.tar",
    ),
    "VOC2007_test": (
        "https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar",
        "VOCtest_06-Nov-2007.tar",
    ),
    "VOC2012_trainval": (
        "https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar",
        "VOCtrainval_11-May-2012.tar",
    ),
}

# Fallback mirror (official Oxford server).
URLS_FALLBACK = {
    "VOC2007_trainval": (
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "VOCtrainval_06-Nov-2007.tar",
    ),
    "VOC2007_test": (
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
        "VOCtest_06-Nov-2007.tar",
    ),
    "VOC2012_trainval": (
        "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        "VOCtrainval_11-May-2012.tar",
    ),
}


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct}%)")
    else:
        mb = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  {mb:.1f} MB")
    sys.stdout.flush()


def download_and_extract(url: str, filename: str, root: str) -> None:
    os.makedirs(root, exist_ok=True)
    filepath = os.path.join(root, filename)

    if not os.path.isfile(filepath):
        print(f"Downloading {filename} ...")
        try:
            urllib.request.urlretrieve(url, filepath, reporthook=_progress_hook)
        except Exception:
            print(f"\n  Primary URL failed, trying fallback ...")
            raise
        print()

    print(f"Extracting {filename} ...")
    with tarfile.open(filepath) as tar:
        tar.extractall(path=root)
    print("  Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PASCAL VOC")
    parser.add_argument("--data_root", type=str, default=config.DATA_ROOT)
    args = parser.parse_args()

    for key in URLS:
        url, fname = URLS[key]
        try:
            download_and_extract(url, fname, args.data_root)
        except Exception:
            # Try fallback.
            url_fb, fname_fb = URLS_FALLBACK[key]
            print(f"  Trying fallback URL for {key} ...")
            download_and_extract(url_fb, fname_fb, args.data_root)

    # Verify.
    for year in ["2007", "2012"]:
        p = os.path.join(args.data_root, "VOCdevkit", f"VOC{year}")
        if os.path.isdir(p):
            print(f"VOC{year}: OK  ({p})")
        else:
            print(f"VOC{year}: MISSING  ({p})")

    print("\nDataset download complete.")


if __name__ == "__main__":
    main()
