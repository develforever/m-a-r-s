"""Pobiera wektory GloVe 6B (Stanford NLP) do data/.

Kod projektu potrzebuje tylko data/glove.6B.50d.txt (domyslna sciezka
w src/mars_cl_semantic.py). Archiwum zip zawiera tez 100/200/300d.

Uzycie:  python scripts/download_glove.py
"""
import os
import sys
import urllib.request
import zipfile

URLS = [  # kolejno probowane mirrory (zrodlo: nlp.stanford.edu/projects/glove)
    "https://nlp.stanford.edu/data/glove.6B.zip",
    "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip",
    "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip",
]
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TARGET = os.path.join(DATA_DIR, "glove.6B.50d.txt")
ZIP_PATH = os.path.join(DATA_DIR, "glove.6B.zip")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(TARGET):
        print(f"OK: {TARGET} juz istnieje.")
        return
    if not os.path.exists(ZIP_PATH):
        err = None
        for url in URLS:
            try:
                print(f"Pobieram {url} (~822 MB, moze potrwac)...")
                urllib.request.urlretrieve(url, ZIP_PATH, _progress)
                print()
                break
            except Exception as e:  # timeout/404 -> nastepny mirror
                err = e
                print(f"\n  nie wyszlo ({e}), probuje nastepny mirror...")
        else:
            raise SystemExit(f"Wszystkie mirrory zawiodly: {err}")
    print("Rozpakowuje glove.6B.50d.txt...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extract("glove.6B.50d.txt", DATA_DIR)
    os.remove(ZIP_PATH)
    print(f"OK: {TARGET}")


def _progress(blocks, block_size, total):
    if total > 0:
        pct = min(100, blocks * block_size * 100 // total)
        sys.stdout.write(f"\r  {pct}%")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
