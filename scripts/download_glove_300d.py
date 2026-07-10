"""Pobiera wektory GloVe 6B i rozpakowuje wariant 300d do data/.

Osobny skrypt (J4): download_glove.py kasuje zip po ekstrakcji 50d,
wiec 300d wymaga ponownego pobrania (~822 MB). Ten skrypt rozpakowuje
OBA wymiary (50d, jesli brakuje, i 300d) i dopiero potem kasuje zip.

Uzycie:  python scripts/download_glove_300d.py
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
TARGETS = ["glove.6B.50d.txt", "glove.6B.300d.txt"]
ZIP_PATH = os.path.join(DATA_DIR, "glove.6B.zip")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    missing = [t for t in TARGETS
               if not os.path.exists(os.path.join(DATA_DIR, t))]
    if not missing:
        print("OK: 50d i 300d juz istnieja.")
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
    with zipfile.ZipFile(ZIP_PATH) as z:
        for t in missing:
            print(f"Rozpakowuje {t}...")
            z.extract(t, DATA_DIR)
    os.remove(ZIP_PATH)
    for t in TARGETS:
        print(f"OK: {os.path.join(DATA_DIR, t)}")


def _progress(blocks, block_size, total):
    if total > 0:
        pct = min(100, blocks * block_size * 100 // total)
        sys.stdout.write(f"\r  {pct}%")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
