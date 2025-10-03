import os
import argparse
from tools.metrics import compute_wer, compute_cer

def read_txt_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--refs_dir", required=True, help="directory with reference txt files (uttid.txt)")
    p.add_argument("--preds_dir", required=True, help="directory with predicted txt files")
    args = p.parse_args()

    refs_files = {os.path.splitext(os.path.basename(f))[0]: f for f in [os.path.join(args.refs_dir, fn) for fn in os.listdir(args.refs_dir) if fn.endswith(".txt")]}
    preds_files = {os.path.splitext(os.path.basename(f))[0]: f for f in [os.path.join(args.preds_dir, fn) for fn in os.listdir(args.preds_dir) if fn.endswith(".txt")]}

    common = set(refs_files.keys()) & set(preds_files.keys())
    if len(common) == 0:
        print("No matching files found between refs and preds.")
        return

    sum_wer = 0.0
    sum_cer = 0.0
    count = 0
    for utt in sorted(common):
        ref = read_txt_file(refs_files[utt])
        pred = read_txt_file(preds_files[utt])
        wer = compute_wer(ref, pred)
        cer = compute_cer(ref, pred)
        sum_wer += wer
        sum_cer += cer
        count += 1
        print(f"{utt}\tWER={wer:.3f}\tCER={cer:.3f}\tREF={ref}\tPRED={pred}")

    print(f"Average WER: {sum_wer/count:.4f}, Average CER: {sum_cer/count:.4f} on {count} files")

if __name__ == "__main__":
    main()
