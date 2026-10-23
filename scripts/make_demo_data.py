"""Generate artificial data from a fixed seed; never read competition data."""
import argparse
import csv
from pathlib import Path
import random


def make_rows(count, prefix, seed):
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        # Alternate labels to keep both classes present even in small demos.
        label = i % 2
        rows.append({
            "Variant_ID": f"SYNTHETIC_{prefix}_{i:04d}",
            "NUM_1": round(rng.gauss(label * 1.4, 1), 6),
            "NUM_2": round(rng.gauss(-label * 0.9, 1), 6),
            "NUM_3": round(rng.uniform(-2, 2), 6),
            "CAT_1": rng.choice(("A", "B", "C")),
            "Label": label,
        })
    rng.shuffle(rows)
    return rows


def write_csv(path, rows, labeled=True):
    fields = [k for k in rows[0] if labeled or k != "Label"]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/demo"))
    args = parser.parse_args(argv)
    out = args.output_dir.resolve()
    files = [out / name for name in (
        "YARISMA_TRAIN_KANSER.csv", "synthetic_test.csv", "synthetic_unlabeled.csv")]
    if any(p.exists() for p in files):
        parser.error("Demo files already exist. Choose a new output folder; existing data is never overwritten.")
    out.mkdir(parents=True, exist_ok=True)
    train = make_rows(240, "TRAIN", 43)
    test = make_rows(80, "TEST", 44)
    write_csv(files[0], train)
    write_csv(files[1], test)
    write_csv(files[2], test, labeled=False)
    print(f"Synthetic demo created: {out} (240 train / 80 test; no real variants)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
