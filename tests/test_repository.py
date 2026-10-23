"""Exercise the publication boundary, synthetic data and launch paths."""
import csv
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


audit = module("audit", ROOT / "scripts/check_public_repo.py")
demo = module("demo", ROOT / "scripts/make_demo_data.py")


class DataProtectionTests(unittest.TestCase):
    def test_only_explicit_files(self):
        for path in ("data/train.csv", "leak.json", "model.pkl", "backup.zip", "docs/rows.md"):
            self.assertTrue(audit.check_blob(path, b"sample"), path)
        self.assertFalse(audit.check_blob("README.md", b"Documentation"))

    def test_binary_symlink_and_credentials(self):
        self.assertTrue(audit.check_blob("README.md", b"\0\xff"))
        self.assertTrue(audit.check_blob("README.md", b"target", "120000"))
        fake = ("gh" + "p_" + "a" * 36).encode()
        self.assertTrue(audit.check_blob("README.md", fake))

    def test_gitignore_covers_data_and_generated_artifacts(self):
        paths = ["data/train.csv", "datasets/payload.json", "sample.csv", "sample.CSV",
                 "runs/result.json", "model.pkl", "archive.rar", ".env", "docs/unreviewed.md"]
        process = subprocess.run(["git", "check-ignore", "--stdin"], cwd=ROOT,
                                 input="\n".join(paths) + "\n", text=True, capture_output=True)
        self.assertEqual(set(paths), set(process.stdout.splitlines()))

    def test_staged_content_and_deleted_historical_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*args):
                return subprocess.run(["git", "-C", str(root), *args], check=True,
                                      capture_output=True, env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1"})

            git("init")
            git("config", "user.name", "Repository test")
            git("config", "user.email", "test@example.invalid")
            (root / "README.md").write_text("gh" + "p_" + "a" * 36, encoding="utf-8")
            git("add", "README.md")
            (root / "README.md").write_text("Clean working file", encoding="utf-8")
            self.assertTrue(audit.check_index(root)[1], "Must inspect index, not working copy")
            git("add", "README.md")
            (root / "private.csv").write_text("synthetic fixture", encoding="utf-8")
            git("add", "private.csv")
            self.assertTrue(audit.check_index(root)[1])
            git("-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", "commit", "-m", "fixture")
            git("rm", "private.csv")
            git("-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", "commit", "-m", "remove fixture")
            self.assertFalse(audit.check_index(root)[1])
            self.assertTrue(audit.check_history(root)[2], "Deleted data must still be found in history")


class DemoTests(unittest.TestCase):
    def test_deterministic_disjoint_balanced_demo(self):
        a = demo.make_rows(240, "TRAIN", 43)
        b = demo.make_rows(80, "TEST", 44)
        self.assertEqual(a, demo.make_rows(240, "TRAIN", 43))
        self.assertFalse({r["Variant_ID"] for r in a} & {r["Variant_ID"] for r in b})
        self.assertEqual(sum(r["Label"] for r in a), 120)

    def test_demo_files_and_launcher_without_training(self):
        with tempfile.TemporaryDirectory(prefix="biosense demo ") as directory:
            data = Path(directory) / "data"
            output = Path(directory) / "runs"
            self.assertEqual(demo.main(["--output-dir", str(data)]), 0)
            with (data / "synthetic_unlabeled.csv").open(encoding="utf-8") as handle:
                self.assertNotIn("Label", csv.DictReader(handle).fieldnames)
            with self.assertRaises(SystemExit):
                demo.main(["--output-dir", str(data)])
            base = [sys.executable, str(ROOT / "train.py"), "kanser", "--data-dir", str(data)]
            result = subprocess.run(base + ["--output-dir", str(output), "--dry-run"], capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(output.exists(), "Dry run must not create artifacts")
            overlap = subprocess.run(base + ["--output-dir", str(data), "--dry-run"], capture_output=True)
            self.assertNotEqual(overlap.returncode, 0)
            missing = subprocess.run([sys.executable, str(ROOT / "train.py"), "cftr",
                                      "--data-dir", str(data), "--dry-run"], capture_output=True)
            self.assertNotEqual(missing.returncode, 0)


if __name__ == "__main__":
    unittest.main()
