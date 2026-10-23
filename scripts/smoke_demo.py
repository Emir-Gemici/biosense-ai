"""Run the real UI helpers on synthetic inputs, then check initial Streamlit render.

No competition data is opened and the expensive Optuna pipeline is not launched.
Requires requirements.txt. This is a functionality check, not a model benchmark.
"""
import ast
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    fbeta_score, f1_score, roc_auc_score, accuracy_score, matthews_corrcoef,
    confusion_matrix, precision_score, recall_score, average_precision_score,
)
from catboost import CatBoostClassifier
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_demo_data import make_rows

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "apps/evaluate.py"
    source = ast.parse(path.read_text(encoding="utf-8"))
    # Load actual function definitions, without executing file upload widgets.
    functions = ast.Module(body=[n for n in source.body if isinstance(n, ast.FunctionDef)], type_ignores=[])
    namespace = dict(globals())
    exec(compile(functions, str(path), "exec"), namespace)
    train = pd.DataFrame(make_rows(80, "TRAIN", 43))
    test = pd.DataFrame(make_rows(30, "TEST", 44))
    test.loc[0, "CAT_1"] = "UNSEEN"
    train.loc[0, "NUM_3"] = np.nan
    test.loc[1, "NUM_3"] = np.nan
    features = ["NUM_1", "NUM_2", "NUM_3", "CAT_1"]
    xtr, xte, cats = namespace["kategorik_isle"](train[features], test[features])
    assert str(xte.loc[0, "CAT_1"]) == "EKSİK"
    assert list(xtr["CAT_1"].cat.categories) == list(xte["CAT_1"].cat.categories)
    for algorithm, parameters in (
        ("logistic", {"max_iter": 200, "random_state": 43}),
        ("catboost", {"iterations": 12, "depth": 3, "thread_count": 1,
                      "random_seed": 43, "allow_writing_files": False}),
    ):
        proba = namespace["model_egit_tahmin"](algorithm, parameters, xtr, train.Label.to_numpy(), xte, cats)
        assert proba.shape == (30,) and np.isfinite(proba).all()
        assert ((proba >= 0) & (proba <= 1)).all()
        metrics = namespace["tum_metrikler"](test.Label.to_numpy(), proba, 0.5)
        assert all(np.isfinite(v) for v in metrics.values())
        assert sum(metrics[k] for k in ("TP", "TN", "FP", "FN")) == 30
        print(f"PASS: {algorithm}, unseen category, missing values and metric computation")
    app = AppTest.from_file(str(path)).run(timeout=30)
    assert len(app.exception) == 0, [e.message for e in app.exception]
    assert len(app.title) == 1
    print("PASS: Streamlit initial render")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
