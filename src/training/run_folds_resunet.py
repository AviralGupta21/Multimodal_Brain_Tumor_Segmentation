import numpy as np
from train_resunet import train_one_fold  


def summarize(arr):
    arr = np.array(arr)
    return arr.mean(axis=0), arr.std(axis=0)


def run_five_folds(year):

    print("\n==============================")
    print(f"Running 5-Fold CV for BraTS{year} — ResUNet+")
    print("==============================")

    val_results = {
        "train_dice": [],
        "test_dice": [],
        "train_iou": [],
        "test_iou": [],
        "train_paper": [],
        "test_paper": [],
    }

    train_results = {
        "train_dice": [],
        "test_dice": [],
        "train_iou": [],
        "test_iou": [],
        "train_paper": [],
        "test_paper": [],
    }

    for fold in range(5):

        print(f"\nStarting Fold {fold+1}/5")

        try:
            fold_results = train_one_fold(year, fold)

        except RuntimeError as e:
            print("\nFold crashed due to runtime error.")
            print(str(e))
            continue

        if fold_results is None:
            print("Fold skipped due to OOM.")
            continue

        for key in val_results.keys():
            val_results[key].append(
                fold_results["val_model"][key]
            )

        for key in train_results.keys():
            train_results[key].append(
                fold_results["train_model"][key]
            )

    val_summary = {}
    train_summary = {}

    for key in val_results.keys():
        val_summary[key] = summarize(val_results[key])

    for key in train_results.keys():
        train_summary[key] = summarize(train_results[key])

    print("\n======================================")
    print(f"FINAL 5-FOLD RESULTS — BraTS{year} — ResUNet+")
    print("======================================")

    def print_block(title, summary):
        print(f"\n{title}")

        for metric in [
            "train_dice", "test_dice",
            "train_iou", "test_iou",
            "train_paper", "test_paper"
        ]:

            mean, std = summary[metric]

            print(f"{metric.upper()} → "
                  f"WT:{mean[0]:.3f}±{std[0]:.3f} "
                  f"TC:{mean[1]:.3f}±{std[1]:.3f} "
                  f"ET:{mean[2]:.3f}±{std[2]:.3f}")

    print_block("VALIDATION-SELECTED MODEL (Paper-like)", val_summary)
    print_block("TRAINING-SELECTED MODEL (Overfit)", train_summary)

    return {
        "val_model": {
            key: val_summary[key][0] for key in val_summary
        },
        "train_model": {
            key: train_summary[key][0] for key in train_summary
        },
    }


if __name__ == "__main__":
    run_five_folds(2018)