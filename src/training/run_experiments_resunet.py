from run_folds_resunet import run_five_folds

def print_block(title, results):

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    print(f"""
TRAIN Dice       → WT:{results['train_dice'][0]:.3f} TC:{results['train_dice'][1]:.3f} ET:{results['train_dice'][2]:.3f}
TRAIN True IoU   → WT:{results['train_iou'][0]:.3f} TC:{results['train_iou'][1]:.3f} ET:{results['train_iou'][2]:.3f}
TRAIN Paper Jacc → WT:{results['train_paper'][0]:.3f} TC:{results['train_paper'][1]:.3f} ET:{results['train_paper'][2]:.3f}

TEST  Dice       → WT:{results['test_dice'][0]:.3f} TC:{results['test_dice'][1]:.3f} ET:{results['test_dice'][2]:.3f}
TEST  True IoU   → WT:{results['test_iou'][0]:.3f} TC:{results['test_iou'][1]:.3f} ET:{results['test_iou'][2]:.3f}
TEST  Paper Jacc → WT:{results['test_paper'][0]:.3f} TC:{results['test_paper'][1]:.3f} ET:{results['test_paper'][2]:.3f}
""")


def run_all_experiments():

    years = [2018, 2019, 2020]
    all_results = {}

    for year in years:

        print("\n\n########################################")
        print(f"Running ResUNet+ experiments for BraTS{year}")
        print("########################################")

        try:
            results = run_five_folds(year)
            all_results[year] = results

        except Exception as e:
            print(f"\nExperiment crashed for BraTS{year}")
            print(str(e))
            continue

    print("\n\n======================================")
    print("FINAL PAPER-STYLE RESULTS — ResUNet+")
    print("======================================")

    for year in years:

        if year not in all_results:
            continue

        print(f"\n\nBraTS{year} RESULTS")

        print_block(
            "VALIDATION-SELECTED MODEL (Paper-like Selection)",
            all_results[year]["val_model"]
        )

        print_block(
            "TRAINING-SELECTED MODEL (Overfit Selection)",
            all_results[year]["train_model"]
        )


if __name__ == "__main__":
    run_all_experiments()