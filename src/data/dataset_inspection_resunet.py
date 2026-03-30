import os
import numpy as np
import matplotlib.pyplot as plt
from preprocessing_resunet import (
    load_nii,
    normalize,
    resize_image,
    resize_mask,
    create_brats_masks,
    paper_roi_detection,
    get_bounding_box
)


PROJECT_ROOT = os.path.expanduser("~/Downloads/BraTS")

RAW_DIR  = os.path.join(PROJECT_ROOT, "BraTS_Datasets")
PROC_DIR = os.path.join(PROJECT_ROOT, "BraTS_processed_data")

DATASET_SUFFIX = "_processed_resunet"
EXPECTED_IMG_SHAPE = (3, 224, 224)
EXPECTED_MASK_SHAPE = (3, 224, 224)

print("PROJECT ROOT:", PROJECT_ROOT)
print("RAW DIR:", RAW_DIR)
print("PROC DIR:", PROC_DIR)

def collect_all_cases():
    cases = []
    for year in ["BraTS2018", "BraTS2019", "BraTS2020"]:
        year_path = os.path.join(RAW_DIR, year)
        for case in os.listdir(year_path):
            case_dir = os.path.join(year_path, case)
            if os.path.isdir(case_dir):
                cases.append(case_dir)
    return cases


def load_brats_case(case_path):
    files = os.listdir(case_path)

    def find_file(key):
        for f in files:
            if key in f.lower():
                return os.path.join(case_path, f)
        raise FileNotFoundError(f"{key} not found in {case_path}")

    t1   = load_nii(find_file("t1n"))
    t1ce = load_nii(find_file("t1c"))
    t2   = load_nii(find_file("t2w"))
    flair= load_nii(find_file("t2f"))
    seg  = load_nii(find_file("seg"))

    return t1, t1ce, t2, flair, seg


def find_tumor_slices(seg):
    return np.unique(np.where(seg > 0)[2])

def preprocess_slice(t1, t1ce, t2, flair, seg, s):

    t1_s    = normalize(t1[:, :, s])
    t1ce_s  = normalize(t1ce[:, :, s])
    t2_s    = normalize(t2[:, :, s])
    flair_s = normalize(flair[:, :, s])
    seg_s   = seg[:, :, s]

    roi_mask = paper_roi_detection(flair_s)
    bbox = get_bounding_box(roi_mask)

    if bbox is None:
        gt_mask = seg_s > 0
        bbox = get_bounding_box(gt_mask)
        if bbox is None:
            return None

    y_min, y_max, x_min, x_max = bbox

    t1_crop    = t1_s[y_min:y_max, x_min:x_max]
    t1ce_crop  = t1ce_s[y_min:y_max, x_min:x_max]
    t2_crop    = t2_s[y_min:y_max, x_min:x_max]
    flair_crop = flair_s[y_min:y_max, x_min:x_max]

    if min(t1_crop.shape) < 10:
        return None

    t1_r    = resize_image(t1_crop)
    t1ce_r  = resize_image(t1ce_crop)
    t2_r    = resize_image(t2_crop)
    flair_r = resize_image(flair_crop)

    flair_t2_mul = normalize(flair_r * t2_r)

    X = np.stack([t1ce_r, flair_t2_mul, t1_r], axis=0)

    WT_full, TC_full, ET_full = create_brats_masks(seg_s)

    WT_crop = WT_full[y_min:y_max, x_min:x_max]
    TC_crop = TC_full[y_min:y_max, x_min:x_max]
    ET_crop = ET_full[y_min:y_max, x_min:x_max]

    WT = (resize_mask(WT_crop) > 0.5).astype(np.float32)
    TC_raw = (resize_mask(TC_crop) > 0.5).astype(np.float32)
    ET = (resize_mask(ET_crop) > 0.5).astype(np.float32)

    TC = np.logical_or(TC_raw, ET).astype(np.float32)
    WT = np.logical_or(WT, TC).astype(np.float32)

    Y = np.stack([WT, TC, ET], axis=0)

    return X, Y


def visualize_sample(x, y):

    fig, ax = plt.subplots(1, 6, figsize=(24, 4))

    titles = [
        "T1ce",
        "FLAIR×T2",
        "T1",
        "WT Mask",
        "TC Mask",
        "ET Mask"
    ]

    for i in range(3):
        ax[i].imshow(x[i], cmap="gray")
        ax[i].set_title(titles[i])
        ax[i].axis("off")

    for i in range(3):
        ax[i+3].imshow(y[i], cmap="jet")
        ax[i+3].set_title(titles[i+3])
        ax[i+3].axis("off")

    plt.tight_layout()
    plt.show()


def inspect_random_slices(case_path, num_samples=3):
    t1, t1ce, t2, flair, seg = load_brats_case(case_path)
    tumor_slices = find_tumor_slices(seg)

    print(f"Total tumor slices: {len(tumor_slices)}")

    for _ in range(num_samples):
        s = np.random.choice(tumor_slices)
        sample = preprocess_slice(t1, t1ce, t2, flair, seg, s)

        if sample is None:
            continue

        x, y = sample
        print(f"Slice {s} → Input {x.shape}, Mask {y.shape}")
        visualize_sample(x, y)


def check_saved_dataset():

    print("\nDATASET FILE CHECK")

    for year in ["BraTS2018", "BraTS2019", "BraTS2020"]:

        dataset_name = f"{year}{DATASET_SUFFIX}"
        img_dir = os.path.join(PROC_DIR, dataset_name, "images")
        mask_dir = os.path.join(PROC_DIR, dataset_name, "masks")

        img_files = sorted(os.listdir(img_dir))
        mask_files = sorted(os.listdir(mask_dir))

        print(f"\n{dataset_name}")
        print("Images:", len(img_files))
        print("Masks :", len(mask_files))

        if len(img_files) != len(mask_files):
            print("Mismatch in counts")
            continue

        sample_img = np.load(os.path.join(img_dir, img_files[0]))
        sample_mask = np.load(os.path.join(mask_dir, mask_files[0]))

        print("Sample image shape:", sample_img.shape)
        print("Sample mask shape :", sample_mask.shape)

        if sample_img.shape != EXPECTED_IMG_SHAPE:
            print("Incorrect image shape")

        if sample_mask.shape != EXPECTED_MASK_SHAPE:
            print("Incorrect mask shape")


def check_class_balance():

    print("\nCLASS DISTRIBUTION")

    total_wt = total_tc = total_et = 0
    total_pixels = 0

    for year in ["BraTS2018", "BraTS2019", "BraTS2020"]:

        dataset_name = f"{year}{DATASET_SUFFIX}"
        mask_dir = os.path.join(PROC_DIR, dataset_name, "masks")

        for f in os.listdir(mask_dir):
            y = np.load(os.path.join(mask_dir, f))
            total_wt += y[0].sum()
            total_tc += y[1].sum()
            total_et += y[2].sum()
            total_pixels += y[0].size

    print("WT:", total_wt / total_pixels)
    print("TC:", total_tc / total_pixels)
    print("ET:", total_et / total_pixels)
    print("Background:", 1 - (total_wt / total_pixels))


def is_binary(mask):
    return np.all(np.isin(mask, [0, 1]))


def verify_mask_integrity(num_samples=2000):

    print("\nMASK INTEGRITY CHECK")

    for year in ["BraTS2018", "BraTS2019", "BraTS2020"]:

        dataset_name = f"{year}{DATASET_SUFFIX}"
        mask_dir = os.path.join(PROC_DIR, dataset_name, "masks")

        files = os.listdir(mask_dir)
        indices = np.random.choice(len(files),
                                   min(num_samples, len(files)),
                                   replace=False)

        violations = {
            "non_binary": 0,
            "et_not_in_tc": 0,
            "tc_not_in_wt": 0
        }

        for idx in indices:
            y = np.load(os.path.join(mask_dir, files[idx]))
            WT, TC, ET = y[0], y[1], y[2]

            if not is_binary(WT) or not is_binary(TC) or not is_binary(ET):
                violations["non_binary"] += 1

            if np.any(ET > TC):
                violations["et_not_in_tc"] += 1

            if np.any(TC > WT):
                violations["tc_not_in_wt"] += 1

        print(f"\n{dataset_name}")
        for k, v in violations.items():
            print(k, ":", v)

        if sum(violations.values()) == 0:
            print("MASKS VALID")
        else:
            print("MASK PROBLEMS DETECTED")


if __name__ == "__main__":

    case_paths = collect_all_cases()
    random_case = np.random.choice(case_paths)

    print("\nInspecting RAW case:", random_case)
    inspect_random_slices(random_case)

    check_saved_dataset()
    check_class_balance()
    verify_mask_integrity()