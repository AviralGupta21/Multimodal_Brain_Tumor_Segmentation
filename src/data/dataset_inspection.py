import os
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import (
    load_nii,
    normalize,
    resize_image,
    resize_mask,
    create_brats_masks,
    paper_roi_detection,
    get_bounding_box,
    hat_transform,
    big_median_filter,
    multiply_with_flair,
    background_subtraction,
    remove_background,
    enhance_roi,
    final_cleanup
)

PROJECT_ROOT = "/home/drrenu/Downloads/BraTS"

RAW_DIR  = os.path.join(PROJECT_ROOT, "BraTS_Datasets")
PROC_DIR = os.path.join(PROJECT_ROOT, "BraTS_processed_data")

print("PROJECT ROOT:", PROJECT_ROOT)
print("RAW DIR:", RAW_DIR)
print("PROC DIR:", PROC_DIR)

def collect_all_cases():

    cases = []

    for year in ["BraTS2018", "BraTS2019", "BraTS2020"]:
        year_path = os.path.join(RAW_DIR, year)
        print("Scanning:", year_path)

        if not os.path.exists(year_path):
            print("WARNING: folder missing:", year_path)
            continue

        for case in os.listdir(year_path):
            case_dir = os.path.join(year_path, case)

            if os.path.isdir(case_dir) or os.path.islink(case_dir):
                cases.append(os.path.realpath(case_dir))

    print("Total cases found:", len(cases))
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

    t1_s    = (t1[:,:,s])
    t1ce_s  = (t1ce[:,:,s])
    t2_s    = (t2[:,:,s])
    flair_s = normalize(flair[:,:,s])
    seg_s   = seg[:,:,s]

    roi_mask = paper_roi_detection(flair_s)
    bbox = get_bounding_box(roi_mask)

    if bbox is None:
        gt_mask = seg_s > 0
        bbox = get_bounding_box(gt_mask)
        if bbox is None:return None

    y_min, y_max, x_min, x_max = bbox

    t1_crop    = t1_s[y_min:y_max, x_min:x_max]
    t1ce_crop  = t1ce_s[y_min:y_max, x_min:x_max]
    t2_crop    = t2_s[y_min:y_max, x_min:x_max]
    flair_crop = flair_s[y_min:y_max, x_min:x_max]

    t1_crop    = normalize(t1_crop)
    t1ce_crop  = normalize(t1ce_crop)
    t2_crop    = normalize(t2_crop)
    flair_crop = normalize(flair_crop)

    if min(t1_crop.shape) < 10:
        return None

    t1_r    = resize_image(t1_crop)
    t1ce_r  = resize_image(t1ce_crop)
    t2_r    = resize_image(t2_crop)
    flair_r = resize_image(flair_crop)

    WT_full, TC_full, ET_full = create_brats_masks(seg_s)

    WT_crop = WT_full[y_min:y_max, x_min:x_max]
    TC_crop = TC_full[y_min:y_max, x_min:x_max]
    ET_crop = ET_full[y_min:y_max, x_min:x_max]

    WT = (resize_mask(WT_crop) > 0.5).astype(np.float32)
    TC_raw = (resize_mask(TC_crop) > 0.5).astype(np.float32)
    ET = (resize_mask(ET_crop) > 0.5).astype(np.float32)

    TC = np.logical_or(TC_raw, ET).astype(np.float32)
    WT = np.logical_or(WT, TC).astype(np.float32)

    X = np.stack([t1_r, t1ce_r, t2_r, flair_r], axis=0)
    Y = np.stack([WT, TC, ET], axis=0)

    return X, Y

def visualize_sample(x, y):
    fig, ax = plt.subplots(1, 7, figsize=(28,4))
    
    titles = ['T1', 'T1ce', 'T2', 'FLAIR', 'WT Mask', 'TC Mask', 'ET Mask']

    for i in range(4):
        ax[i].imshow(x[i], cmap='gray')
        ax[i].set_title(titles[i])
        ax[i].axis('off')

    ax[4].imshow(y[0], cmap='jet')
    ax[4].set_title('WT Mask')
    ax[4].axis('off')

    ax[5].imshow(y[1], cmap='jet')
    ax[5].set_title('TC Mask')
    ax[5].axis('off')

    ax[6].imshow(y[2], cmap='jet')
    ax[6].set_title('ET Mask')
    ax[6].axis('off')

    plt.tight_layout()
    plt.show()

def visualize_roi_pipeline(flair_slice):

    flair = normalize(flair_slice)

    from skimage.morphology import disk, opening, closing
    selem = disk(3)

    imtophat = flair - opening(flair, selem)

    imbothat = closing(flair, selem) - flair

    hat = (flair + imtophat) - imbothat

    step_e = big_median_filter(hat)

    step_f = multiply_with_flair(step_e, flair)

    step_g = background_subtraction(flair)

    step_h = remove_background(step_f, step_g)

    step_i = enhance_roi(step_h, flair)

    final_roi = final_cleanup(step_i)

    low, high = np.percentile(final_roi, [2, 98])
    vis_roi = np.clip(final_roi, low, high)

    vis_roi = (vis_roi - vis_roi.min()) / (vis_roi.max() - vis_roi.min() + 1e-8)

    from skimage.filters import threshold_otsu
    thresh = threshold_otsu(vis_roi)
    roi_mask = vis_roi > thresh
    images = [
        flair,
        imtophat,
        imbothat,
        hat,
        step_f,
        step_h,
        step_i,
        roi_mask
    ]

    labels = [
        "a) FLAIR",
        "b) imtophat",
        "c) imbothat",
        "d) hat transform",
        "e) multiply",
        "f) background removed",
        "g) enhanced ROI",
        "h) final mask"
    ]

    fig, ax = plt.subplots(2,4, figsize=(12,6))

    for i in range(8):

        r = i // 4
        c = i % 4

        img = images[i]

        if img.dtype == bool:
            ax[r,c].imshow(img, cmap='gray')
        elif i == 0 or i == 3:  
            ax[r,c].imshow(img, cmap='gray')
        else:
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            ax[r,c].imshow(img, cmap='gray')
        ax[r,c].set_title(labels[i], fontsize=10)
        ax[r,c].axis('off')

    plt.tight_layout()
    plt.savefig("roi_pipeline_figure.png", dpi=600, bbox_inches="tight")
    plt.show()

def inspect_random_slices(case_path, num_samples=3):

    case_path = os.path.realpath(case_path)

    t1, t1ce, t2, flair, seg = load_brats_case(case_path)
    tumor_slices = find_tumor_slices(seg)

    if len(tumor_slices) == 0:
        print("No tumor slices found in this case.")
        return

    print(f"Total tumor slices in case: {len(tumor_slices)}")

    example_slice = tumor_slices[np.argmax(np.sum(seg[:,:,tumor_slices] > 0, axis=(0,1)))]
    print("\nShowing preprocessing pipeline for slice:", example_slice)
    visualize_roi_pipeline(flair[:,:,example_slice])

    for _ in range(num_samples):

        slice_id = np.random.choice(tumor_slices)

        sample = preprocess_slice(t1, t1ce, t2, flair, seg, slice_id)

        if sample is None:
            continue

        x, y = sample
        print(f"Slice {slice_id} → Input {x.shape}, Mask {y.shape}")

        visualize_sample(x, y)

def visualize_saved_sample():
    year = np.random.choice([
        "BraTS2018_processed",
        "BraTS2019_processed",
        "BraTS2020_processed"
    ])
    img_dir = os.path.join(PROC_DIR, year, "images")
    mask_dir = os.path.join(PROC_DIR, year, "masks")

    file = np.random.choice(os.listdir(img_dir))

    x = np.load(os.path.join(img_dir, file))
    y = np.load(os.path.join(mask_dir, file))

    visualize_sample(x, y)

def check_saved_dataset():

    print("\nDATASET FILE CHECK")

    for year in ["BraTS2018_processed",
                 "BraTS2019_processed",
                 "BraTS2020_processed"]:

        img_dir = os.path.join(PROC_DIR, year, "images")
        mask_dir = os.path.join(PROC_DIR, year, "masks")

        img_files = sorted(os.listdir(img_dir))
        mask_files = sorted(os.listdir(mask_dir))

        print(f"\n{year}")
        print("Images:", len(img_files))
        print("Masks :", len(mask_files))

        if len(img_files) != len(mask_files):
            print(" Mismatch in image/mask counts")
            continue

        mismatches = sum(i != m for i, m in zip(img_files, mask_files))
        if mismatches > 0:
            print(" Filename mismatch detected")
        else:
            print(" Filenames aligned")

        sample_img = np.load(os.path.join(img_dir, img_files[0]))
        sample_mask = np.load(os.path.join(mask_dir, mask_files[0]))

        print("Sample image shape:", sample_img.shape)
        print("Sample mask shape :", sample_mask.shape)

        if sample_img.shape != (4, 240, 240):
            print(" Image shape incorrect")

        if sample_mask.shape != (3, 240, 240):
            print(" Mask shape incorrect")

def check_class_balance(num_samples=2000):

    print("\nCLASS DISTRIBUTION")

    total_wt = total_tc = total_et = 0
    total_pixels = 0

    for year in ["BraTS2018_processed",
                 "BraTS2019_processed",
                 "BraTS2020_processed"]:

        mask_dir = os.path.join(PROC_DIR, year, "masks")
        files = sorted(os.listdir(mask_dir))

        indices = range(len(files))

        for idx in indices:
            y = np.load(os.path.join(mask_dir, files[idx]))
            total_wt += y[0].sum()
            total_tc += y[1].sum()
            total_et += y[2].sum()
            total_pixels += y[0].size

    print("WT:", total_wt / total_pixels)
    print("TC:", total_tc / total_pixels)
    print("ET:", total_et / total_pixels)
    print("Background:", 1 - (total_wt/total_pixels))

def is_binary(mask):
    return np.all(np.isin(mask, [0, 1]))


def verify_mask_integrity(num_samples=2000):

    print("\nMASK INTEGRITY CHECK")

    for year in ["BraTS2018_processed",
                 "BraTS2019_processed",
                 "BraTS2020_processed"]:

        mask_dir = os.path.join(PROC_DIR, year, "masks")
        files = sorted(os.listdir(mask_dir))

        indices = np.random.choice(len(files),
                                   min(num_samples, len(files)))

        violations = {
            "empty_masks": 0,
            "et_not_in_tc": 0,
            "tc_not_in_wt": 0,
            "non_binary": 0
        }

        eps = 1e-6

        for idx in indices:
            y = np.load(os.path.join(mask_dir, files[idx]))
            WT, TC, ET = y[0], y[1], y[2]

            if not is_binary(WT): violations["non_binary"] += 1
            if not is_binary(TC): violations["non_binary"] += 1
            if not is_binary(ET): violations["non_binary"] += 1

            if WT.sum() == 0:
                violations["empty_masks"] += 1

            if np.any(ET > TC + eps):
                violations["et_not_in_tc"] += 1

            if np.any(TC > WT + eps):
                violations["tc_not_in_wt"] += 1

        print(f"\n{year}")
        for k,v in violations.items():
            print(k, ":", v)

        if sum(violations.values()) == 0:
            print(" MASKS VALID")
        else:
            print(" MASK PROBLEMS DETECTED")

if __name__ == "__main__":
    case_paths = collect_all_cases()
    random_case = np.random.choice(case_paths)

    print("Inspecting case:", random_case)

    inspect_random_slices(random_case)
    check_saved_dataset()
    check_class_balance()
    verify_mask_integrity()