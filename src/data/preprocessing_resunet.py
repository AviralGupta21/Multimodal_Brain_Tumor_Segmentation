import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import median_filter, uniform_filter
from skimage.morphology import disk, opening, closing
from skimage.transform import resize
from skimage.filters import threshold_otsu


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR  = os.path.join(BASE_DIR, "BraTS_Datasets")
OUT_DIR  = os.path.join(BASE_DIR, "BraTS_processed_data")

TARGET_SIZE = (224, 224)


def load_nii(path):
    nii = nib.load(path)
    nii = nib.as_closest_canonical(nii)
    return nii.get_fdata().astype(np.float32)

def normalize(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def resize_image(img):
    return resize(img, TARGET_SIZE, order=1, preserve_range=True)

def resize_mask(mask):
    return resize(mask, TARGET_SIZE, order=0, preserve_range=True)

def create_brats_masks(seg):

    if np.max(seg) == 4:
        et_label = 4
    else:
        et_label = 3

    WT = np.isin(seg, [1, 2, et_label]).astype(np.float32)
    TC = np.isin(seg, [1, et_label]).astype(np.float32)
    ET = (seg == et_label).astype(np.float32)

    return WT, TC, ET

SELEM = disk(3)

def high_hat(img):
    return img - opening(img, SELEM)

def low_hat(img):
    return closing(img, SELEM) - img

def hat_transform(flair):
    th = high_hat(flair)
    bh = low_hat(flair)
    return (flair + th) - bh

def big_median_filter(img):
    return median_filter(img, size=25)

def multiply_with_flair(filtered, flair):
    return filtered * flair

def background_subtraction(flair):
    avg = uniform_filter(flair, size=150)
    inverted_flair = 1 - flair
    return inverted_flair - avg

def remove_background(step5, step6):
    return step6 - step5

def enhance_roi(img, flair):
    return img / (flair + 1e-8)

def final_cleanup(img):
    return closing(img, SELEM)

def paper_roi_detection(flair_slice):

    step3 = hat_transform(flair_slice)
    step4 = big_median_filter(step3)
    step5 = multiply_with_flair(step4, flair_slice)
    step6 = background_subtraction(flair_slice)
    step7 = remove_background(step5, step6)
    step8 = enhance_roi(step7, flair_slice)
    step9 = final_cleanup(step8)

    if np.std(step9) < 1e-6:
        return np.zeros_like(step9, dtype=np.uint8)

    thresh = threshold_otsu(step9)
    mask = step9 > thresh
    return (mask).astype(np.uint8)

def get_bounding_box(mask, padding=0):

    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None

    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()

    y_min = max(0, y_min - padding)
    y_max = min(mask.shape[0], y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(mask.shape[1], x_max + padding)

    return y_min, y_max + 1, x_min, x_max + 1


def preprocess_case(case_path):

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

    tumor_slices = np.unique(np.where(seg > 0)[2])
    samples = []

    for s in tumor_slices:

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
                continue

        y_min, y_max, x_min, x_max = bbox

        t1_crop    = t1_s[y_min:y_max, x_min:x_max]
        t1ce_crop  = t1ce_s[y_min:y_max, x_min:x_max]
        t2_crop    = t2_s[y_min:y_max, x_min:x_max]
        flair_crop = flair_s[y_min:y_max, x_min:x_max]

        if min(t1_crop.shape) < 10:
            continue

        t1_r    = resize_image(t1_crop)
        t1ce_r  = resize_image(t1ce_crop)
        t2_r    = resize_image(t2_crop)
        flair_r = resize_image(flair_crop)

        flair_t2_mul = normalize(flair_r * t2_r)

        X = np.stack([
            t1ce_r,          
            flair_t2_mul,    
            t1_r             
        ], axis=0)

        WT_full, TC_full, ET_full = create_brats_masks(seg_s)

        WT_crop = WT_full[y_min:y_max, x_min:x_max]
        TC_crop = TC_full[y_min:y_max, x_min:x_max]
        ET_crop = ET_full[y_min:y_max, x_min:x_max]

        WT = (resize_mask(WT_crop) > 0.5).astype(np.float32)
        TC_raw = (resize_mask(TC_crop) > 0.5).astype(np.float32)
        ET = (resize_mask(ET_crop) > 0.5).astype(np.float32)

        TC = np.logical_or(TC_raw, ET).astype(np.float32)
        WT = np.logical_or(WT, TC).astype(np.float32)

        if WT.sum() == 0 and TC.sum() == 0 and ET.sum() == 0:
            continue

        if np.any(ET > TC):
            continue
        if np.any(TC > WT):
            continue

        Y = np.stack([WT, TC, ET], axis=0)

        samples.append((X, Y))

    return samples

def process_year(year):

    raw_year_dir = os.path.join(RAW_DIR, year)
    out_img = os.path.join(OUT_DIR, f"{year}_processed_resunet/images")
    out_msk = os.path.join(OUT_DIR, f"{year}_processed_resunet/masks")

    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_msk, exist_ok=True)

    cases = [c for c in os.listdir(raw_year_dir)
             if os.path.isdir(os.path.join(raw_year_dir, c))]

    total_slices = 0

    for case in tqdm(cases, desc=year):
        case_path = os.path.join(raw_year_dir, case)
        samples = preprocess_case(case_path)

        for slice_idx, (X, Y) in enumerate(samples):
            filename = f"{case}_slice_{slice_idx:03d}.npy"
            np.save(os.path.join(out_img, filename), X)
            np.save(os.path.join(out_msk, filename), Y)
            total_slices += 1

    print(f"{year} done → {total_slices} slices")


if __name__ == "__main__":
    for year in ["BraTS2018", "BraTS2019", "BraTS2020"]:
        process_year(year)