import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from BraTS.BraTS_python_files.unet_model import UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = "/home/drrenu/Downloads/BraTS/BraTS_processed_data"

def crop_to_brain(flair, mask):

    brain = flair > 0.05
    coords = np.argwhere(brain)

    if coords.shape[0] == 0:
        return flair, mask  

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    pad = 5
    y_min = max(0, y_min - pad)
    x_min = max(0, x_min - pad)
    y_max = min(flair.shape[0], y_max + pad)
    x_max = min(flair.shape[1], x_max + pad)

    flair_crop = flair[y_min:y_max, x_min:x_max]
    mask_crop = mask[:, y_min:y_max, x_min:x_max]

    return flair_crop, mask_crop

def get_best_sample(year):

    img_dir = os.path.join(BASE_DIR, f"BraTS{year}_processed", "images")
    mask_dir = os.path.join(BASE_DIR, f"BraTS{year}_processed", "masks")

    candidates = []

    for file in os.listdir(mask_dir):

        y = np.load(os.path.join(mask_dir, file))
        WT, TC, ET = y

        if WT.sum() > 0 and TC.sum() > 0 and ET.sum() > 0:
            score = WT.sum() + TC.sum() + ET.sum()
            candidates.append((score, file))

    candidates.sort(reverse=True)

    if year == 2020:
        idx = int(len(candidates) * 0.2) 
    elif year == 2019:
        idx = int(len(candidates) * 0.5)   
    else:  
        idx = int(len(candidates) * 0.8)   

    idx = min(idx, len(candidates)-1)

    selected_file = candidates[idx][1]

    x = np.load(os.path.join(img_dir, selected_file))
    y = np.load(os.path.join(mask_dir, selected_file))

    return x, y

@torch.no_grad()
def get_prediction(model, x):

    x = torch.tensor(x).unsqueeze(0).to(DEVICE)
    logits = model(x)

    pred = torch.sigmoid(logits)[0].cpu().numpy()
    pred = (pred > 0.5).astype(np.float32)

    return pred

def create_overlay(flair, mask):

    WT, TC, ET = mask

    edema = np.logical_and(WT == 1, TC == 0)
    tumor_core = np.logical_and(TC == 1, ET == 0)
    enhancing = ET == 1

    flair = (flair - flair.min()) / (flair.max() - flair.min() + 1e-8)
    overlay = np.stack([flair]*3, axis=-1)

    overlay[edema] = (1, 0, 0)
    overlay[enhancing] = (0, 1, 0)
    overlay[tumor_core] = (0, 0, 1)

    return overlay

def visualize_year(year):

    print(f"\nProcessing BraTS{year}")

    x, y = get_best_sample(year)

    flair = x[3]

    model = UNet().to(DEVICE)
    ckpt = f"checkpoints/best_TRAIN_model_year{year}_fold0.pth"

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    pred = get_prediction(model, x)

    flair_crop, y_crop = crop_to_brain(flair, y)
    _, pred_crop = crop_to_brain(flair, pred)

    gt_overlay = create_overlay(flair_crop, y_crop)
    pred_overlay = create_overlay(flair_crop, pred_crop)

    return gt_overlay, pred_overlay

def plot_final():

    years = [2020, 2019, 2018]

    fig, ax = plt.subplots(3, 2, figsize=(6, 8))  

    ax[0,0].set_title("Ground Truth", fontsize=13, fontweight='bold')
    ax[0,1].set_title("UNet", fontsize=13, fontweight='bold')

    for i, year in enumerate(years):

        gt, pred = visualize_year(year)

        ax[i, 0].imshow(gt)
        ax[i, 0].axis("off")

        ax[i, 1].imshow(pred)
        ax[i, 1].axis("off")

        ax[i,0].set_aspect('equal')
        ax[i,1].set_aspect('equal')

        ax[i,0].text(
            -0.25, 0.5,
            f"BraTS {year}",
            transform=ax[i,0].transAxes,
            fontsize=12,
            fontweight='bold',
            va='center'
        )

    plt.subplots_adjust(hspace=0.02, wspace=0.02)

    for a in ax.flat:
        for spine in a.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.3)

    fig.text(
        0.5, 0.02,
        "Blue: Non-Enhancing Tumor   |   Green: Enhancing Tumor   |   Red: Edema",
        ha='center',
        fontsize=10
    )

    plt.savefig("unet_final_visual.png", dpi=600, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    plot_final()