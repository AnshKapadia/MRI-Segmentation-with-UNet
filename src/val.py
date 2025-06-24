import os
import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm

from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

from dataloader import get_val_loader
from modules import get_unet_model, load_checkpoint

def save_nifti(pred_tensor, reference_nifti_path, output_path):
    reference = nib.load(reference_nifti_path)
    affine = reference.affine
    pred_np = pred_tensor.cpu().numpy().astype(np.uint8)
    pred_img = nib.Nifti1Image(pred_np, affine)
    nib.save(pred_img, output_path)

def run_val(model, device, val_path=r"C:\Users\anshk\MRI_segmentation\data\validation_data"):
    val_loader = get_val_loader(val_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_unet_model().to(device)
    model = load_checkpoint(model, path=r"C:\Users\anshk\MRI_segmentation\checkpoints\unet_brats.pt", device=device)
    model.eval()

    post_pred = AsDiscrete(argmax=True)
    os.makedirs("inference_outputs", exist_ok=True)

    for i, batch in enumerate(tqdm(val_loader)):
        case_id = f"case_{i:03d}"
        inputs = batch["image"].to(device)

        with torch.no_grad():
            outputs = sliding_window_inference(inputs, roi_size=(128, 128, 128), sw_batch_size=1, predictor=model)
            outputs = [post_pred(o) for o in decollate_batch(outputs)]

        pred = outputs[0][0]  # shape: (H, W, D)

        # Use one of the input channels as shape reference
        ref_path = batch["image"].meta["filename_or_obj"][0]
        case_id = os.path.basename(os.path.dirname(ref_path))
        out_path = os.path.join(r"C:\Users\anshk\MRI_segmentation\results", f"{case_id}_seg.nii.gz")
        save_nifti(pred, ref_path, out_path)
