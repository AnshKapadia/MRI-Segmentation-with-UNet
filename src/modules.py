from monai.transforms import MapTransform
import torch
import numpy as np
import nibabel as nib
from monai.networks.nets import UNet
import matplotlib.pyplot as plt
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Compose, Spacingd, Orientationd)

class PadToMultipleOf(MapTransform):
    """
    Pad image and label spatially to the next multiple of `k`.
    Padding is applied to the end of each dimension.
    """
    def __init__(self, keys, k=8, mode="constant"):
        super().__init__(keys)
        self.k = k
        self.mode = mode

    def __call__(self, data):
        for key in self.keys:
            img = data[key]
            spatial_shape = img.shape[1:]  # skip channel dim
            pad_sizes = []

            for dim_size in reversed(spatial_shape):
                next_multiple = int(np.ceil(dim_size / self.k) * self.k)
                pad_sizes.extend([0, next_multiple - dim_size])

            data[key] = torch.nn.functional.pad(
                img, pad_sizes, mode=self.mode
            )
        return data

t1ce_vis_transform = Compose([
    LoadImaged(keys=["t1c"]),
    EnsureChannelFirstd(keys=["t1c"]),
    Spacingd(keys=["t1c"],pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    Orientationd(keys=["t1c"],axcodes="RAS"),
    PadToMultipleOf(keys=["t1c"],k=8)
])
def overlay_segmentation(t1c, seg, slice_index, title, save_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(t1c[:, :, slice_index], cmap='gray')
    plt.imshow(np.ma.masked_where(seg[:, :, slice_index] == 0, seg[:, :, slice_index]), cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.title(title)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def save_nifti(pred_tensor, reference_nifti_path, output_path):
    reference = nib.load(reference_nifti_path)
    affine = reference.affine
    pred_np = pred_tensor.cpu().numpy().astype(np.uint8)
    pred_img = nib.Nifti1Image(pred_np, affine)
    nib.save(pred_img, output_path)
    

    
def get_unet_model():
    return UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=5,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2
    )

def save_checkpoint(model, path="checkpoints/unet.pt"):
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path="checkpoints/unet.pt", device="cpu"):
    model.load_state_dict(torch.load(path, map_location=device))
    return model
