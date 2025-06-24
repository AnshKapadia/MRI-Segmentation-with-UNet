from monai.transforms import MapTransform
import torch
import numpy as np
from monai.networks.nets import UNet
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
