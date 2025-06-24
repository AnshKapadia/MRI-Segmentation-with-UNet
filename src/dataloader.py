import os
import glob
from modules import PadToMultipleOf
from monai.transforms import (
    LoadImaged,SpatialPadd, EnsureChannelFirstd, Compose, Spacingd, Orientationd,
    ScaleIntensityRanged, CropForegroundd, RandFlipd, RandShiftIntensityd, ToTensord
)
from monai.data import CacheDataset, DataLoader, Dataset

def get_training_loader(train_dir, batch_size=1, num_workers=2):
    case_dirs = sorted(glob.glob(os.path.join(train_dir, "BraTS*")))
    data_dicts = []

    for case in case_dirs:
        files = {
            "t1": glob.glob(os.path.join(case, "*t1n.nii.gz")),
            "t1ce": glob.glob(os.path.join(case, "*t1c.nii.gz")),
            "t2": glob.glob(os.path.join(case, "*t2w.nii.gz")),
            "flair": glob.glob(os.path.join(case, "*t2f.nii.gz")),
            "label": glob.glob(os.path.join(case, "*seg.nii.gz"))
        }

        if all(len(v) == 1 for v in files.values()):
            data_dicts.append({
                "image": [files["t1"][0], files["t1ce"][0], files["t2"][0], files["flair"][0]],
                "label": files["label"][0]
            })
    for i, case in enumerate(data_dicts):
        if len(case["image"]) != 4:
            print(f"Broken at case {i}: {case['image']}")

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=300, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        PadToMultipleOf(keys=["image", "label"], k=8),
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.5),
        ToTensord(keys=["image", "label"])
    ])

    train_ds = CacheDataset(data=data_dicts, transform=train_transforms, cache_num=200, num_workers=num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_loader

def get_val_loader(val_dir, num_workers=2):
    case_dirs = sorted(glob.glob(os.path.join(val_dir, "BraTS*")))
    data_dicts = []

    for case in case_dirs:
        files = {
            "t1": glob.glob(os.path.join(case, "*t1n.nii.gz")),
            "t1ce": glob.glob(os.path.join(case, "*t1c.nii.gz")),
            "t2": glob.glob(os.path.join(case, "*t2w.nii.gz")),
            "flair": glob.glob(os.path.join(case, "*t2f.nii.gz")),
        }

        if all(len(v) == 1 for v in files.values()):
            data_dicts.append({
                "image": [files["t1"][0], files["t1ce"][0], files["t2"][0], files["flair"][0]],
            })

    val_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=300, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image"),
        PadToMultipleOf(keys=["image"], k=8),
        ToTensord(keys=["image"])
    ])

    val_ds = Dataset(data=data_dicts, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers)
    return val_loader