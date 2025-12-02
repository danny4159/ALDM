import os
import numpy as np

import torch
from torch.utils.data import Dataset

from monai import transforms
from monai.data import Dataset as MonaiDataset


def get_transforms(phase="train", spatial_size=(144, 192, 144)):
    modalities = ["mr", "ct"]

    if phase == "train":
        train_transforms = transforms.Compose(
            [
                transforms.RandFlipd(keys=modalities, prob=0.1, spatial_axis=0, allow_missing_keys=True),
                transforms.RandFlipd(keys=modalities, prob=0.1, spatial_axis=1, allow_missing_keys=True),
                transforms.RandFlipd(keys=modalities, prob=0.1, spatial_axis=2, allow_missing_keys=True),
            ]
        )
    
    return transforms.Compose(
        [
            transforms.LoadImaged(keys=modalities, allow_missing_keys=True),
            transforms.AddChanneld(keys=modalities, allow_missing_keys=True),
            transforms.Orientationd(keys=modalities, axcodes="RAS", allow_missing_keys=True),
            transforms.EnsureTyped(keys=modalities, allow_missing_keys=True),
            transforms.CropForegroundd(keys=modalities, source_key="mr", margin=0, allow_missing_keys=True),
            transforms.SpatialPadd(keys=modalities, spatial_size=spatial_size, allow_missing_keys=True),
            transforms.CenterSpatialCropd(keys=modalities, roi_size=spatial_size, allow_missing_keys=True),
            transforms.ScaleIntensityRangePercentilesd(keys=modalities, lower=0.5, upper=99.5, b_min=-1, b_max=1, allow_missing_keys=True),
            train_transforms if phase == "train" else transforms.Compose([])
        ]
    )


def get_synthrad_dataset(data_path, phase="train", spatial_size=(144, 192, 144)):
    transform = get_transforms(phase=phase, spatial_size=spatial_size)

    datalist = sorted(os.listdir(data_path))

    data = []
    for subject in datalist:
        sub_path = os.path.join(data_path, subject)

        if os.path.exists(sub_path) is False:
            continue

        mr = os.path.join(sub_path, "mr.nii.gz")
        ct = os.path.join(sub_path, "ct.nii.gz")

        data.append({"mr": mr, "ct": ct, "subject_id": subject, "path": mr})

    print(phase, " num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform)


class SynthRadBase(Dataset):
    def __init__(self, source=None, target=None, **kwargs):
        super().__init__()
        self.data = None
        self.modalities = ["mr", "ct"]
        self.source = source
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = dict(self.data[i])

        if self.source is None:
            source, target = np.random.choice(self.modalities, size=2, replace=False)
        else:
            source, target = self.source, self.target

        item["source"] = item[source]

        item["target"] = item[target]
        item["target_class"] = torch.tensor(self.modalities.index(target))

        return item


class SynthRadTrain(SynthRadBase):
    def __init__(self, data_path, phase="train", source=None, target=None, spatial_size=(144, 192, 144)):
        super().__init__(source=source, target=target)
        self.data = get_synthrad_dataset(data_path, phase, spatial_size)


class SynthRadVal(SynthRadBase):
    def __init__(self, data_path, phase="val", source=None, target=None, spatial_size=(144, 192, 144)):
        super().__init__(source=source, target=target)
        self.data = get_synthrad_dataset(data_path, phase, spatial_size)


class SynthRadTest(SynthRadBase):
    def __init__(self, data_path, phase="test", source=None, target=None, spatial_size=(144, 192, 144)):
        super().__init__(source=source, target=target)
        self.data = get_synthrad_dataset(data_path, phase, spatial_size)
