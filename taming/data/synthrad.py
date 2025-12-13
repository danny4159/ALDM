import os
import numpy as np

import torch
from torch.utils.data import Dataset

from monai import transforms
from monai.data import Dataset as MonaiDataset


def get_transforms(phase="train", spatial_size=(144, 192, 144), apply_rand_crop=True, apply_foreground_crop=True):
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
            transforms.CropForegroundd(keys=modalities, source_key="mr", margin=0, allow_missing_keys=True) if apply_foreground_crop else transforms.Compose([]),
            transforms.SpatialPadd(keys=modalities, spatial_size=spatial_size, allow_missing_keys=True) if spatial_size is not None else transforms.Compose([]),
            transforms.RandSpatialCropd(keys=modalities, roi_size=spatial_size, random_center=True, random_size=False, allow_missing_keys=True) if (apply_rand_crop and spatial_size is not None) else transforms.Compose([]),
            transforms.ScaleIntensityRangePercentilesd(keys=modalities, lower=0.5, upper=99.5, b_min=-1, b_max=1, allow_missing_keys=True),
            train_transforms if phase == "train" else transforms.Compose([])
        ]
    )


def get_synthrad_dataset(data_path, phase="train", spatial_size=(144, 192, 144), apply_rand_crop=True, apply_foreground_crop=True, subject_prefix_mode="all"):
    transform = get_transforms(phase=phase, spatial_size=spatial_size, apply_rand_crop=apply_rand_crop, apply_foreground_crop=apply_foreground_crop)

    datalist = sorted(os.listdir(data_path))

    prefix_mode = (subject_prefix_mode or "all").lower()

    def _accept(name: str) -> bool:
        if prefix_mode in ("all", "none"):
            return True

        # dataset-specific defaults for seen/unseen splits
        data_path_lower = data_path.lower()
        if "headneck" in data_path_lower:
            seen_prefixes = ["1HNA"]
            unseen_prefixes = ["1HNC", "1HND"]
        else:  # pelvis (default)
            seen_prefixes = ["1PA"]
            unseen_prefixes = ["1PC"]

        # map high-level modes to prefixes
        if prefix_mode in ("seen",):
            prefixes = seen_prefixes
        elif prefix_mode in ("unseen",):
            prefixes = unseen_prefixes
        elif prefix_mode in ("1pa", "pa"):
            prefixes = ["1PA"]
        elif prefix_mode in ("1pc", "pc"):
            prefixes = ["1PC"]
        elif prefix_mode in ("1hna", "hna"):
            prefixes = ["1HNA"]
        elif prefix_mode in ("1hnc", "hnc"):
            prefixes = ["1HNC"]
        elif prefix_mode in ("1hnd", "hnd"):
            prefixes = ["1HND"]
        else:
            prefixes = []

        if not prefixes:
            return True

        name_upper = name.upper()
        return any(name_upper.startswith(p.upper()) for p in prefixes)

    data = []
    for subject in datalist:
        if not _accept(subject):
            continue
        sub_path = os.path.join(data_path, subject)

        if os.path.exists(sub_path) is False:
            continue

        mr = os.path.join(sub_path, "mr.nii.gz")
        ct = os.path.join(sub_path, "ct.nii.gz")

        data.append({"mr": mr, "ct": ct, "subject_id": subject, "path": mr, "target_path": ct})

    print(phase, " num of subject:", len(data))

    return MonaiDataset(data=data, transform=transform) # 아하 이런식으로 경로를 data에 넣고, dataset을 만드는구나. 데이터 자체를 load하기보다 이게 가볍겠다.


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
        item["source_class"] = torch.tensor(self.modalities.index(source))

        return item


class SynthRadTrain(SynthRadBase):
    def __init__(self, data_path, phase="train", source=None, target=None, spatial_size=(144, 192, 144), apply_rand_crop=True, apply_foreground_crop=True, subject_prefix_mode="all"):
        super().__init__(source=source, target=target)
        self.data = get_synthrad_dataset(data_path, phase, spatial_size, apply_rand_crop=apply_rand_crop, apply_foreground_crop=apply_foreground_crop, subject_prefix_mode=subject_prefix_mode)


class SynthRadVal(SynthRadBase):
    def __init__(self, data_path, phase="val", source=None, target=None, spatial_size=(144, 192, 144), apply_rand_crop=True, apply_foreground_crop=True, subject_prefix_mode="all"):
        super().__init__(source=source, target=target)
        self.data = get_synthrad_dataset(data_path, phase, spatial_size, apply_rand_crop=apply_rand_crop, apply_foreground_crop=apply_foreground_crop, subject_prefix_mode=subject_prefix_mode)


class SynthRadTest(SynthRadBase):
    def __init__(self, data_path, phase="test", source=None, target=None, spatial_size=(144, 192, 144), apply_rand_crop=True, apply_foreground_crop=True, subject_prefix_mode="all"):
        super().__init__(source=source, target=target)
        self.data = get_synthrad_dataset(data_path, phase, spatial_size, apply_rand_crop=apply_rand_crop, apply_foreground_crop=apply_foreground_crop, subject_prefix_mode=subject_prefix_mode)
