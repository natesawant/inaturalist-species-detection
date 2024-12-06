import json
import os
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

from PIL import Image

import pandas as pd

from download_dataset import download_s3_folder, download_s3_object

from torchvision.io import read_image
from torchvision.datasets import VisionDataset

MARINE_CLASSES = [
    "Actinopterygii",
    "Gastropoda",
    "Malacostraca",
    "Bivalvia",
    "Anthozoa",
    "Elasmobranchii",
    "Asteroidea",
    "Polyplacophora",
    "Hexanauplia",
    "Echinoidea",
    "Scyphozoa",
    "Cephalopoda",
    "Hydrozoa",
    "Ascidiacea",
    "Holothuroidea",
    "Ophiuroidea",
]

FISH_CLASSES = [
    "Actinopterygii",
    "Elasmobranchii",
]

BUCKET_NAME = "inaturalist-marine-only-dataset"
DATA_DIR = Path("data")
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"


class iNaturalistDataset(VisionDataset):
    def __init__(
        self,
        root: Union[str, Path] = None,  # type: ignore[assignment]
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train: bool = True,
        download: bool = False,
        classes=[],
    ) -> None:
        if len(classes) == 0:
            classes = MARINE_CLASSES
        if download:
            self.download(classes, train)
        self.root = root
        self.transforms = transforms
        self.transform = transform
        self.target_transform = target_transform
        self.annotations = self.compile_annotations(classes, train)

    def compile_annotations(self, classes, train):
        compiled_annotations = DATA_DIR / ("train.csv" if train else "val.csv")
        if compiled_annotations.exists():
            annotations = pd.read_csv(compiled_annotations)
            self.classes = annotations["common_name"].unique()
            with open(DATA_DIR / f'{"train" if train else "test"}_order.json', encoding='utf-8') as f:
                self.id_to_order = json.load(f)
            
            return annotations
        
        print("recompiling annotations to", compiled_annotations)
        
        with open(DATA_DIR / ("train.json" if train else "val.json")) as f:
            data = json.load(f)

        categories = pd.DataFrame.from_dict(data["categories"])
        categories = categories[categories["class"].isin(classes)]
        categories = categories.reset_index()

        cids = categories["id"]

        categories["order"] = categories.index
        categories = categories.set_index("id")
        self.id_to_order = categories["order"].to_dict()
        
        with open(DATA_DIR / f'{"train" if train else "test"}_order.json', 'w', encoding='utf-8') as f:
            json.dump(self.id_to_order, f, ensure_ascii=False, indent=4)

        annotations = pd.DataFrame.from_dict(data["annotations"])
        annotations = annotations.set_index(keys="id")
        annotations = annotations[annotations["category_id"].isin(cids)]

        images = pd.DataFrame.from_dict(data["images"]).set_index("id")

        annotations = annotations.set_index("image_id").join(images)
        annotations = annotations[["category_id", "width", "height", "file_name"]]

        annotations = annotations.join(
            categories,
            on="category_id",
            how="left",
            lsuffix="_left",
            rsuffix="_right",
        )

        annotations = annotations.reset_index()
        
        annotations.to_csv(compiled_annotations)

        self.classes = annotations["common_name"].unique()
        
        with open(DATA_DIR / f'{"train" if train else "test"}_order.json', encoding='utf-8') as f:
                self.id_to_order = json.load(f)

        return annotations

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Any:
        img_path = os.path.join(DATA_DIR, self.annotations["file_name"][idx])
        image = Image.open(img_path)
        label = self.id_to_order[str(self.annotations["category_id"][idx].item())]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def download(self, classes, train: bool):
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)

        annotations = "train.json" if train else "val.json"
        images = "train" if train else "val"

        # Download annotations
        print("Downloading annotations")
        download_s3_object(BUCKET_NAME, annotations, DATA_DIR)

        with open(DATA_DIR / annotations) as f:
            data = json.load(f)

            df = pd.DataFrame.from_dict(data["categories"])
            df = df.set_index(keys="id")
            df = df[df["class"].isin(classes)]
            selected_folders = df["image_dir_name"].to_list()

        # Download specified folders
        print("Downloading images")
        for folder in selected_folders:
            download_s3_folder(BUCKET_NAME, f"{images}/{folder}", DATA_DIR)
