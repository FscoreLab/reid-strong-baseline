import glob
import re
from pathlib import Path
import json
import os.path as osp


from .bases import BaseImageDataset


class Peta(BaseImageDataset):
    base_dir = Path("data/processed/Peta")

    def __init__(self,
                 train_file_dir: Path = base_dir / "Splits/split_train_gt2.txt",
                 query_file_dir: Path = base_dir / "Splits/split_test_query_gt2.txt",
                 gallery_file_dir: Path = base_dir / "Splits/split_test_gallery_gt2.txt",
                 mapper_path: Path = base_dir / "mapper.json",
                 root: str = "unused stuff"):
        super(Peta, self).__init__()
        self.num_train_pids = 1  # need for make_data_loader

        self.image_dir = Path("data/processed/Peta")
        self.mapper_dict = json.load(open(mapper_path, "r"))
        self.mapper_dict = {val: int(key) for key, val in self.mapper_dict.items()}

        train = self._process_file(train_file_dir, relabel=True)
        query = self._process_file(query_file_dir)
        gallery = self._process_file(gallery_file_dir)

        self.query = query
        self.gallery = gallery
        self.train = train

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _process_file(self, dir_path: Path, relabel:bool = False):
        img_paths = [self.base_dir.parent / line.strip() for line in open(dir_path, "r")]
        if relabel:
            pid_container = set()
            for img_path in img_paths:
                pid, _, _ = img_path.name.partition("_")
                pid_container.add(self.mapper_dict[pid])
            pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for i, img_path in enumerate(img_paths):
            pid, _, camid = img_path.name.partition("_")
            if relabel: pid = pid2label[self.mapper_dict[pid]]
            dataset.append((img_path, pid, i))

        return dataset