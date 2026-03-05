# Copyright 2023 cansik.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import time
from collections import defaultdict
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from imagesize import imagesize
from pycocotools.coco import COCO

from .coco import CocoDataset


def get_yolo_file_list(img_paths, ann_paths, type=".txt"):
    """
    Get file list from paired image and annotation directories.
    img_paths and ann_paths must be one-to-one correspondence.
    
    Args:
        img_paths: str or list of str - image directory path(s)
        ann_paths: str or list of str - annotation directory path(s)
        type: file extension to filter (default: ".txt")
    
    Returns:
        list of tuples: (img_base_path, ann_base_path, txt_rel_path, img_rel_path)
        - img_base_path: the image folder root
        - ann_base_path: the annotation folder root (corresponding to img_base_path)
        - txt_rel_path: relative path of txt from ann_base_path
        - img_rel_path: relative path of image from img_base_path (derived from txt filename)
    """
    file_tuples = []
    
    # Normalize to lists
    if isinstance(img_paths, str):
        img_paths = [img_paths]
    if isinstance(ann_paths, str):
        ann_paths = [ann_paths]
    
    if len(img_paths) != len(ann_paths):
        raise ValueError(
            f"img_paths and ann_paths must have the same length! "
            f"Got img_paths: {len(img_paths)}, ann_paths: {len(ann_paths)}. "
            "Please ensure they are one-to-one correspondence in YAML config."
        )
    
    for img_base, ann_base in zip(img_paths, ann_paths):
        for maindir, subdir, file_name_list in os.walk(ann_base):
            for filename in file_name_list:
                apath = os.path.join(maindir, filename)
                ext = os.path.splitext(apath)[1]
                if ext == type:
                    # TXT relative path from its annotation root
                    txt_rel_path = os.path.relpath(apath, ann_base)
                    # Image relative path (same relative structure as TXT)
                    img_name_no_ext = os.path.splitext(txt_rel_path)[0]
                    # Try common image extensions
                    for img_ext in [".jpg", ".png", ".jpeg", ".bmp", ".tiff"]:
                        candidate = os.path.join(img_base, img_name_no_ext + img_ext)
                        if os.path.exists(candidate):
                            img_rel_path = img_name_no_ext + img_ext
                            file_tuples.append((img_base, ann_base, txt_rel_path, img_rel_path))
                            break
                    else:
                        logging.warning(f"Could not find image for {apath}")
    
    return file_tuples


class CocoYolo(COCO):
    def __init__(self, annotation):
        """
        Constructor of Microsoft COCO helper class for
        reading and visualizing annotations.
        :param annotation: annotation dict
        :return:
        """
        # load dataset
        super().__init__()
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        dataset = annotation
        assert type(dataset) == dict, "annotation file format {} not supported".format(
            type(dataset)
        )
        self.dataset = dataset
        self.createIndex()


class YoloDataset(CocoDataset):
    def __init__(self, class_names, **kwargs):
        self.class_names = class_names
        super(YoloDataset, self).__init__(**kwargs)

    def yolo_to_coco(self, ann_path):
        """
        convert yolo annotations to coco_api
        :param ann_path: tuple of (img_paths, ann_paths) - paired image and annotation paths
        :return:
        """
        logging.info("loading annotations into memory...")
        tic = time.time()
        
        # ann_path is now (img_paths, ann_paths) tuple from BaseDataset
        img_paths, ann_paths = ann_path
        ann_file_tuples = get_yolo_file_list(img_paths, ann_paths, type=".txt")
        logging.info("Found {} annotation files.".format(len(ann_file_tuples)))
        image_info = []
        categories = []
        annotations = []
        for idx, supercat in enumerate(self.class_names):
            categories.append(
                {"supercategory": supercat, "id": idx + 1, "name": supercat}
            )
        ann_id = 1

        for idx, (img_base, ann_base, txt_rel_path, img_rel_path) in enumerate(ann_file_tuples):
            ann_file = os.path.join(ann_base, txt_rel_path)
            image_file = os.path.join(img_base, img_rel_path)

            with open(ann_file, "r") as f:
                lines = f.readlines()

            width, height = imagesize.get(image_file)

            # Store the relative path from corresponding image folder
            # Also store the img_base_path for correct image loading
            info = {
                "file_name": img_rel_path,
                "height": height,
                "width": width,
                "id": idx + 1,
                "img_base_path": img_base,  # Custom field for multi-folder support
            }
            image_info.append(info)
            for line in lines:
                data = [float(t) for t in line.split(" ")]
                cat_id = int(data[0])
                locations = np.array(data[1:]).reshape((len(data) // 2, 2))
                bbox = locations[0:2]

                bbox[0] -= bbox[1] * 0.5

                bbox = np.round(bbox * np.array([width, height])).astype(int)
                x, y = bbox[0][0], bbox[0][1]
                w, h = bbox[1][0], bbox[1][1]

                if cat_id >= len(self.class_names):
                    logging.warning(
                        f"Category {cat_id} is not defined in config ({txt_rel_path})"
                    )
                    continue

                if w < 0 or h < 0:
                    logging.warning(
                        "WARNING! Find error data in file {}! Box w and "
                        "h should > 0. Pass this box annotation.".format(txt_rel_path)
                    )
                    continue

                coco_box = [max(x, 0), max(y, 0), min(w, width), min(h, height)]
                ann = {
                    "image_id": idx + 1,
                    "bbox": coco_box,
                    "category_id": cat_id + 1,
                    "iscrowd": 0,
                    "id": ann_id,
                    "area": coco_box[2] * coco_box[3],
                }
                annotations.append(ann)
                ann_id += 1

        coco_dict = {
            "images": image_info,
            "categories": categories,
            "annotations": annotations,
        }
        logging.info(
            "Load {} txt files and {} boxes".format(len(image_info), len(annotations))
        )
        logging.info("Done (t={:0.2f}s)".format(time.time() - tic))
        
        # Log image folder statistics
        img_folder_stats = {}
        for img in image_info:
            img_base = img.get("img_base_path", "N/A")
            img_folder_stats[img_base] = img_folder_stats.get(img_base, 0) + 1
        
        logging.info("=" * 60)
        logging.info("Image Folder Statistics:")
        for folder, count in sorted(img_folder_stats.items()):
            logging.info(f"  {folder}: {count} images")
        logging.info(f"  TOTAL: {len(image_info)} images")
        logging.info("=" * 60)
        
        return coco_dict

    def get_data_info(self, ann_path):
        """
        Load basic information of dataset such as image path, label and so on.
        :param ann_path: tuple of (img_paths, ann_paths) - passed from BaseDataset
        :return: image info:
        [{'file_name': '000000000139.jpg',
          'height': 426,
          'width': 640,
          'id': 139},
         ...
        ]
        """
        coco_dict = self.yolo_to_coco(ann_path)
        self.coco_api = CocoYolo(coco_dict)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)
        return img_info
