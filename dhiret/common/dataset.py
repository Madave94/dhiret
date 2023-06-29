import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from dhiret.common.utils import load_image
from PIL import Image
import random

Image.MAX_IMAGE_PIXELS = None

class RetrievalEvaluation(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, k_l1=5, k_l2=10):
        self.data = pd.read_csv(csv_file)
        self.data["primary_instance"].str.lower()
        self.data["secondary_categories"].str.lower()
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.k_l1 = k_l1
        self.k_l2 = k_l2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["file_name"]
        img_path = self.img_dir / img_name
        image = self._load_image_with_postfix(img_path)

        primary_instance = self.data.iloc[idx]["primary_instance"]
        secondary_categories = self.data.iloc[idx]["secondary_categories"].split(";")
        assert isinstance(secondary_categories, list), "Secondary categories needs to be in a list. Error at image {}.".format(
            img_name
        )
        one_word = self.data.iloc[idx]["1-word"]
        three_word = self.data.iloc[idx]["3-word"]
        sentence = self.data.iloc[idx]["sentence"]

        sample = {
            "image_name": img_name,
            "image": image,
            "primary_instance": primary_instance,
            "secondary_categories": secondary_categories,
            "one_word": one_word,
            "three_word": three_word,
            "sentence": sentence,
        }

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample

    def _load_image_with_postfix(self, img_name):
        postfixes = ['.jpg', '.png', '.jpeg']
        for postfix in postfixes:
            img_path = img_name.with_suffix( postfix )
            if img_path.exists():
                return load_image(img_path)
        raise FileNotFoundError(f"No image found with name {img_name} and any of the postfixes: {postfixes}")

    def evaluate_retrieval(self, annoy_index, image_name_list):
        primary_instances = self.data["primary_instance"].unique()
        instance_aps = []
        category_aps = []

        for cls in primary_instances:
            instance_indices = self.data.index[self.data["primary_instance"] == cls].tolist()
            if len(instance_indices) > 1:
                instance_ap = self._primary_instance_average_precision(annoy_index, image_name_list, instance_indices)
                instance_aps.append(instance_ap)
            category_ap = self._secondary_categories_average_precision(annoy_index, image_name_list, instance_indices)
            category_aps += category_ap

        return np.mean(instance_aps), np.mean(category_aps)

    def _primary_instance_average_precision(self, annoy_index, image_name_list, instance_indices):
        num_relevant_items = len(instance_indices) - 1 if len(instance_indices) - 1 <= self.k_l1 else self.k_l1 - 1
        assert num_relevant_items > 0, "Found an image with no relevant primary instance, this should not be possible."
        total_ap = 0

        for idx in instance_indices:
            query_img_name = self.data.iloc[idx]["file_name"]
            query_img_idx = image_name_list.index(query_img_name)
            similar_img_indices = annoy_index.get_nns_by_item(query_img_idx, self.k_l1)

            retrieved_items = 0
            ap = 0

            for rank, similar_idx in enumerate(similar_img_indices):
                if similar_idx == idx:  # Skip if the similar image is the query image itself
                    continue
                if self.data.iloc[similar_idx]["primary_instance"] == self.data.iloc[idx]["primary_instance"]:
                    retrieved_items += 1
                    ap += retrieved_items / (rank + 1)

            total_ap += ap / num_relevant_items if retrieved_items > 0 else 0

        return total_ap / num_relevant_items

    def _secondary_categories_average_precision(self, annoy_index, image_name_list, instance_indices):
        category_ap = []

        for idx in instance_indices:
            query_img_name = self.data.iloc[idx]["file_name"]
            query_img_idx = image_name_list.index(query_img_name)
            similar_img_indices = annoy_index.get_nns_by_item(query_img_idx, self.k_l2)

            # num relevant items is unique for each item.
            query_secondary_category = set(self.data.iloc[idx]["secondary_categories"].split(";"))
            num_relevant_items = 0
            for _, row in self.data.iterrows():
                similar_secondary_categories = set(row["secondary_categories"].split(";"))
                if query_secondary_category.intersection(similar_secondary_categories):
                    num_relevant_items += 1
            num_relevant_items = min(self.k_l2, num_relevant_items-1)
            if not (num_relevant_items > 0):
                continue

            retrieved_items = 0
            ap = 0

            for rank, similar_idx in enumerate(similar_img_indices):
                if similar_idx == idx:  # Skip if the similar image is the query image itself
                    continue
                similar_secondary_categories = set(self.data.iloc[similar_idx]["secondary_categories"].split(";"))
                if query_secondary_category.intersection(similar_secondary_categories):
                    retrieved_items += 1
                    ap += retrieved_items / (rank + 1)

            category_ap.append( ap / num_relevant_items if retrieved_items > 0 else 0 )

        return category_ap

class VLPImageClassificationDataset(Dataset):
    def __init__(self, data_root, csv_file, transform=None, debug=False):
        self.data_root = Path(data_root)
        self.transform = transform
        self.data = pd.read_csv(self.data_root / csv_file)
        self.subset = self.data["subset"].unique()
        self.debug = debug

        self.content_type_to_label = {'map': 0, 'illustration': 1, 'technical drawing': 2, 'object': 3, 'scheme': 4,
                                      'set-up': 5, 'person': 6, 'section': 7, 'graphical recording': 8, 'curve': 9,
                                      'interior view': 10, 'musical notation': 11, 'exterior view': 12, 'floorplan': 13}
        self.media_type_to_label = {'drawing': 0, 'graphic': 1, 'photography': 2}

    def __len__(self):
        if self.debug:
            return 50
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = self.data_root / row["subset"] / row["file_name"]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        content_type = self.content_type_to_label[row["content_type"]]
        media_type = self.media_type_to_label[row["media_type"]]
        difficult = row["difficult"]

        return {"image": img,
                "content_type": content_type,
                "media_type": media_type,
                "difficult": difficult}

class VLPSiameseDataset(VLPImageClassificationDataset):
    def __init__(self, data_root, csv_file, task='content_type', transform=None, augmentation=None, debug=False):
        super().__init__(data_root, csv_file, transform, debug=debug)
        self.task = task
        self.augmentation = augmentation

    def __getitem__(self, idx):
        row1 = self.data.iloc[idx]
        img_path1 = self.data_root / row1["subset"] / row1["file_name"]
        img1 = Image.open(img_path1).convert("RGB")

        # Choose a positive or negative pair randomly
        should_get_same_class = random.choice([True, False])

        if should_get_same_class:
            same_class_data = self.data[self.data[self.task] == row1[self.task]]
            row2 = same_class_data.sample().iloc[0]
        else:
            diff_class_data = self.data[self.data[self.task] != row1[self.task]]
            row2 = diff_class_data.sample().iloc[0]

        img_path2 = self.data_root / row2["subset"] / row2["file_name"]
        img2 = Image.open(img_path2).convert("RGB")

        if self.augmentation:
            img1 = self.augmentation(img1)
            img2 = self.augmentation(img2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label1 = self.content_type_to_label[row1["content_type"]] if self.task == "content_type" else self.media_type_to_label[row1["media_type"]]
        label2 = self.content_type_to_label[row2["content_type"]] if self.task == "content_type" else self.media_type_to_label[row2["media_type"]]

        label = int(label1 == label2)

        return {
            "image1": img1,
            "image2": img2,
            "label": label,
        }

class TaskSelector:
    def __init__(self, task1_dataset, task2_dataset):
        self.datasets = {
            'task1': task1_dataset,
            'task2': task2_dataset
        }
        self.current_task = 'task1'

    def set_task(self, task_name):
        assert task_name in self.datasets, f"Invalid task name: {task_name}. Available tasks are {list(self.datasets.keys())}."
        self.current_task = task_name

    def random_task(self):
        self.current_task = random.choice(list(self.datasets.keys()))

    def __len__(self):
        return len(self.datasets[self.current_task])

    def __getitem__(self, idx):
        return self.datasets[self.current_task][idx]

if __name__ == "__main__":
    pass