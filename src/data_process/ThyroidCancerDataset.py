from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np

""" 
Hướng dẫn sử dụng ThyroidCancerDataset:
import torchvision.transforms as transforms

# Định nghĩa các phép biến đổi
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomRotation(180),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Tạo dataset cho tập huấn luyện với cân bằng nhãn
train_dataset = ThyroidCancerDataset(data_dir='path/to/data_dir/train', transform=transform, balance=True, mode='train')

# Tạo dataset cho tập xác thực và kiểm thử mà không cần cân bằng nhãn
val_dataset = ThyroidCancerDataset(data_dir='path/to/data_dir/val', transform=transform, mode='valid')
test_dataset = ThyroidCancerDataset(data_dir='path/to/data_dir/test', transform=transform, mode='test')

# Lấy một mẫu từ dataset huấn luyện
train_image, train_label = train_dataset[0]

# Lấy một mẫu từ dataset xác thực
val_image, val_label = val_dataset[0]

# Lấy một mẫu từ dataset kiểm thử
test_image, test_label = test_dataset[0]
"""


class ThyroidCancerDataset(Dataset):
    """
    Dataset structure:
    data_dir
    |__ train
        |__ B2
        |__ B5
        |__ B6
    |__ valid
        |__ B2
        |__ B5
        |__ B6
    |__ test
        |__ B2
        |__ B5
        |__ B6

    Args:
        data_dir (str): Path to the data directory (data_dir).
        transform (callable, optional): Optional transform to be applied on a sample.
        classes (dict): A dictionary where keys are class indices and values are lists of sub-directory names.
        balance (bool): If True, balance the dataset by oversampling (only for training set).
        mode (str): One of 'train', 'valid', 'test' indicating the dataset split.

    Các chức năng có ở ThyroidCancerDataset:
    |__ Đọc dữ liệu từ folder data_dir và load vào các class:
        |__ classes={0: ['B2'], 1: ['B5'], 2: ['B6']} thì class 0 gồm các ảnh trong thư mục B2, class 1 gồm các ảnh trong thư mục B5, ...
        |__ classes={0: ['B2'], 1: ['B5', 'B6']} thì class 0 gồm các ảnh trong thư mục B2, class 1 gồm các ảnh trong thư mục B5 và B6
        |__ classes={0: ['B5'], 1: ['B6']} thì class 0 gồm các ảnh trong thư mục B5, class 1 gồm các ảnh trong thư mục B6
        |__ classes={0: ['B6'], 1: ['B2', 'B5']} thì class 0 gồm các ảnh trong thư mục B6, class 1 gồm các ảnh trong thư mục B2, B5
        |__ ...
    |__ Tăng cường dữ liệu (các phép biến đổi an toàn như: Resize về 256x256 rồi chọn tâm bất kỳ để cắt ra ảnh 224x224. Quay 180 độ và Lật ảnh)
    |__ Cân bằng các nhãn khi load ảnh vào dataset ở tập train
    """

    def __init__(
        self,
        data_dir="/drive/MyDrive/Dataset/ThyroidCancerData/processed/ver1",
        transform=None,
        classes={0: ["B2"], 1: ["B5"], 2: ["B6"]},
        balance=False,
        mode="train",
        num_x=100,
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = classes
        self.img_paths = []
        self.labels = []
        self.balance = balance
        self.mode = mode

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop(224),
                    transforms.RandomRotation(180),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                ]
            )

        # Collect all image paths and their corresponding labels
        # Eg: classes.items() = [(0, ['B2']), (1, ['B5', 'B6'])]
        for label, sub_dirs in self.classes.items():
            # Eg: label=1, sub_dirs=['B5', 'B6']
            for sub_dir in sub_dirs:
                # Eg: sub_dir='B5'
                sub_dir_path = os.path.join(data_dir, mode, sub_dir)
                if os.path.exists(sub_dir_path):
                    for img_name in os.listdir(sub_dir_path):
                        img_path = os.path.join(sub_dir_path, img_name)
                        self.img_paths.append(img_path)
                        self.labels.append(label)
                else:
                    print(f"Path {sub_dir_path} does not exist")

        # Only balance the dataset if it's the training set
        if self.balance and (self.mode == "train"):  # or self.mode == "valid"):
            print("Balancing the dataset")
            self.balance_classes(num_x)

    def balance_classes(self, num_x=100):
        # Find the maximum class size
        class_counts = np.bincount(self.labels)
        max_count = class_counts.max() * num_x

        balanced_img_paths = []
        balanced_labels = []

        # Oversample the minority classes
        for label in self.classes.keys():
            img_paths_for_label = [
                path for path, lbl in zip(self.img_paths, self.labels) if lbl == label
            ]
            oversampled_paths = np.random.choice(
                img_paths_for_label, size=max_count, replace=True
            )
            balanced_img_paths.extend(oversampled_paths)
            balanced_labels.extend([label] * max_count)

        print(f"Original dataset size: {len(self.img_paths)} | {len(self.labels)}")
        print(
            f"Balanced dataset size: {len(balanced_img_paths)} | {len(balanced_labels)}"
        )
        self.img_paths = balanced_img_paths
        self.labels = balanced_labels

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
