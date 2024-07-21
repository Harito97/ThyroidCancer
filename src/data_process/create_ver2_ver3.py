# Step 1. Read image from train/valid/test folder + Identify the bounding boxes of the thyroid nodule cluster by cell cluster detection model

# Step 2.1. Remove the noise of images (not in the bounding boxes) and save them to the corresponding folder
# This will create dataver2

# Step 2.2. Crop the bounding boxes and save them to the corresponding folder
# this will create dataver3

import os
import glob
from ultralytics import YOLO
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import random


def identify_bounding_boxes(model, image_path):
    """
    Identify bounding boxes of cell clusters in an image.

    Args:
        model (YOLO): Loaded YOLO model.
        image_path (str): Path to the input image.

    Returns:
        torch.Tensor: Tensor of bounding boxes in [x1, y1, x2, y2] format.
    """
    result = model(image_path)
    bounding_boxes = result[0].boxes.xyxy
    return bounding_boxes


def remove_noise(bounding_boxes, image_np):
    """
    Remove noise from the image by keeping only the areas within bounding boxes.

    Args:
        bounding_boxes (torch.Tensor): Tensor of bounding boxes.
        image_np (numpy.ndarray): Input image as a numpy array.

    Returns:
        tuple: (PIL.Image, dict, list) Denoised image, areas of bounding boxes, and bounding boxes as list.
    """
    areas = {}
    new_image = np.ones_like(image_np) * 255
    bounding_boxes_list = []

    for index, box in enumerate(bounding_boxes):
        x1, y1, x2, y2 = map(int, box)
        areas[index] = (x2 - x1) * (y2 - y1)
        bounding_boxes_list.append((x1, y1, x2, y2))
        # print(x1, y1, x2, y2)
        new_image[y1:y2, x1:x2] = image_np[y1:y2, x1:x2]

    if len(areas) > 0:
        return Image.fromarray(new_image), areas, bounding_boxes_list
    return Image.fromarray(image_np), areas, bounding_boxes_list


def get_5_largest_areas(areas):
    """
    Get the indices of the 5 largest areas.

    Args:
        areas (dict): Dictionary of area indices and their sizes.

    Returns:
        list: Indices of the 5 largest areas (or fewer if there are less than 5).
    """
    sorted_areas = sorted(areas.items(), key=lambda x: x[1], reverse=True)
    return [index for index, _ in sorted_areas[:5]]


def augment_images(images, num_augmentations, augmentation_transforms):
    """
    Augment the given images to reach a total of 5 images using various augmentation techniques.

    Args:
        images (list): List of PIL.Image objects.
        num_augmentations (int): Number of augmentations needed.

    Returns:
        list: List of augmented PIL.Image objects.
    """
    augmented = images.copy()

    while len(augmented) < 5:
        # Randomly select an image to augment
        base_image = random.choice(images)

        # Apply the basic augmentation transforms
        augmented_image = augmentation_transforms(base_image)

        augmented.append(augmented_image)

    return augmented[:5]  # Ensure we return exactly 5 images


def crop_5_largest_areas(image, areas, bounding_boxes, augmentation_transforms):
    """
    Crop the 5 largest areas from the image or create 5 sub-images if no bounding boxes.

    Args:
        image (PIL.Image): Input image.
        areas (dict): Dictionary of area indices and their sizes.
        bounding_boxes (list): List of bounding boxes.
        path_to_save (str): Path to save the cropped images (not used in this function).

    Returns:
        list: List of 5 PIL.Image objects (cropped or sub-images).
    """
    images = []
    if len(areas) > 0:
        for index in get_5_largest_areas(areas):
            x1, y1, x2, y2 = bounding_boxes[index]
            cropped_image = image.crop((x1, y1, x2, y2))
            images.append(cropped_image)

        # Augment the image to have 5 areas
        if len(images) < 5:
            images = augment_images(images, 5 - len(images), augmentation_transforms)
    else:
        image_size = image.size
        x1, y1, x2, y2 = 0, 0, image_size[0] // 2, image_size[1] // 2
        images = [
            image.crop((x1, y1, x2, y2)),
            image.crop((x2, y1, image_size[0], y2)),
            image.crop((x1, y2, x2, image_size[1])),
            image.crop((x2, y2, image_size[0], image_size[1])),
            image,
        ]
    return images


def main(model_path, dataver1_dir="/dataver1", dataver2_dir="/dataver2", dataver3_dir="/dataver3"):
    # model_path = "/Data/Projects/ThyroidCancerClassifier/src/model/cell_detect/runs/detect/train2/weights/best.pt"
    model = YOLO(model_path)
    print('Loaded model')
    # Define a set of augmentation transforms
    augmentation_transforms = transforms.Compose(
        [
            # transforms.Resize((256, 256)),
            # transforms.RandomCrop(224),
            transforms.RandomRotation(180),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ]
    )
    print('Loaded augmentation transforms')

    for subset_dir in ["train", "valid", "test"]:
        for label in ['.B2', 'B5', 'B6']:
            for index, image_path in enumerate(
                glob.glob(os.path.join(dataver1_dir, subset_dir, label, "*.jpg"))
            ):
                image_np = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_np)
                bounding_boxes = identify_bounding_boxes(model, image_path)
                print(f"{image_path} have {len(bounding_boxes)} cell importance clusters")

                image_dataver2, areas, bounding_boxes = remove_noise(
                    bounding_boxes, image_np
                )
                save_dataver2_dir = os.path.join(dataver2_dir, subset_dir, label, f"{index}.jpg")
                image_dataver2.save(save_dataver2_dir)
                print('Saved the image to dataver2')

                images_dataver3 = crop_5_largest_areas(
                    image_pil, areas, bounding_boxes, augmentation_transforms
                )
                for i, img in enumerate(images_dataver3):
                    # Resize if needed: img = img.resize((224, 224))
                    save_dataver3_dir = os.path.join(
                        dataver3_dir, subset_dir, label, f"{index}_{i}.jpg"
                    )
                    img.save(save_dataver3_dir)
                print('Saved 5 images cropped to dataver3')



# !mkdir -p /kaggle/working/dataver2/train/.B2 /kaggle/working/dataver2/train/B5 /kaggle/working/dataver2/train/B6
# !mkdir -p /kaggle/working/dataver2/valid/.B2 /kaggle/working/dataver2/valid/B5 /kaggle/working/dataver2/valid/B6
# !mkdir -p /kaggle/working/dataver2/test/.B2 /kaggle/working/dataver2/test/B5 /kaggle/working/dataver2/test/B6

# !mkdir -p /kaggle/working/dataver3/train/.B2 /kaggle/working/dataver3/train/B5 /kaggle/working/dataver3/train/B6
# !mkdir -p /kaggle/working/dataver3/valid/.B2 /kaggle/working/dataver3/valid/B5 /kaggle/working/dataver3/valid/B6
# !mkdir -p /kaggle/working/dataver3/test/.B2 /kaggle/working/dataver3/test/B5 /kaggle/working/dataver3/test/B6

# main(model_path='/kaggle/input/cell-cluster-detection/pytorch/train300epoches/1/best.pt', dataver1_dir='/kaggle/input/thyroidcancer-ver1/ver1', dataver2_dir='/kaggle/working/dataver2', dataver3_dir='/kaggle/working/dataver3')
