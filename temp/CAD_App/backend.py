# backend.py

from PIL import Image, ImageEnhance

def load_image(file_path):
    image = Image.open(file_path)
    return image

def process_image(image, action):
    if action == "enhance":
        enhancer = ImageEnhance.Contrast(image)
        processed_image = enhancer.enhance(2)  # tăng độ tương phản
    elif action == "gray":
        processed_image = image.convert("L")  # chuyển sang ảnh xám
    elif action == "invert":
        processed_image = Image.eval(image, lambda x: 255 - x)  # đảo ngược màu sắc
    return processed_image
