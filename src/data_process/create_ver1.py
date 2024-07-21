import os
import glob
from PIL import Image
from sklearn.model_selection import train_test_split

# Định nghĩa hàm resize và lưu ảnh
def resize_and_save(image_path, output_path, size=(1024, 768)):
    image = Image.open(image_path)
    image = image.resize(size)
    image.save(output_path)

# Định nghĩa hàm phân chia dữ liệu và đổi tên
def process_images(class_dirs, output_dir, train_ratio=0.8, valid_ratio=0.1):
    test_ratio = 1 - train_ratio - valid_ratio
    for class_dir in class_dirs:
        images = glob.glob(os.path.join(class_dir, '*.jpg'))
        # Phân chia dữ liệu
        train_val, test = train_test_split(images, test_size=test_ratio)
        train, valid = train_test_split(train_val, test_size=valid_ratio)
        
        for dataset, images in zip(['train', 'valid', 'test'], [train, valid, test]):
            for i, image_path in enumerate(images, 1):
                class_type = os.path.basename(class_dir)
                output_path = os.path.join(output_dir, dataset, class_type, f"{class_type}_{i}.jpg")
                resize_and_save(image_path, output_path)

# Hàm main thực hiện
def main(raw_dir:str='data/raw', processed_dir:str='data/processed/ver1'):
    # Lấy danh sách các thư mục nhãn
    class_dirs = [os.path.join(raw_dir, class_dir) for class_dir in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, class_dir))]

    # Tạo thư mục output nếu chưa tồn tại
    for dataset in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(processed_dir, dataset), exist_ok=True)

    # Xử lý ảnh
    process_images(class_dirs, processed_dir)