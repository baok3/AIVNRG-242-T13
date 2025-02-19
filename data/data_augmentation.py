import os
import random
from PIL import Image
import albumentations as A
import numpy as np

# Đọc file classes.txt để lấy tên các class
with open('classes.txt', 'r') as f:
    classes = f.read().splitlines()

# Định nghĩa các phương pháp augmentation
transform = A.Compose([
    A.RandomRotate90(),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.Transpose(),
    A.RandomBrightnessContrast(),
])

valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
split_ratio = 0.8  # 80% train, 20% test

# Duyệt qua các class
for class_name in classes:
    class_dir = os.path.join('../data/processed/rare_disease', class_name)
    if not os.path.isdir(class_dir):
        continue

    # Tạo thư mục lưu ảnh augmentation theo class nếu chưa tồn tại
    train_class_dir = os.path.join('splits', 'train', class_name)
    test_class_dir = os.path.join('splits', 'test', class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    images = [img for img in os.listdir(class_dir) if os.path.splitext(img)[1].lower() in valid_extensions]
    random.shuffle(images)
    split_idx = int(len(images) * split_ratio)
    train_images, test_images = images[:split_idx], images[split_idx:]

    # Duyệt qua tập train
    for image_name in train_images:
        image_path = os.path.join(class_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        for i in range(5):  # Tạo 5 ảnh augmentation cho mỗi ảnh gốc
            augmented = transform(image=np.array(image))['image']
            augmented_image = Image.fromarray(augmented)
            save_path = os.path.join(train_class_dir, f'{os.path.splitext(image_name)[0]}_aug_{i}.png')
            augmented_image.save(save_path)

    # Duyệt qua tập test
    for image_name in test_images:
        image_path = os.path.join(class_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        save_path = os.path.join(test_class_dir, image_name)
        image.save(save_path)  # Lưu ảnh gốc vào tập test

print("Augmentation completed and images saved in 'splits' folder with train-test split.")
