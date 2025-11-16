import os, shutil
from sklearn.model_selection import train_test_split

base_dir = "/kaggle/working/BreaKHis_prepared"
split_base = "/kaggle/working/BreaKHis_split"
os.makedirs(split_base, exist_ok=True)

splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(os.path.join(split_base, split), exist_ok=True)

# Split for each class
for cls in os.listdir(base_dir):
    cls_path = os.path.join(base_dir, cls)
    if not os.path.isdir(cls_path):
        continue
    
    images = os.listdir(cls_path)
    train, temp = train_test_split(images, test_size=0.3, random_state=42, stratify=None)
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=None)
    
    for split_name, split_list in zip(splits, [train, val, test]):
        split_cls_dir = os.path.join(split_base, split_name, cls)
        os.makedirs(split_cls_dir, exist_ok=True)
        for img in split_list:
            shutil.copy(os.path.join(cls_path, img), os.path.join(split_cls_dir, img))

print("Dataset successfully split into train / val / test.")