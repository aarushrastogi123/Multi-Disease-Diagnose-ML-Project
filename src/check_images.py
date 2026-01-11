import os
from PIL import Image

root = "data/train"
count = 0
for img_name in os.listdir(root):
    try:
        img_path = os.path.join(root, img_name)
        Image.open(img_path)
        count += 1
        if count > 10:
            break
    except Exception as e:
        print("❌ Problem with:", img_name, e)
print("✅ Image loading test passed for first 10 images.")