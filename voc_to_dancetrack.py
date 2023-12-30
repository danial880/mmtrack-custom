import os
import xml.etree.ElementTree as ET
import cv2
from tqdm import tqdm


def pascal_to_dancetrack(xml_path, output_path, categories):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.find('filename').text
    frm = filename.split("_")[-1]
    frame_id = int(frm[:-4]) + 1
    categ_id = 0
    
    with open(output_path, 'a') as output_file:
        for obj in root.iter('object'):
            obj_name = obj.find('name').text
            if obj_name in categories:
                bbox = obj.find('bndbox')
                left = bbox.find('xmin').text
                top = bbox.find('ymin').text
                width = str(int(bbox.find('xmax').text) - int(left))
                height = str(int(bbox.find('ymax').text) - int(top))
                output_line = f"{frame_id}, {categ_id}, {left}, {top}, {width}, {height}, 1, 1, 1\n"
                output_file.write(output_line)

def convert_dataset(dataset_path, output_folder, categories):
    num=1    
    video_folders = sorted(os.listdir(dataset_folder))
    
    for folder in tqdm(video_folders, desc="Creating Dancetrack"):
        gt_path = f"dancetrack{num:04d}/gt"
        img_path = f"dancetrack{num:04d}/img1"
        folder_path = os.path.join(dataset_path, folder)
        os.makedirs(os.path.join(output_folder, gt_path), exist_ok=True)
        os.makedirs(os.path.join(output_folder, img_path), exist_ok=True)
        files = sorted([filename for filename in os.listdir(folder_path) if filename.endswith(".xml")],
                       key=lambda x: int(x.split("_")[-1].split(".")[0]))
        img_files = sorted([filename for filename in os.listdir(folder_path) if filename.endswith(".png")],
                       key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if img_files:
            bum = 1
            for img in img_files:
                image = cv2.imread(os.path.join(folder_path, img))
                image_name = f"{bum:08d}" + ".jpg"              
                cv2.imwrite(os.path.join(output_folder, img_path, image_name), image)
                bum+=1
        if files:            
            for fyle in files:
                xml_path = os.path.join(folder_path, fyle)
                output_file = "gt.txt"
                output_path = os.path.join(output_folder, gt_path, output_file)
                pascal_to_dancetrack(xml_path, output_path, categories)
            num+=1

# Specify your dataset folder and output folder
dataset_folder = 'orig1/val'
output_folder = 'dancetrack/val'
categories = ["can"]
# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Convert the dataset
convert_dataset(dataset_folder, output_folder, categories)





