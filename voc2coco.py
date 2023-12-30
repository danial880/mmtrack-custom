import os
import json
from xml.etree import ElementTree as ET
import numpy as np

def parse_pascal_voc_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    objects = []
    for obj in root.findall('object'):
        label = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        category_id = 1 if label == 'can' else 2 if label == 'guide' else None
        if category_id is not None:
            objects.append({
                "category_id": category_id,
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "area": (xmax - xmin) * (ymax - ymin),
                "iscrowd": 0
            })

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "objects": objects
    }

def pascal_to_coco(pascal_dir, output_json):
    coco_data = {
        "info": {},
        "licenses": [],
        "categories": [
            {"id": 1, "name": "can", "supercategory": "object"},
            {"id": 2, "name": "guide", "supercategory": "object"}
        ],
        "images": [],
        "annotations": []
    }

    image_id = 1
    annotation_id = 1

    for filename in os.listdir(pascal_dir):
        if filename.endswith(".xml"):
            xml_path = os.path.join(pascal_dir, filename)
            annotation_data = parse_pascal_voc_xml(xml_path)

            image_info = {
                "id": image_id,
                "width": annotation_data["width"],
                "height": annotation_data["height"],
                "file_name": annotation_data["filename"]
            }

            coco_data["images"].append(image_info)

            for obj in annotation_data["objects"]:
                annotation_info = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": obj["category_id"],
                    "bbox": obj["bbox"],
                    "area": obj["area"],
                    "iscrowd": obj["iscrowd"],
                    "segmentation": []
                }

                coco_data["annotations"].append(annotation_info)
                annotation_id += 1

            image_id += 1

    with open(output_json, "w") as json_file:
        json.dump(coco_data, json_file)

if __name__ == "__main__":
    pascal_dir = "./coco/val"  # Change this to the directory containing Pascal VOC annotations
    output_json = "./coco/val.json"  # Change this to the desired output JSON file

    pascal_to_coco(pascal_dir, output_json)

