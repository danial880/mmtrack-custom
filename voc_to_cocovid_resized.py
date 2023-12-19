import os
import sys
import json
from xml.etree import ElementTree as ET
import cv2

def resize_bbox(bbox, width_ratio, height_ratio):
        xmin, ymin, xmax, ymax = bbox
        new_xmin = xmin * width_ratio
        new_ymin = ymin * height_ratio
        new_xmax = xmax * width_ratio
        new_ymax = ymax * height_ratio
        new_width = new_xmax - new_xmin
        new_height = new_ymax - new_ymin
        return [new_xmin, new_ymin, new_width, new_height]
    


def voc_to_coco(image_folder, categories, video_id, image_id, annotation_id, target_width, target_height, output_folder):
    images = []
    annotations = []

    filenames = sorted([filename for filename in os.listdir(image_folder) if filename.endswith(".png")],
                       key=lambda x: int(x.split("_")[-1].split(".")[0]))
    for filename in filenames:
        if filename.endswith(".png"):
            image_file_path = os.path.join(image_folder, filename)
            xml_file = os.path.join(image_folder, f"{os.path.splitext(filename)[0]}.xml")

            if not os.path.exists(xml_file):
                continue

            tree = ET.parse(xml_file)
            root = tree.getroot()

            frame_id = None
            try:
                frm = root.find("filename").text.split("_")[-1]
                frame_id = int(frm[:-4])
            except (ValueError, IndexError):
                print(f"Error extracting frame_id from {xml_file}")

            if frame_id is not None:
                # Read the image using OpenCV
                image = cv2.imread(image_file_path)

                # Resize the image
                resized_image = cv2.resize(image, (target_width, target_height))

                # Update the image file path to the output folder with the same structure
                relative_path = os.path.relpath(image_file_path, image_folder)
                fld_name = image_file_path.split('/')[-2]
                rel_path = os.path.join(fld_name, relative_path)
                #print(fld_name,image_file_path)
                #print(relative_path)
                
                output_image_path = os.path.join(output_folder, rel_path)
                #print(output_image_path)
                #sys.exit()

                # Create directories if they do not exist
                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

                # Save the resized image
                cv2.imwrite(output_image_path, resized_image)

                image_data = {
                    "id": image_id,
                    "file_name": output_image_path,
                    "width": target_width,
                    "height": target_height,
                    "video_id": video_id,
                    "frame_id": frame_id,
                }
                images.append(image_data)

                for obj in root.findall("object"):
                    category = obj.find("name").text
                    try:
                        category_id = categories.index(category) + 1
                    except:
                        continue
                    if category_id == 1:
                        bbox = [
                            float(obj.find("bndbox/xmin").text),
                            float(obj.find("bndbox/ymin").text),
                            float(obj.find("bndbox/xmax").text),
                            float(obj.find("bndbox/ymax").text)
                        ]

                        # Resize the bounding box coordinates
                        #print(image.shape[0], image.shape[1])
                        resized_bbox = resize_bbox(bbox, target_width / image.shape[1], target_height / image.shape[0])
                        #print(resized_bbox)

                    #if category_id == 1:
                        annotation_data = {
                            "id": annotation_id,
                            "image_id": image_id,
                            "video_id": video_id,
                            "category_id": category_id,
                            "instance_id": frame_id,
                            "bbox": resized_bbox,
                            "area": resized_bbox[2] * resized_bbox[3],
                            "occluded": False,
                            "truncated": False,
                            "iscrowd": False,
                            "ignore": False,
                            "is_vid_train_frame": True,
                            "visibility": 1.0,
                        }
                        annotations.append(annotation_data)
                        annotation_id += 1

                image_id += 1

    return {"images": images, "annotations": annotations}, image_id, annotation_id

def create_cocovid_annotations(dataset_folder, output_folder, target_width, target_height):
    categories = ["can"]

    dataset = {"categories": [{"id": i + 1, "name": cat} for i, cat in enumerate(categories)],
               "videos": [],
               "images": [],
               "annotations": []}

    video_folders = sorted(os.listdir(dataset_folder))
    image_id = 1
    annotation_id = 1
    for video_id, video_folder in enumerate(video_folders, start=1):
        video_path = os.path.join(dataset_folder, video_folder)
        if os.path.isdir(video_path):
            image_folder = video_path
            video_data, new_image_id, new_annotation_id = voc_to_coco(
                image_folder, categories, video_id, image_id, annotation_id,
                target_width, target_height, output_folder
            )
            dataset["videos"].append({"id": video_id, "name": video_folder})
            dataset["images"].extend(video_data["images"])
            dataset["annotations"].extend(video_data["annotations"])
            image_id, annotation_id = new_image_id, new_annotation_id

    return dataset

# Example usage with user-provided width, height, and output folder
dataset_folder_path = "orig/train"
output_folder_path = "resized_dataset/train"
output_dataset = create_cocovid_annotations(dataset_folder_path, output_folder_path, target_width=1000, target_height=1000)

# Optionally, you can save the dataset to a JSON file
output_json_path = "single_category_ann/cocovid_train_annotations.json"
with open(output_json_path, "w") as json_file:
    json.dump(output_dataset, json_file, indent=4)

