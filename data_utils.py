from detectron2.structures import BoxMode
import xml.etree.cElementTree as ET
import os, json, cv2
import numpy as np

def get_balloon_dicts(img_dir):
    # Converting balloon_dataset to detectron2's standard format.
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def dataset_to_dict(dataset_path):
    dataset_dicts = []
    for idx, filename in enumerate(os.listdir(dataset_path)):
        if filename.endswith(".xml"): #label files/xml's
            record = {}

            filename_img = dataset_path + "/" + filename[:-4] + ".png"
            # import ipdb; ipdb.set_trace()
            height, width = cv2.imread(filename_img).shape[:2]
            
            record["file_name"] = filename_img
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
        
            objs = []
            tree = ET.parse(dataset_path + "/" + filename)
            root = tree.getroot() 
            root_tag = root.tag
            for form in root.findall("object"):
                obj = {
                    "bbox": [int(form[4][0].text), int(form[4][1].text), int(form[4][2].text), int(form[4][3].text)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [], # [poly]
                    "category_id": 0,
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts

def xml_to_dict(dataset_path):
    dataset_dicts = []
    for idx, filename in enumerate(os.listdir(dataset_path)):
        if filename.endswith(".xml"): #label files/xml's
            record = {}

            filename_img = dataset_path + "/" + filename[:-4] + ".png"
            # import ipdb; ipdb.set_trace()
            height, width = cv2.imread(filename_img).shape[:2]
            
            record["file_name"] = filename_img
            record["image_id"] = idx
            record["height"] = height
            record["width"] = width
        
            objs = []
            tree = ET.parse(dataset_path + "/" + filename)
            root = tree.getroot() 
            root_tag = root.tag
            for form in root.findall("object"):
                obj = {
                    "bbox": [int(form[4][0].text), int(form[4][1].text), int(form[4][2].text), int(form[4][3].text)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [], # [poly]
                    "category_id": 0,
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
    return dataset_dicts

# xml_to_dict("/home/saumyas/Projects/IAM-Vision/Vision_based_pose_estimation/Vision_based_pose_estimation/datasets/mug/")