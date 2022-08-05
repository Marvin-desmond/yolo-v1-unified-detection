import glob
import cv2
from xml.dom import minidom
import pandas as pd
import numpy as np
import tqdm

from typing import List, Dict

images: List[str] = [i[:-4] for i in glob.glob("./VOC_dataset/2012_images/*.jpg")]
annotations: List[str] = [i[:-4] for i in glob.glob("./VOC_dataset/2012_annotations/*.xml")]

print(f"Images: {len(images)} Annotations: {len(annotations)}")

detections: Dict[str, list] = {
    "path": [],
    "detection_object": [],
    "xmin": [],
    "ymin": [],
    "xmax": [],
    "ymax": []
}

for annotation in tqdm.tqdm(annotations):
    path = f'{annotation.replace("annotations", "images")}.jpg'
    xml = f'{annotation}.xml'
    tree = minidom.parse(xml)
    detection_object = tree.getElementsByTagName('name')
    bboxes = tree.getElementsByTagName('bndbox')
    for char, bbox in zip(detection_object, bboxes):
        detection_object = char.firstChild.nodeValue
        xmin = int(float(bbox.getElementsByTagName('xmin')[0].firstChild.nodeValue))
        ymin = int(float(bbox.getElementsByTagName('ymin')[0].firstChild.nodeValue))
        xmax = int(float(bbox.getElementsByTagName('xmax')[0].firstChild.nodeValue))
        ymax = int(float(bbox.getElementsByTagName('ymax')[0].firstChild.nodeValue))
        detections["path"].append(path)
        detections["detection_object"].append(detection_object)
        detections["xmin"].append(xmin)
        detections["ymin"].append(ymin)
        detections["xmax"].append(xmax)
        detections["ymax"].append(ymax)

df = pd.DataFrame(detections)
df.to_parquet('detections.parquet')

