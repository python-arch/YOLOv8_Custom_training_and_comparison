import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET
import time

def read_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bounding_boxes = []
    for obj in root.findall('.//object'):
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        bounding_boxes.append((xmin, ymin, xmax - xmin, ymax - ymin))
    return bounding_boxes

def visualize_images_in_directory(image_dir, xml_dir):
    for image_file in os.listdir(image_dir):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, image_file)
            xml_file = os.path.splitext(image_file)[0] + '.xml'
            xml_path = os.path.join(xml_dir, xml_file)

            # Read the image
            image = cv2.imread(image_path)
            
            # Convert BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Read bounding boxes from XML
            bounding_boxes = read_xml(xml_path)
            
            # Create figure and axes
            fig, ax = plt.subplots(1)
            
            # Display the image
            ax.imshow(image_rgb)
            
            # Add bounding boxes to the image
            for bbox in bounding_boxes:
                x, y, w, h = bbox
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            
            # Show the image with bounding boxes
            plt.title(f"Image: {image_file}")
            plt.show()    

# Example usage:
target_directory = '/Users/python/Desktop/Abdelrahman_ElSayed(python)_Task_7/augmented_dataset_voc'

visualize_images_in_directory(target_directory, target_directory)
