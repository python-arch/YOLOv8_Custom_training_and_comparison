from ImageAugmentor import ImageAugmenter
import os 
import shutil
from pylabel import importer
#Augment the voc dataset to generate the test data
imageaugmentor = ImageAugmenter()
imageaugmentor.run_augmentation()

# export the test dataset to yolov8 format
source_directory = "/Users/python/Desktop/Abdelrahman_ElSayed(python)_Task_7/test_dataset"
output_directory = "/Users/python/Desktop/Abdelrahman_ElSayed(python)_Task_7/test_dataset_yolo"
dataset = importer.ImportVOC(path= source_directory)
dataset.export.ExportToYoloV5(output_path=output_directory)

def copy_images(source_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over files in the source directory
    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)

        # Check if the file is a JPG file
        if filename.lower().endswith('.jpg'):
            output_path = os.path.join(output_dir, filename)

            # Copy the JPG file to the output directory
            shutil.copy2(source_path, output_path)
            print(f"Copied: {filename} to {output_dir}")
# copy the images to the test dataset along with the annotations
copy_images(source_directory , output_directory)

