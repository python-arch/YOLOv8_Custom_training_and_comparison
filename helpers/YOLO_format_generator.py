# simple script to convert from Pascal VOC to YOLO format and split the data into train/test 
from pylabel import importer
import os 
import shutil

source_directory = "/Users/python/Desktop/Abdelrahman_ElSayed(python)_Task_7/augmented_dataset_voc"
output_directory = "/Users/python/Desktop/Abdelrahman_ElSayed(python)_Task_7/yolo_format_dataset"
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

copy_images(source_directory, output_directory)


# # train test split the data
train_dir = os.path.join(output_directory, 'train')
val_dir = os.path.join(output_directory, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# # List all files in the dataset directory
files = os.listdir(output_directory)

# # Calculate the split point for 80% training and 20% validation
split_point = int(0.8 * len(files)) // 2
count=0
for file in files:
    if file.endswith('.jpg'):
        img_src = os.path.join(output_directory, file)
        annot_src = os.path.join(output_directory, file.replace('.jpg', '.txt'))
        if count <= split_point:
            img_dst = os.path.join(train_dir, file)
            annot_dst = os.path.join(train_dir, file.replace('.jpg', '.txt'))
        else:
            img_dst = os.path.join(val_dir, file)
            annot_dst = os.path.join(val_dir, file.replace('.jpg', '.txt'))
        shutil.move(img_src, img_dst)
        shutil.move(annot_src, annot_dst)
        count += 1
        
print("Data set Splitted and saved")