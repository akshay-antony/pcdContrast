import numpy as np
import cv2
import os

def png_to_jpg(png_path, jpg_path):
    png = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    cv2.imwrite(jpg_path, png, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    # print(f"size of {png_path}: {os.path.getsize(png_path)}")
    # print(f"size of {jpg_path}: {os.path.getsize(jpg_path)}")
    # return
    # delete the png file
    os.remove(png_path)

# Example
png_path = "/home/xsamplecontrast/xsampleContrastive/data/ShapeNetCore.v2_2d/02773838/2ca6df7a5377825cfee773c7de26c274/roll_0_pitch_-45_yaw_0.png"
jpg_path = "/home/xsamplecontrast/xsampleContrastive/data/ShapeNetCore.v2_2d/02773838/2ca6df7a5377825cfee773c7de26c274/roll_0_pitch_-45_yaw_0.jpg"
# png_to_jpg(png_path, jpg_path)\\\

top_folder = "/home/xsamplecontrast/xsampleContrastive/data/ShapeNetCore.v2_2d"
#use regex to find all png files
import re
import os
import glob

filenames = []
for root, dirs, files in os.walk(top_folder):
    for file in files:
        if file.endswith(".png"):
            filenames.append(os.path.join(root, file))

print(f"Number of png files: {len(filenames)}")
from tqdm import tqdm
for filename in tqdm(filenames):
    jpg_filename = filename.replace(".png", ".jpg")
    png_to_jpg(filename, jpg_filename)
    # print(f"Converted {filename} to {jpg_filename}")
    # break
print("Done")