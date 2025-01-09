import os
import glob
from tqdm import tqdm
import shutil

if __name__ == "__main__":
    top_folder = "/home/xsamplecontrast/xsampleContrastive/data/ShapeNetCore.v2_3d" 
    for class_name in os.listdir(top_folder):
        class_folder = os.path.join(top_folder, class_name)
        if not os.path.isdir(class_folder):
            continue
        for instance_folder in tqdm(os.listdir(class_folder), total=len(os.listdir(class_folder)), desc=f"Class: {class_name}"):
            # check for a folder named images and if exist delete it
            if os.path.exists(os.path.join(class_folder, instance_folder, "images")):
                shutil.rmtree(os.path.join(class_folder, instance_folder, "images"))

            instance_folder = os.path.join(class_folder, instance_folder, "models")
            # print(instance_folder)
            for file in os.listdir(instance_folder):
                if not file.endswith(".obj") and not file.endswith(".mtl") and not file.endswith(".npy"):
                    if os.path.isfile(os.path.join(instance_folder, file)):
                        os.remove(os.path.join(instance_folder, file))
                    else:
                        shutil.rmtree(os.path.join(instance_folder, file))
        # exit()
                

    print("Done")