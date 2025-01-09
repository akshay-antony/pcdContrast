import numpy as np
import os
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import requests
import torch
from tqdm import tqdm

def generate_embeddings(image,
                        processor,
                        model,
                        device,):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states[0, 0, :].detach().cpu().numpy()


def main(): 
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    model = AutoModel.from_pretrained("facebook/dinov2-base")
    model.to(device)
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    
    images_folder = "../data/ShapeNetCore.v2_2d/"
    for class_name in os.listdir(images_folder):
        class_folder = os.path.join(images_folder, class_name)
        loop = tqdm(os.listdir(class_folder), desc=f"Processing {class_name}", total=len(os.listdir(class_folder)), colour="green")
        for instance_folder in loop:
            instance_folder = os.path.join(class_folder, instance_folder)
            all_embeddings = []
            # check if any npy file exists
            if os.path.exists(os.path.join(instance_folder, "mean_embeddings.npy")):
                print(f"Skipping {instance_folder}")
                continue    
            for image_name in os.listdir(instance_folder):
                if not image_name.endswith(".png"):
                    continue
                image_path = os.path.join(instance_folder, image_name)
                image = Image.open(image_path).convert("RGB")
                embeddings = generate_embeddings(image, processor, model, device)
                all_embeddings.append(embeddings)
            all_embeddings = np.array(all_embeddings)
            print(all_embeddings.shape)
            mean_embeddings = np.mean(all_embeddings, axis=0)
            output_file = os.path.join(instance_folder, "mean_embeddings.npy")
            # print(f"Saving embeddings to: {output_file}")
            np.save(output_file, mean_embeddings)
            tqdm.write(f"Saved embeddings to: {output_file}")
            # return

if __name__ == "__main__":
    main()