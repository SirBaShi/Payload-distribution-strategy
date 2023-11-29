import os
import cv2
import numpy as np
import torch
from SRNet import SRNet 

model = SRNet()
model.load_state_dict(torch.load('srnet.pth'))
model.eval()

def load_images(dataset_path):
    images = []
    for file in os.listdir(dataset_path):
        if file.endswith('.pgm'):
            image = cv2.imread(os.path.join(dataset_path, file))
            images.append(image)
    return images

def get_probability(image):
    image = torch.from_numpy(image).float()
    image = image.permute(2, 0, 1).unsqueeze(0)
    output = model(image)
    probability = torch.sigmoid(output).item()
    return probability

def sort_images(images):
    image_prob_list = []
    for image in images:
        probability = get_probability(image)
        image_prob_list.append((image, probability))
    image_prob_list.sort(key=lambda x: x[1], reverse=True)
    return image_prob_list

def save_images(image_prob_list, save_path):
    for i, (image, probability) in enumerate(image_prob_list):
        filename = f'image_{i+1}_{probability:.4f}.jpg'
        cv2.imwrite(os.path.join(save_path, filename), image)

dataset_path = 'dataset'
save_path = 'sorted_images'
if not os.path.exists(save_path):
    os.mkdir(save_path)

images = load_images(dataset_path)
image_prob_list = sort_images(images)
save_images(image_prob_list, save_path)
