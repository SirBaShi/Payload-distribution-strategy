import cv2 
import numpy as np 
import torch 
import random 
import string
from SRNet import SRNet
from UT-GAN import UT-GAN

N = 10 
M = 100 
delta = 5 

cover_images = []
for i in range(N):
    image = cv2.imread(f"image_{i}.png") 
    carrier_images.append(image) 

secret_message = "".join(random.choices(["0", "1"], k=M)) 

srnet_model = torch.hub.load("DNNResearch/srnet", "srnet", pretrained=True)
srnet_model.eval() 

security_scores = [] 
for image in carrier_images:
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() 
    output = srnet_model(image_tensor) 
    score = output[0][0].item() 
    security_scores.append(score) 

sorted_images = [x for _, x in sorted(zip(security_scores, carrier_images), reverse=True)] 
sorted_scores = sorted(security_scores, reverse=True) 

payloads = [M // N] * N 

utgan_model = torch.hub.load("DNNResearch/utgan", "utgan", pretrained=True)
utgan_model.eval() 

def embed_message(image, message, payload):
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    prob_map = utgan_model(image_tensor) 
    prob_map = prob_map.squeeze(0).permute(1, 2, 0).numpy() 

    height, width, channels = image.shape
    prob_list = prob_map.flatten().tolist() 
    index_list = list(range(height * width * channels)) 
    sorted_index = [x for _, x in sorted(zip(prob_list, index_list), reverse=True)] # 根据修改概率对索引进行排序

    stego_image = image.copy() 
    remain_message = message 

    for index in sorted_index:
        if len(remain_message) == 0 or len(message) - len(remain_message) == payload:
            break

        i = index // (width * channels) 
        j = (index % (width * channels)) // channels 
        k = (index % (width * channels)) % channels 

        pixel = stego_image[i][j][k]
        bit = remain_message[0]

        if bit == "0":
            pixel = pixel & 254 
        else:
            pixel = pixel | 1 
        stego_image[i][j][k] = pixel

        remain_message = remain_message[1:]
    return stego_image, remain_message

stego_images = [] 
for i in range(N):
    image = sorted_images[i]
    payload = payloads[i]
    stego_image, remain_message = embed_message(image, secret_message, payload)
    stego_tensor = torch.from_numpy(stego_image).permute(2, 0, 1).unsqueeze(0).float()
    output = srnet_model(stego_tensor) 
    score = output[0][0].item() 

    if score > 0.5: 
        embed_length = payload - len(remain_message)
    else: 
        stego_image, remain_message = embed_message(image, secret_message, payload // 2)
        embed_length = payload // 2 - len(remain_message)
    stego_images.append(stego_image)
    secret_message = remain_message

