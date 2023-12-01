import cv2
import numpy as np

COVER_PATH = "cover.pgm"
STEGO_PATH = "stego.pgm"
SECRET_MESSAGE = "secret message."
PAYLOAD = 1
BLOCK_SIZE = 8
FILTER_NUM = 4
FILTER_SIZE = 3
FILTER_WEIGHT = np.array([1.0, 1.0, 1.0, 1.0])
AGGREGATION_PARAM = 1.0

FILTER_KERNELS = np.array([
    [[-1, 2, -1], [0, 0, 0], [1, -2, 1]], 
    [[-1, 0, 1], [2, 0, -2], [-1, 0, 1]], 
    [[0, -1, 2], [1, 0, -1], [-2, 1, 0]], 
    [[2, -1, 0], [-1, 0, 1], [0, 1, -2]]  
])

def message_to_bits(message):
    bytes = message.encode("utf-8")
    bits = []
    for byte in bytes:
        for i in range(8):
            bits.append((byte >> (7 - i)) & 1)
    return bits

def bits_to_message(bits):
    bytes = []
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte += bits[i + j] << (7 - j)
        bytes.append(byte)
    message = bytes.decode("utf-8")
    return message

def directional_residuals(image):
    residuals = np.zeros((FILTER_NUM, image.shape[0], image.shape[1]))
    for k in range(FILTER_NUM):
        residuals[k] = cv2.filter2D(image, -1, FILTER_KERNELS[k])
    return residuals

def distortion_function(image):
    residuals = directional_residuals(image)
    distortion = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            weighted_residuals = np.abs(residuals[:, i, j]) * FILTER_WEIGHT
            distortion[i, j] = np.sum(weighted_residuals ** AGGREGATION_PARAM) ** (1 / AGGREGATION_PARAM)
    return distortion

def embed(cover, message):
    bits = message_to_bits(message)
    distortion = distortion_function(cover)
    stego = cover.copy()
    bit_count = 0
    for i in range(cover.shape[0]):
        for j in range(cover.shape[1]):
            if bit_count < len(bits):
                pixel = cover[i, j]
                bit = bits[bit_count]
                if pixel % 2 != bit:
                    if pixel == 255:
                        pixel -= 1
                    elif pixel == 0:
                        pixel += 1
                    else:
                        if distortion[i, j] > distortion[i, j + 1]:
                            pixel += 1
                        else:
                            pixel -= 1
                stego[i, j] = pixel
                bit_count += 1
            else:
                break
    return stego

def extract(stego, length):
    bits = []
    bit_count = 0
    for i in range(stego.shape[0]):
        for j in range(stego.shape[1]):
            if bit_count < length:
                pixel = stego[i, j]
                bit = pixel % 2
                bits.append(bit)
                bit_count += 1
            else:
                break
    message = bits_to_message(bits)
    return message

cover = cv2.imread(COVER_PATH, cv2.IMREAD_GRAYSCALE)
stego = embed(cover, SECRET_MESSAGE)
cv2.imwrite(STEGO_PATH, stego)
message = extract(stego, len(SECRET_MESSAGE) * 8)
print(message)
