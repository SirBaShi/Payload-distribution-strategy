import numpy as np
import cv2
import stc

H = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

L1 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
L2 = np.array([[1, 0, -1], [0, 0, 0], [-1, 0, 1]]) / 4

def embed(cover, message, payload):
    cover = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
    height, width = cover.shape
    message = ''.join(format(ord(c), '08b') for c in message)
    stego, _ = stc.embed(cover, message, payload)
    R = cv2.filter2D(cover, -1, H)
    R1 = cv2.filter2D(R, -1, L1)
    R2 = cv2.filter2D(R, -1, L2)
    D = np.abs(R) * (np.abs(R1) + np.abs(R2))
    stego = cover.copy()
    for i in range(height):
        for j in range(width):
            if stego[i, j] != cover[i, j]:
                if stego[i, j] == 1:
                    if D[i, j, 0] < D[i, j, 2]:
                        stego[i, j] = cover[i, j] - 1
                    else:
                        stego[i, j] = cover[i, j] + 1
                else:
                    if D[i, j, 0] > D[i, j, 2]:
                        stego[i, j] = cover[i, j] - 1
                    else:
                        stego[i, j] = cover[i, j] + 1
    return stego

def extract(stego, payload):
    stego = cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY)
    message, _ = stc.extract(stego, payload)
    message = ''.join(chr(int(message[i:i+8], 2)) for i in range(0, len(message), 8))
    return message

cover = cv2.imread('cover.png')
message = 'secret message.'
payload = 0.4
stego = embed(cover, message, payload)
cv2.imwrite('stego.png', stego)
message = extract(stego, payload)
print(message)
print(decrypted_text)
