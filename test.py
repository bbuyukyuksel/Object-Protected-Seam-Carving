import numpy as np
from termcolor import cprint, colored
import os
import cv2

# dict_keys(['grey', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'])

# Create Test Image
x = np.random.randint(0,255, size=(64,64), dtype=np.uint8)
'''
x = np.array(
[[ 72,  43, 239,  46,  51,  46, 232, 118, 113, 108, 121, 228,  16, 191, 249,], 
 [157, 108, 120,  24,  99, 214,  17, 199, 147,  10, 156, 229, 166, 241, 195,], 
 [131,  91, 103, 127,  28, 247, 126, 249,  57, 194, 175, 235, 162, 225,  62,], 
 [254, 237, 214, 148, 202, 153, 238, 139, 100, 185, 153,  36, 161, 145, 201,], 
 [ 52, 163,  27,   8, 121, 187, 100,  94, 204,  85,  93,  98,  69, 206,  86,], 
 [191, 173,  16, 235,  51, 193, 242, 193, 222, 104,  71, 100, 125, 149, 230,], 
 [136, 193, 199, 102, 203, 243,  68, 232,  56, 227, 198, 207, 134, 246,  59,],])
'''
print(">> Test Image")
print(x)
r, c = x.shape
print(">> Shape")
print(r, c)


energy_map = x

M = energy_map.copy()
backtrack = np.zeros_like(M, dtype=np.int)

cprint("Debugging Start", 'blue')
for i in range(1, r):
    for j in range(0, c):
        os.system('cls')
        print(M)
        print(f"R:{colored(i, 'blue')} C:{colored(j, 'green')}")
        # Handle the left edge of the image, to ensure we don't index -1
        if j == 0:
            print("1->        ", M[i - 1, j:j + 2])
            idx = np.argmin(M[i - 1, j:j + 2])
            print("2-> idx    ", idx)
            backtrack[i, j] = idx + j
            print("3-> idx+j  ", idx + j)
            print("4->\n", backtrack)
            min_energy = M[i - 1, idx + j]
            print("5->        ", min_energy)
        else:
            print("1->        ", M[i - 1, j - 1:j + 2])
            idx = np.argmin(M[i - 1, j - 1:j + 2])
            print("2-> idx    ", idx)
            backtrack[i, j] = idx + j - 1
            print("3-> idx+j-1", idx + j - 1)
            print("4->\n", backtrack)
            min_energy = M[i - 1, idx + j - 1]
            print("5->        ", min_energy)


        M[i, j] += min_energy
        #input('')
        print("#*20")



#return M, backtrack

cv2.imshow('test', energy_map.astype(np.uint8))
cv2.waitKey(0)
