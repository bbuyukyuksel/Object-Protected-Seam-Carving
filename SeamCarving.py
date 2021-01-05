import sys
import os

import numpy as np
from scipy.ndimage.filters import convolve
import cv2
from scutils import SeamCarvingUtils
from tqdm import trange

class SeamCarving:
    __counter__ = 0
    def __init__(self, bboxs=None):
        self.bboxs = bboxs
        self.energy_offset = 9999999
        self.delay = 10
        self.save_each_iter = 20

    def calc_energy(self, img):
        filter_du = np.array([
            [1.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
            [-1.0, -2.0, -1.0],
        ])
        # This converts it from a 2D filter to a 3D filter, replicating the same
        # filter for each channel: R, G, B
        filter_du = np.stack([filter_du] * 3, axis=2)

        filter_dv = np.array([
            [1.0, 0.0, -1.0],
            [2.0, 0.0, -2.0],
            [1.0, 0.0, -1.0],
        ])
        # This converts it from a 2D filter to a 3D filter, replicating the same
        # filter for each channel: R, G, B
        filter_dv = np.stack([filter_dv] * 3, axis=2)

        img = img.astype('float32')
        convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))

        # We sum the energies in the red, green, and blue channels
        energy_map = convolved.sum(axis=2)

        return energy_map

    def minimum_seam(self, img, imgbboxsmask=np.array([]), crop_type=None):
        r, c, _ = img.shape
        energy_map = self.calc_energy(img)

        if imgbboxsmask.any():
            indices = np.where(imgbboxsmask > 0)
            energy_map[indices[0], indices[1]] += self.energy_offset
        
        if crop_type == 'c':
            cv2.imshow("Energy Map", energy_map.astype(np.uint16)*8)
        elif crop_type == 'r':
            cv2.imshow("Energy Map", SeamCarvingUtils.Rot90Backward(energy_map.astype(np.uint16)*8))

        M = energy_map.copy()
        backtrack = np.zeros_like(M, dtype=np.int)

        for i in range(1, r):
            last_idx = None
            for j in range(0, c):
                # Handle the left edge of the image, to ensure we don't index -1
                if j == 0:
                    idx = np.argmin(M[i - 1, j:j + 2])
                    backtrack[i, j] = idx + j
                    min_energy = M[i - 1, idx + j]
                    last_idx = backtrack[i, j]
                else:
                    idx = np.argmin(M[i - 1, j - 1:j + 2])
                    backtrack[i, j] = idx + j - 1
                    min_energy = M[i - 1, idx + j - 1]
                    last_idx = backtrack[i, j]
                M[i, j] += min_energy
        return M, backtrack

    def crop_c(self, img, scale_c):
        r, c, _ = img.shape
        new_c = int(scale_c * c)

        for i in trange(c - new_c):
            img = self.carve_column(img, crop_type='c')

        return img

    
    def crop_r(self, img, scale_r):
        img = SeamCarvingUtils.Rot90Forward(img)
        r, c, _ = img.shape
        new_c = int(scale_r * c)

        for i in trange(c - new_c):
            img = self.carve_column(img, crop_type='r')

        
        img = SeamCarvingUtils.Rot90Backward(img)
        return img
    

    def carve_column(self, img, crop_type):
        img_bboxs_mask = np.array([])
        colors = None
        
        if self.bboxs and crop_type=='c':
            img_bboxs_mask, colors = SeamCarvingUtils.GenerateBBOXMask(img, self.bboxs)

        elif self.bboxs and crop_type=='r':
            img_bboxs_mask, colors = SeamCarvingUtils.GenerateBBOXMask(
                                                    SeamCarvingUtils.Rot90Backward(img), 
                                                    self.bboxs)
            img_bboxs_mask = SeamCarvingUtils.Rot90Forward(img_bboxs_mask)
        
        
        r, c, _ = img.shape
        M, backtrack = self.minimum_seam(img, imgbboxsmask=img_bboxs_mask, crop_type=crop_type)

        # Create a (r, c) matrix filled with the value True
        # We'll be removing all pixels from the image which
        # have False later
        mask = np.ones((r, c), dtype=np.bool)

        # Find the position of the smallest element in the
        # last row of M
        j = np.argmin(M[-1])

        for i in reversed(range(r)):
            # Mark the pixels for deletion
            mask[i, j] = False
            j = backtrack[i, j]


        # Since the image has 3 channels, we convert our
        # mask to 3D

        # Draw Seams
        indices = np.where(mask==False)
        colored = img.copy()
        colored[indices[0], indices[1], :] = [0, 0, 255]

        if self.bboxs:
            if crop_type == 'c':
                colored = SeamCarvingUtils.DrawBBOXs(colored, bboxs=self.bboxs, colors=colors)
            elif crop_type == 'r':
                colored = SeamCarvingUtils.DrawBBOXs(SeamCarvingUtils.Rot90Backward(colored), bboxs=self.bboxs, colors=colors)
                colored = SeamCarvingUtils.Rot90Forward(colored)
                colored = np.array(colored)

            # UpdateBBOX
            for index in range(len(self.bboxs)):
                bx, by, bw, bh = self.bboxs[index]
                xend = img.shape[1] if bx+bw+1 >= img.shape[1] else bx+bw+1
                yend = img.shape[0] if by+bh+1 >= img.shape[0] else by+bh+1  
                    
                if crop_type == 'c':
                    cv2.imshow("Img-BBOXs-Mask", img_bboxs_mask)
                    try:
                        _indices = np.where(mask[by:yend ,:] == False)
                        _mean = int(np.mean(_indices[1]))
                        if _mean <= bx:
                            #print("\nDeleted Left Seam")
                            #print("Update BBOX")
                            # To Left
                            self.bboxs[index][0] -= 1
                    except:
                        pass

                elif crop_type == 'r':                    
                    cv2.imshow("Img-BBOXs-Mask", SeamCarvingUtils.Rot90Backward(img_bboxs_mask))
                    try:
                        _indices = np.where(SeamCarvingUtils.Rot90Backward(mask.copy())[: , bx:xend] == False)
                        _mean = int(np.mean(_indices[0]))
                        if _mean <= by:
                            #print("\nDeleted Left Seam")
                            #print("Update BBOX")
                            # To Up
                            self.bboxs[index][1] -= 1
                    except:
                        pass            
                # Fill
                #indices = np.where(img_bboxs_mask > 0)
                #colored[indices[0], indices[1], :] = (255,255,255)

        if crop_type == 'r':
            cv2.imshow("Info", SeamCarvingUtils.Rot90Backward(colored))
            cv2.waitKey(self.delay)
                
        elif crop_type == 'c': 
            cv2.imshow("Info", colored)
            cv2.waitKey(self.delay)
        
        mask = np.stack([mask] * 3, axis=2)

        # Delete all the pixels marked False in the mask,
        # and resize it to the new image dimensions
        
        img = img[mask].reshape((r, c - 1, 3))
        
        #if self.__counter__ % self.save_each_iter:
        #    os.makedirs("Results/Last", exist_ok=True)
        #    if crop_type == 'c':
        #        cv2.imwrite(f"Results/Last/Iter-{str(self.__counter__).rjust(4, '0')}.png", img)            
        #    if crop_type == 'r':
        #        cv2.imwrite(f"Results/Last/Iter-{str(self.__counter__).rjust(4, '0')}.png", SeamCarvingUtils.Rot90Backward(img))
        #self.__counter__ += 1
        return img
