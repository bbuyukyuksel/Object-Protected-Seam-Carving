import numpy as np
import cv2
import random

class SeamCarvingUtils:
    @classmethod
    def DrawBBOXs(self, img, bboxs, colors=None):
        t = img.copy()
        for index, bbox in enumerate(bboxs):
            bx, by, bw, bh = bbox
            xend = img.shape[1] if bx+bw+1 >= img.shape[1] else bx+bw+1
            yend = img.shape[0] if by+bh+1 >= img.shape[0] else by+bh+1   

            color = self.GenerateColor() if colors == None else colors[index] 
            cv2.rectangle(t, (bx, by), (xend, yend), color, 2)
            cv2.putText(t, f"bbox{index}", (bx,by), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return t
    @classmethod
    def GenerateColor(self):
        return [random.randint(0,255) for i in range(3)]

    @classmethod
    def GenerateBBOXMask(self, img, bboxs):
        __colors__ = []

        BBOX_Mask = np.zeros_like(img, dtype=np.uint8)
        for index, bbox in enumerate(bboxs):
            bx, by, bw, bh = bbox
            xend = img.shape[1] if bx+bw+1 >= img.shape[1] else bx+bw+1
            yend = img.shape[0] if by+bh+1 >= img.shape[0] else by+bh+1    
            __color__ = self.GenerateColor()
            BBOX_Mask[by:yend, bx:xend] = __color__
            __colors__.append(__color__)
        
        return (BBOX_Mask, __colors__)

    @classmethod
    def Rot90Forward(self, img):
        return np.array(np.rot90(img.copy(), 1, (0, 1))).copy()

    @classmethod
    def Rot90Backward(self, img):
        return np.array(np.rot90(img.copy(), 3, (0, 1))).copy()
        


if __name__ == '__main__':
    bboxs = [
        #x,y,w,h
        [208, 342, 142, 94],
        [349, 285, 43, 34],
    ]

    I = cv2.imread("Images/image.jpg")

    image_bboxs_mask, colors = SeamCarvingUtils.GenerateBBOXMask(I, bboxs=bboxs)
    image_bboxs = SeamCarvingUtils.DrawBBOXs(I, bboxs=bboxs, colors=colors)

    cv2.imshow("Image-BBOXs", image_bboxs)
    cv2.imshow("Image-BBOXs-Mask", image_bboxs_mask)
    cv2.waitKey(0)

    cv2.imwrite("Assets/image_bboxs.png", image_bboxs)
    cv2.imwrite("Assets/image_bboxs_mask.png", image_bboxs_mask)


    '''
    image_bboxs = SeamCarvingUtils.Rot90Forward(image_bboxs)
    image_bboxs_mask = SeamCarvingUtils.Rot90Forward(image_bboxs_mask)

    
    cv2.imshow("Image-BBOXs-F90", image_bboxs)
    cv2.imshow("Image-BBOXs-Mask-F90", image_bboxs_mask)
    cv2.waitKey(0)


    image_bboxs = SeamCarvingUtils.Rot90Backward(image_bboxs)
    image_bboxs_mask = SeamCarvingUtils.Rot90Backward(image_bboxs_mask)
    cv2.imshow("Image-BBOXs-B90", image_bboxs)
    cv2.imshow("Image-BBOXs-Mask-B90", image_bboxs_mask)
    cv2.waitKey(0)

    # Fill Area
    indices = np.where(image_bboxs_mask > 0)
    image_bboxs[indices[0], indices[1], :] = (255,255,255)
    cv2.imshow("Image-BBOXs", image_bboxs)
    cv2.waitKey(0)
    '''

    



