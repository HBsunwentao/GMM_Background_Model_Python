import numpy as np
import cv2
import os
import logging
import  sys

# global constant
GAUSSION_MODEL_NUMBER =4
LEARNING_RATE =0.005
INITIAL_SIGMA =30
INITIAL_WEIGHT =0.05
GAUSSION_THRESHHOLD=0.7
IMAGE_NUM = 287
IMAGE_TEST = 200
class gauss():
    def __init__(self):
        self.mean=np.zeros(4)
        self.sigma=np.zeros(4)
        self.weight =np.zeros(4)
        self.compo_num=0
        #self.next =None
        #self.previous =None



    

def Foreground():
    image_names, image_arr = input_images()
    num_image = len(image_names)
    GaussModel = model_init(image_arr[0])

    # cv2.namedWindow('input_image', cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow('after_process', cv2.WINDOW_AUTOSIZE)
    fps = 16
    row =len(image_arr[0])
    col =len(image_arr[0][0])
    size = (col, row)
    videowriter = cv2.VideoWriter("a.avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
    for i in range(1, num_image):
        image = image_arr[i]
        result=one_image(GaussModel, image)

        videowriter.write(result)
        logging.info('%3dth image'%i )
        swt = 12
    #     cv2.imshow('input_image', image)
    #     cv2.imshow('after_process', result)
    #     cv2.waitKey(10)
    # cv2.destroyAllWindows()




def input_images():
    filepath = 'WavingTrees/'
    image_names = []
    image_arr = []
    for filename in os.listdir(filepath):
        image_names.append(filename)
        image = cv2.imread(filepath+filename)
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image_arr.append(image_gray)
    return image_names, image_arr

def one_image(model,image):
    row = len(image)
    col = len(image[0])
    result=np.zeros([row, col])
    for i in range(0, row):
        for j in range(0, col):
            pixel = image[i][j]
            judge = PixelFitsModel(model, i, j, pixel)
            if judge:
                result[i][j] = 0
            else:
                result[i][j] = 255


    return result


def model_init(image):
    row = len(image)
    col = len(image[0])
    image_gauss_model = []
    for i in range(0, row):
        pixels_row = []
        for j in range(0, col):
            pixel = image[i][j]
            pixel_model = gauss()
            pixel_model.mean[0]=pixel
            pixel_model.sigma[0]=INITIAL_SIGMA
            pixel_model.weight[0]=INITIAL_WEIGHT
            pixel_model.compo_num=1
            pixels_row.append(pixel_model)
        image_gauss_model.append(pixels_row)
    return  image_gauss_model

def PixelFitsModel(model,row,col,pixel):
    dif =np.abs(model[row][col].mean-pixel)
    judge = dif <= 2.2*model[row][col].sigma
    # ow =np.zeros(4)
    # kk=np.where(model[row][col].weight==np.max(model[row][col].weight))
    # ow[kk]=1
    dw = LEARNING_RATE * (1-model[row][col].weight)
    wh=np.where(judge == True)
    for i in wh:
        model[row][col].weight[i] += dw[i]
        model[row][col].mean[i] =(1-LEARNING_RATE)*model[row][col].mean[i] + LEARNING_RATE*pixel
        model[row][col].sigma[i] =(1-LEARNING_RATE)*model[row][col].sigma[i]+LEARNING_RATE*dif[i]



    if len(wh)<1:
        num=model[row][col].compo_num
        pixel_model = model[row][col]
        if num == 4:
            sort_key = model[row][col].weight / model[row][col].sigma
            kk = np.where(sort_key==np.min(sort_key))
            i=kk[0]
            pixel_model.mean[i] = pixel
            pixel_model.sigma[i] = INITIAL_SIGMA
            pixel_model.weight[i] = INITIAL_WEIGHT
        else:
            pixel_model.mean[num] = pixel
            pixel_model.sigma[num] = INITIAL_SIGMA
            pixel_model.weight[num] = INITIAL_WEIGHT
            pixel_model.compo_num += 1;



    wei =np.sum(model[row][col].weight)
    model[row][col].weight = model[row][col].weight / wei

    if len(wh)>0:
        return True
    else:
        return False






def ModelUpdate1(model,row,col,pixel):
    return 0


def ModelUpdate2(model,row,col,pixel):
    return  0
if __name__ == "__main__":
    Foreground()

