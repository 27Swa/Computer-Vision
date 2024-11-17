from numpy.ma.extras import median

import Utilities as utl
import cv2
import numpy as np
import matplotlib.pyplot as plt


def NonMaximalSuppression(img, radius):
    """
    consider only the max value
    within window of size(radius x radius)
    around each pixel and assume all other value with 0
    """

    """
        steps:
            - iterate over the image after padding it
            - get the maximum point in each matrix
            - get its index and calculate the index in the image
            - pass it to the result image
    """

    pad_size = radius//2
    padded_img = np.zeros((img.shape[0] + radius - 1, img.shape[1] + radius - 1))
    dim = padded_img.shape
    padded_img[pad_size:dim[0] - pad_size, pad_size:dim[1] - pad_size] = img

    updated_img = np.zeros((img.shape[0],img.shape[1]))
    for row in range(pad_size,dim[0] -  pad_size,radius - 1):
        for col in range(pad_size,dim[1] - pad_size, radius - 1 ):
            # Calculating the start and end of the window size in rows and col
            st_row = row - pad_size
            end_row = row + pad_size + 1
            st_col = col - pad_size
            end_col = col + pad_size + 1
            # getting the data in the image
            sub_image = padded_img[st_row:end_row,st_col:end_col]
            # getting the maximum value and its index
            max_val = np.max(sub_image)
            indx_as_1d = np.argmax(sub_image)
            indx =  np.unravel_index(indx_as_1d, sub_image.shape)
            # handling the indices in case it gets indices out of the range
            ind_x = min(indx[0] + st_row ,updated_img.shape[0] - 1)
            ind_y = min(indx[1] + st_col ,updated_img.shape[1] - 1)
            indx = (ind_x,ind_y)

            updated_img[indx[0],indx[1]] = max_val


    return updated_img

"""
1- gradients in both the X and Y directions.
2- smooth the derivative a little using gaussian 
> try on TransA, SimA
> save output as  lab4-1-a-1.png, lab4-1-a-1.png
3- Calculate R:
	3.1 Loop on each pixel:
	3.2 Calculate M for each pixel:
		3.2.1 calculate a11=Gx^2, a12=GxGy, a21=GxGy, a22=Gy^2 
	3.3 Calculate Det_M = np.linalg.det(a) or Det_M = a11*a22 - a12*a21; and trace=a11+a22
	3.4 Calculate Response at this pixel = det-k*trace^2
	3.5 Display the result, but make sure to re-scale the data in the range 0 to 255 
4- Threshold and Non-Maximal Suppression 

"""
# 1- gradients in both the X and Y directions.
def harris(img, thresh=200, radius=2, verbose=True):
    Gx, Gy = utl.get_gradients_xy(img, 5)
    if verbose:
        cv2.imshow("Gradients", np.hstack([Gx, Gy]))

    # 2- smooth the derivative a little using gaussian
    #Student Code ~ 2 Lines
    Gx = cv2.GaussianBlur(Gx, (5, 5), sigmaX=3,sigmaY=0)
    Gy = cv2.GaussianBlur(Gy, (5, 5), sigmaX=0,sigmaY=3)
    #End Student Code

    cv2.imshow("Blured", np.hstack([Gx, Gy]))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # 3- Calculate R:
    R = np.zeros(img.shape)
    k = 0.04

    # 	3.1 Loop on each pixel:
    for i in range(len(Gx)):
        for j in range(len(Gx[i])):
    # 	3.2 Calculate M for each pixel:
    # 		    M = [[a11, a12],
    #                [a21, a22]]
    #           with a11=Gx^2, a12=GxGy, a21=GxGy, a22=Gy^2
            #Student Code ~ 1 line of code
            M = np.array([[int(Gx[i,j])*int(Gx[i,j]), int(Gx[i,j])*int(Gy[i, j])],
                          [int(Gx[i,j])*int(Gy[i,j]), int(Gy[i,j])*int(Gy[i, j])]])
            #Student Code

    # 	3.3 Calculate Det_M = np.linalg.det(a) or Det_M = a11*a22 - a12*a21; and trace=a11+a22
            Det_M = np.linalg.det(M)

    # 	3.4 Calculate Response at this pixel = det-k*trace^2
    #   where trace of M is the sum of its diagonals
            #Student Code ~ 1 line of code
            R[i, j] = Det_M - k*(M[0,0]+M[1, 1])**2
            #End Student Code

    # 4 Display the result, but make sure to re-scale the data in the range 0 to 255

    R = utl.rescale(R, 0, 255)

    plt.imshow(R, cmap="gray")
    plt.show()
    # 5- Threshold and Non-Maximal Suppression
    # Student Code ~ 2 lines of code
    # Threshold for an optimal value, it may vary depending on the image.
    R = NonMaximalSuppression(R, radius)

    R[R > thresh] = 255
    R[R <= thresh] = 0
    # End Student Code
    plt.imshow(R, cmap="gray")
    plt.show()

    return R

img_pairs = [['check.bmp', 'check_rot.bmp']]
dir = 'input/'
i = 0

for [img1,img2] in img_pairs:
    i += 1
    img1 = cv2.imread(dir+img1, 0)
    img2 = cv2.imread(dir+img2, 0)
    r1 = harris(img1,radius= 3, thresh = 160)
    r2 = harris(img2,radius= 9,thresh = 160,verbose = False)
    plt.figure(i)
    plt.subplot(221), plt.imshow(img1, cmap='gray')
    plt.subplot(222), plt.imshow(img2, cmap='gray')
    plt.subplot(223), plt.imshow(r1, cmap='gray')
    plt.subplot(224), plt.imshow(r2, cmap='gray')
    plt.show()