import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import combinations as comb

# reading images from the folder
def reading_data(folder_path):
    images_lst = list()
    ind = 1
    group = list()
    nam = list()
    l = len("image")
    # used to store the name of images
    images_name = list()
    for img_name in os.listdir(folder_path):
        n_path = os.path.join(folder_path,img_name)
        img = cv2.imread(n_path,0)
        img_num = int(img_name[l])
        if img_num != ind:
            ind += 1
            images_lst.append(group[:])
            images_name.append(nam[:])
            group.clear()
            nam.clear()
        group.append(img)
        nam.append(img_name.split('.')[0])
    # if both of lines weren't made then the last 2 images won't be stored
    images_lst.append(group[:])
    images_name.append(nam[:])
    return images_lst,images_name

def calculating_key_points(im):
    #store all the data in list of tuples
    kp_des = list()
    for val in im:
        kp,des = sift.detectAndCompute(val, None)
        kp_des.append((kp,des))
    return kp_des

def ratio_of_david_lowe(ratio,m):
    david = list()
    for p1,p2  in m:
        res = p1.distance / p2.distance
        if res < ratio:
            david.append(p1)
    return david

def thresholding(m):
    dis = [d.distance for d in m]
    """ 
    when plotting the distances it was skewed so it isn't better to use mean or median 
        plt.hist(dis,color= 'red',edgecolor = 'black')
        plt.show()
    """
    threshold =  np.percentile(dis,75)
    # print(threshold)
    threshold_data = list(filter(lambda z: z.distance < threshold, m))
    # print(threshold_data)
    return threshold_data

def cross_check(m):
    # calculating matches from des2 to des1
    r1 = bf.match(des2,des1)
    """ 
    an = bf.knnMatch(des2,des1,2)
    r1 = ratio_of_david_lowe(.75,an)
    r1 = thresholding(r1)
    """
    cross = []
    for m1 in m:
        for m2 in r1:
            if m2.queryIdx == m1.trainIdx and m1.queryIdx == m2.trainIdx:
                cross.append(m1)
    return cross

def ransac(match):
    # sort points based on their distance
    mat = sorted(match, key = lambda dis: dis.distance)
    src_pts = np.float32([key1[m.queryIdx].pt for m in mat]).reshape(-1, 1, 2)
    dst_pts = np.float32([key2[m.trainIdx].pt for m in mat]).reshape(-1, 1, 2)
    # Apply Ransac
    matr, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2)
    return mask.ravel().tolist()

def similarity(matches_num):
    key_points = min(len(key1),len(key2))
    similar_res = len(matches_num)/key_points
    return similar_res * 100

if __name__ == '__main__':
    f_path = "assignment data"
    images,im_name = reading_data(f_path)
    sift = cv2.xfeatures2d.SIFT_create()
    bf = cv2.BFMatcher()
    sim_data = list()
    """
    images = [
    [img1,img2,img3]
    [img4,img5]
    ]
    """
    for i in images:
        """
            in 1st iteration
            i = [img1,img2,img3]
        """
        # getting key points of each image
        key_des = calculating_key_points(i)

        # calculating combinations
        """
            lst = [1,2,3]
            com = [(1,2),(1,3),(2,3)]
        """
        len_image = len(i)
        lst = list(range(len_image))
        com = list(comb(lst,2))

        # iterating over each combination and applying filters
        for j in com:
            """
                in 1st iteration
                j = (1,2)
                access images = i[x], i[y]
                steps:
                    1. matches between images
                    2. apply filters
            """
            """
                key_des:
                [(img1 data), (img2 data), (img3 data)]
                [(key1,des1), (key2,des2), (key3,des3)]
            """
            x,y = j
            key1 = key_des[x][0]
            key2 = key_des[y][0]
            des1 = key_des[x][1]
            des2 = key_des[y][1]


            matches =  bf.knnMatch(des1,des2,2)
            # Applying david lowe's filter
            match2 = ratio_of_david_lowe(0.75,matches)
            #print(type(match2))
            # Applying threshold filter
            match3 = thresholding(match2)

            #print(type(match3))

            # Applying cross-matching filter
            match4 = cross_check(match3)


            # Applying ransac
            match5 = ransac(match4)
            #print(match5)
            """print(f"m1 {len(matches)}")
            print(f"m2 {len(match2)}")
            print(f"m3 {len(match3)}")
            print(f"m4 {len(match4)}")
            print(f"m5 {len(match5)}")"""

            # calculating similarity
            sim_per = similarity(match5)
            sim_data.append(sim_per)

            # Draw matches
            img4 = cv2.drawMatches(i[x],key_des[x][0],i[y],key_des[y][0], match4,matchesMask=match5, flags=2, outImg=None, matchColor=(0,255,255))
            plt.imshow(img4)
            plt.show()

    # Getting the names of the images to be used in printing similarity
    c_name = list()
    for nm in im_name:
        x = list(comb(nm, 2))
        for vals in x:
            c_name.append([vals[0],vals[1]])
    # Apply similarity condition based on the similarity of 2 images which aren't similar
    comp_val = sim_data[-2]
    for i in range(len(sim_data)):
        a,b = c_name[i]
        if comp_val < sim_data[i]:
            print(f"{a} and {b} are similar")
        else:
            print(f"{a} and {b} aren't similar")
