# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 02:19:58 2020

@author: belka
"""


import cv2
import matplotlib.pyplot as plt
import numpy as np



def make_coordinates(image , line_parameters):
 #   slope, intercept = line_parameters
    try:
        slope, intercept = line_parameters
    except TypeError:
        slope, intercept = 0.001, 0
    y1 = img.shape[0]
    y2 = int(y1*(0.6))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])




def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        # reshape each line into 1D array of 4 parameters x1, y1, x2, y2
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        #fit a polynomial to ou x,y points and return the two parameters of each line 'm' and 'b' 
        #syntax : np.polyfit(x,y,DEGREE_OF_THE POLYNOMIAL)
        slope = parameters[0]
        #defining the slopes in the columns [0]
        intercept = parameters[1]
        print(slope, 'slopes')
        print(intercept, 'intercepts')
        print(parameters, 'patameters')
        ######################################################################
        #CHECKING IF THE LINE CORECPONDS INTO A LEFET LINE OR RIGHT LINE
        if slope < 0:
            left_fit.append((slope, intercept))
            #move the parameters into left_fit list
        else:
            right_fit.append((slope, intercept))
            #move the parameters into right_fit list 
        print(left_fit, 'left')
        print(right_fit, 'right')
        print(line, 'line')
        ######################################################################
        #calculating the average of the slopes and the average of intecepts
        left_fit_average = np.average(left_fit, axis=0)
        #syntax : np.average(the_values, axis=colums_of the values)
        right_fit_average = np.average(right_fit, axis=0)
        ######################################################################
        print(left_fit_average)
        print(right_fit_average)
        left_line  = make_coordinates(img, left_fit_average) 
        right_line = make_coordinates(img, right_fit_average) 
    return np.array([left_line, right_line])
        
            


def canny(image) :
     imgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
     imBlur = cv2.GaussianBlur(imgray, (5,5),0)
     canny = cv2.Canny(imBlur, 50, 150)
     return canny

def display_lines (image, lines):
    line_image = np.zeros_like(image)
    #create a black image with same size as the img in this case "image"
    #############CHEKING IF THE ALGORITHM DETECTS A LINE OR NOT##############
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2  = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255,50,50),5)
    return line_image       
    
def region_of_interest(image):
    height = image.shape[0] 
    #getting the height of the image 
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
        ])
    #detrmine the 3 the points corners of the triangle of the road in form of a numpy array
    #syntax: np.array([(X1,Y1),(X2,Y2)),(X3,Y3)])  
    mask = np.zeros_like(image)
    #this means that return an array of zeros with the same shape as the given img in this case "iamge"
    #synatx: np.zeos_like(img_path)
    cv2.fillPoly(mask, polygons, 255)
    ##############finding the lane line by concatinting the mask with our image##############
    masked_image = cv2.bitwise_and(image,mask) 
    #syntax: cv2.bitwise_and(img1,img2) the bitwise function used to make booleans operations like and,or,not,xor...
    return masked_image
    
    


img = cv2.imread("C://Users//belka//Downloads//wayimage.jfif")
canny_img = canny(img)
cropped_img = region_of_interest(canny_img)
# calling the region_of_interest function with the variable canny
lines = cv2.HoughLinesP(cropped_img, 2, np.pi / 180, 50, np.array([]), minLineLength = 40, maxLineGap = 20)
print(lines, 'lines')
#cv2.HoughLinesP() FUNCTION IS USED TO DETECT THE LINES IN AN INPUT IMAGE
#Syntax: cv2.HoughLinesP(img_PATH, pixels_precision, Degree_precision by Radian ,thershold, minLineLenght = ?, maxLineGap = ?)
#threshold : is the min number of votes(intersections) needed to accept a condidate line
#np.array([]) creating an empty array to put inside it the number of the lines detected
#minLineLength: is the min numb of pixels that formes the distance of the line
#maxLineGap: is the max distance between two lines befor joining them together
averaged_lines = average_slope_intercept(img, lines)
line_image = display_lines(img, averaged_lines)####################################averaged_lines
#calling the display_line() funvtion with the two variables "img" wich is the original image and "lines"
combo_image = cv2.addWeighted(img, 0.8, line_image, 1, 1)
#the function cv2.addWeighted() is used for blending (adding) two images 
#syntax: cv2.addWeighted(img1_path, weight_1, img2_path, weight_2, GEMMA)
#THE TWO IMAGES NEEDS TO BE WITHTHE SAME SIZE TO BLENDING THEM
#


 
 
#plt.imshow(imgray)
#plt.imshow(imBlur)
plt.imshow(canny_img)
#################
#cv2.imshow("this is the canny image", canny)
cv2.imshow("this is the image to detect the lines", line_image)
cv2.imshow("this is the image to detect the rigion",combo_image)
plt.imshow(cropped_img)


plt.show()
cv2.waitKey(0)