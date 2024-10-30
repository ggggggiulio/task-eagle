import cv2 as cv
import numpy as np
import os
import urllib.request

def main (image):
    print("you can find the txt files with the coordinates of the cone in: "+ os.getcwd())

    image_swapped = adjust_image(image)
    image_correct = calibrate(image_swapped)
    complete_image = riconoscimento(image_correct)

    cv.imshow("immagine giusta",image_correct)
    cv.imshow("immagine swappata",image_swapped)
    cv.imshow("immagine di base",image)
    cv.imshow("immagine con rettangoli", complete_image)
    f=open("coordinates.txt","r")
    print(f.read())
    cv.waitKey(0)
    cv.destroyAllWindows()

def adjust_image (image1):

    height = image1.shape[0] #960 #
    width = image1.shape[1] 
    
    half_height = int(height/2) #480
    
    bottom_half = image1[:half_height,:width,:] 
    top_half = image1[half_height:height,:width,:]

    new_image = cv.vconcat([top_half,bottom_half])

    return new_image

def calibrate (image2):

    p_b1,p_g1,p_r1 = image2[267,564]
    p_b2,p_g2,p_r2 = image2[541,128]

    pixel_1_corretto = [40,195,240]
    pixel_2_corretto = [250,158,3]

    a_b, b_b = retta(p_b2,pixel_2_corretto[0],p_b1,pixel_1_corretto[0]) #calculate the slope and the shift (ax + b)
    #a_g, b_g = retta(p_g2,pixel_2_corretto[1],p_g1,pixel_1_corretto[1]) doesn't work
    #a_r, b_r = retta(p_r2,pixel_2_corretto[2],p_r1,pixel_1_corretto[2]) doesn't work
    b,g,r = cv.split(image2) #split the image in the basic color of OpenCV

    b1 = np.zeros((b.shape[0],b.shape[1]),np.double) #create new matrix form by 0, this new matrix has the same shape of the matrix for the color blue
    g1 = np.zeros((g.shape[0],g.shape[1]),np.double)
    r1 = np.zeros((r.shape[0],r.shape[1]),np.double)

    b1 = arrange(b,a_b,b_b)
    g1 = arrange(g,5.2,-585)#a_g,b_g) correct value for a_g,b_g
    r1 = arrange(r,1.6,-136)#a_r,b_r) correct value for a_r,b_r

    test = cv.merge((b1,g1,r1)) #merge the 3 new channel of color
    #print(test[541,128])
    #print("pixel",test[267,564])
    return test

def retta(x_s,x_g,y_s,y_g):

    a = (x_g - y_g) / (x_s - y_s) 

    b = x_g - (x_s * a) 

    return a,b

def arrange(color,a,b):

    for x in range(color.shape[0]):
        for y in range(color.shape[1]):
            if color[x,y] != 0:
                new_value = int(a * color[x,y] + b) # ax + b
                color[x, y] = np.clip(new_value, 50, 200) # focre the value between 50 and 200

            else:
                new_value = 50
                color[x,y] = np.clip(new_value,50,200)

    return color

def riconoscimento(image3):

    image4 = image3.copy()
    
    b_low = (90,100,80) 
    b_high = (122,255,255) # value in HSV for the Blue color

    y_low = (10,150,0)
    y_high = (35,255,255) # value in HSV for the Yellow color

    r_low1 = (155,190,180)
    r_high1 = (255,255,255)
    r_low2 = (0,150,200)
    r_high2 = (10,150,255) # value in HSV for the Red color

    image_hsv = cv.cvtColor (image3,cv.COLOR_BGR2HSV) #Convert the image in HSV color-space

    mask_blue = cv.inRange(image_hsv, b_low, b_high) #Isolate all the blue in the image
    mask_yellow = cv.inRange(image_hsv,y_low,y_high) #Isolate all the yellow in the image
    mask_red1 = cv.inRange(image_hsv,r_low1,r_high1) #Isolate all the "high" red in the image
    mask_red2 = cv.inRange(image_hsv,r_low2,r_high2) #Isolate all the "low" red in the image
    mask_red = mask_red1 + mask_red2 # Merge the 2 red

    image_blue = cv.bitwise_and(image3,image3, mask=mask_blue) #create an image where's only blue(intersection between image3 and mask_blue)
    image_yellow = cv.bitwise_and(image3,image3,mask=mask_yellow) #create an image where's only yellow(intersection between image3 and mask_yellow)
    image_red = cv.bitwise_and(image3,image3,mask=mask_red) #create an image where's only red(intersection between image3 and mask_red)

    blue_cone = cv.cvtColor(image_blue, cv.COLOR_BGR2GRAY) #convert the image in gray scale
    yellow_cone = cv.cvtColor(image_yellow, cv.COLOR_BGR2GRAY) #convert the image in gray scale
    red_cone = cv.cvtColor(image_red, cv.COLOR_BGR2GRAY) #convert the image in gray scale

    blue_contours = draw_contours(blue_cone,image4,210,270,"blue_cone")   #function which draw rectangle around the cone
    yellow_contours = draw_contours(yellow_cone,blue_contours,160,200,"yellow_cone")
    red_contours = draw_contours(red_cone,yellow_contours,200,360,"red_cone")

    return red_contours

def draw_contours(image,all_image,l,w,string1):
    
    th1 = cv.adaptiveThreshold(image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2) #create an image with only contours
    edged = cv.Canny(th1, 50, 200) #isolate border
    contours, _ = cv.findContours(edged,cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #array with coordinates of the border
    
    approximation = [None]*len(contours) #array of the same lenght of contours
    boundRect = [None]*len(contours)

    for i, c in enumerate(contours):
        boundRect[i]= cv.boundingRect(approximation[i])  #create the coordinates for rectangles

    x_start = 2000
    y_start = 2000

    for i in range(len(boundRect)): #create only the biggest rectangle around the cone
        a = boundRect[i][0] 
        b = boundRect[i][1]
        c = boundRect[i][2]
        d = boundRect[i][3]
        if a < x_start and a != 0 :
            x_start = a

        if b < y_start and b != 0:
            y_start = b

        if c > l :
            l = c

        if d > w :
            w = d    

    f = open("coordinates.txt", "a") #txt file writing

    all_x = x_start+l
    all_y = y_start+w
    x = str(x_start)
    y = str(y_start)
    k = str(all_x)
    z = str(all_y)

    string = str("$" + string1 + ": (" + x + "," + y + "," + k + "," + z + ")\n")

    Line = [string]
    f.writelines(Line)
    f.close

    cv.rectangle(all_image, (int(x_start), int(y_start)), (int(x_start + l), int(y_start + w)), (0,255,0), 2) #draw the rectangle
    return all_image

if __name__ =="__main__":
    req = urllib.request.urlopen('https://github.com/eagletrt/recruiting-sw/raw/master/driverless/project_1/corrupted.png')
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv.imdecode(arr, -1) 
    main(img)