import cv2
import numpy as np
import os
import csv
from matplotlib import pyplot as plt

def extract_bv(image):      
    b,green_fundus,r = cv2.split(image) #split the image into blue green and red channel
    
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced_green_fundus = clahe.apply(green_fundus)

    # applying alternate sequential filtering (3 times closing opening)
    r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)    
    f4 = cv2.subtract(R3,contrast_enhanced_green_fundus)
    f5 = clahe.apply(f4)        

    # removing very small contours through area parameter noise removal
    ret,f6 = cv2.threshold(f5,10,255,cv2.THRESH_BINARY)
    
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255  
    contours, hierarchy = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
            
    im = cv2.bitwise_and(f5, f5, mask=mask)
    
    ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)
    
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)   

    # removing blobs of unwanted bigger chunks taking in consideration they are not straight lines like blood
    #vessels and also in an interval of area
    fundus_eroded = cv2.bitwise_not(newfin) 
    xmask = np.ones(fundus.shape[:2], dtype="uint8") * 255
    xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)    
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)                  
        if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
            shape = "circle"    
        else:
            shape = "veins"
        
    finimage = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)  
    blood_vessels = cv2.bitwise_not(finimage)
    
    kernel = np.ones((2,2), np.uint8)
    blood_vessels = cv2.subtract(255, blood_vessels)

    #eroding is done to eliminate the minor noises (post processing)
    new = cv2.morphologyEx(blood_vessels, cv2.MORPH_OPEN, kernel)
    new1 = cv2.morphologyEx(new, cv2.MORPH_CLOSE, kernel)
    titles = ['Original Image','Output of CLAHE on Green Channel','Alternate Sequential Filtering','Subtraction Result','Threshold Image with Noise and extra elements','Segmented Blood Vessels']
    images = [image, contrast_enhanced_green_fundus, R3, f4, newfin, new1]
    plt.subplot(2,3,1),plt.imshow(cv2.cvtColor(images[0],cv2.COLOR_BGR2RGB))
    plt.title(titles[0])
    plt.xticks([]),plt.yticks([])
    for i in range(1,6):
        plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()
    return new1    

if __name__ == "__main__":  
    pathFolder = "/home/mutte_ur_rehman/Desktop/retina-features/images"
    filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder,x))]
    destinationFolder = "seg"
    if not os.path.exists(destinationFolder):
        os.mkdir(destinationFolder)
    
    
    
    for file_name in filesArray:
        file_name_no_extension = os.path.splitext(file_name)[0]
        fundus = cv2.imread(pathFolder+"/"+file_name)
        bloodvessel = extract_bv(fundus)
        cv2.imwrite(destinationFolder+file_name_no_extension+"_bloodvessel.png",bloodvessel)
    
