import textwrap
import numpy as np
from scipy import ndimage
import math
from random import randint
import matplotlib.pyplot as plt
import cv2
from datetime import datetime
import myFunctions


print ("*" * 80)
print textwrap.fill('FINAL PROJECT', 80)
print textwrap.fill('CS 563 IMAGE ANALYSIS', 80)
print textwrap.fill('IMPLEMENTED BY: BASSAM ARNAOUT', 80)
print textwrap.fill('SUBMITTED TO: DR. LAYACHI BENTABET', 80)
print textwrap.fill('BISHOPS UNIVERSITY', 80)
print textwrap.fill('\n', 80)
print textwrap.fill('THE OBJECTIVE OF THIS PROGRAM IS TO IMPLEMENT MEAN SHIFT TECHNIQUE FOR COLORED-IMAGE SEGMENTATION', 80)
print "*" * 80


print '\n\nSelect which image from the below you want to process (Enter number between 1 and 4):'
print '1. image1.pgm (AirShip)'
print '2. image2.pgm (Island)'
print '3. image3.pgm (Bears)'
print '4. image4.pgm (Fruits)'


imageNumber = int(raw_input("\nEnter image number: ")) #integer input
print '\n'

imageName = myFunctions.nameOfImageFile(imageNumber)


Hr = int(raw_input("\nEnter Range Domain (Hr), recommended value is between 40 and 90: "))



Iter = 100

img = cv2.imread(imageName,cv2.IMREAD_COLOR)
h,w,d = img.shape
Hs = (h + w)

# functions.plotImageIntoRGBSpace(img)

opImg = np.zeros(img.shape,np.uint8)
boundaryImg = np.zeros(img.shape,np.uint8)




####################################################################################################
# Function: getPixelNeighbors
# It searches the entire Feature Matrix to find the neighbors if a pixel within range of Hs (Spacial Domain)
# and Hr (Range Domain)
#
# Input Parameter:
# param-seed: row of Feature Matrix
# param-featureMatrix : the Feature Matrix extracted from the image
#
# Returns--neighbors : List of neighbors to the seed
####################################################################################################
def getPixelNeighbors(seed,featureMatrix):
    neighbors = []
    nAppend = neighbors.append
    sqrt = math.sqrt
    for i in range(0,len(featureMatrix)):
        cPixel = featureMatrix[i]
        r = sqrt(sum((cPixel[:3]-seed[:3])**2))
        s = sqrt(sum((cPixel[3:5]-seed[3:5])**2))
        if(s < Hs and r < Hr ):
            nAppend(i)
    return neighbors



####################################################################################################
# Function: markPixels
# Deletes the marked pixel from the feature matrix
# It Marks the pixels in an output image with the intensity value gotten after MeanShift Iteration
#
# Input Parameter:
# param-neighbors: row neighbors of the seed from Feature Matrix to me marked
# param-intensity : range and spacial properties of the pixel to be marked
# param-featureMatrix : the Feature Matrix extracted from the image
# param--cluster : Cluster number
####################################################################################################
def markPixels(neighbors,intensity,featureMatrix,cluster):
    for i in neighbors:
        cPixel = featureMatrix[i]
        x=cPixel[3]
        y=cPixel[4]
        opImg[x][y] = np.array(intensity[:3],np.uint8)
        boundaryImg[x][y] = cluster*5
    return np.delete(featureMatrix,neighbors,axis=0)
    # return np.delete(matrix, cPixel, axis=0)




####################################################################################################
# Function: createFeatureMatrix
# Creates a Feature matrix of the image as list of [r,g,b,x,y] for each pixel
#
# Input Parameter:
# param-img: image
#
# return--featureMatrix : Feature matrix
####################################################################################################
def createFeatureMatrix(img):
    h,w,d = img.shape
    print 'image size: h::' + str(h) + ' w::' + str(w)
    featureMatrix = []
    FAppend = featureMatrix.append
    for row in range(0,h):
        for col in range(0,w):
            r,g,b = img[row][col]
            FAppend([r,g,b,row,col])
    featureMatrix = np.array(featureMatrix)
    return featureMatrix




####################################################################################################
# Function: doMeanShift
# This function do the meanShift color-image clustering algorithm on an image
#
# Input Parameter:
# param-img: image
#
# return--clusters : no of cluster detected.
####################################################################################################
def doMeanShift(img):
    clusters = 0
    featureMatrix = createFeatureMatrix(img)

    # Mean shift implementation
    # Iterate over the Feature matrix until it is empty
    while(len(featureMatrix) > 0):
        print 'remPixelsCount : ' + str(len(featureMatrix))

        # Generate a random index between 0 and Length of
        # Feature matrix so that to choose a random
        # Seed (random pixel)
        randomIndex = randint(0,len(featureMatrix)-1)
        seed = featureMatrix[randomIndex]

        # Step 1. for each seed belong to Feature Space, find the neighbouring points.
        # Group all the neighbors based on the threshold Hr and Hs
        neighbors = getPixelNeighbors(seed,featureMatrix)
        print('found neighbors :: '+str(len(neighbors)))

        n_iterations = 1  # was 4
        for it in range(n_iterations):


            #Step 2. for each seed, calculate the mean shift value m(x)
            numerator = 0
            denominator = 0
            kernel_bandwidth = Hr

            for neighbor in featureMatrix[neighbors]:

                distance = myFunctions.euclid_distance(neighbor, seed)
                weight = myFunctions.gaussian_kernel(distance, kernel_bandwidth)
                numerator += (weight * neighbor)
                denominator += weight

            new_x = numerator / denominator
            # print new_x

        meanShift = abs(new_x - seed)
        # print 'meanShift:' + str(meanShift)

        if(np.mean(meanShift)<Iter):
            #Step 3. for each seed , update it to the meanShift Value
            featureMatrix = markPixels(neighbors, new_x, featureMatrix, clusters)
            # print 'a cluster found'
            clusters+=1


    return clusters

# Method main
def main():


    now1 = datetime.now()
    current_time = now1.strftime("%H:%M:%S")
    print '\n'
    print("Start Time =", current_time)

    clusters = doMeanShift(img)
    # origlabelledImage, orignumobjects = ndimage.label(opImg)

    # cv2.imshow('Origial Image',img)
    # cv2.imshow('OP Image',opImg)
    # cv2.imshow('Boundry Image',boundaryImg)
    
    cv2.imwrite('temp.jpg',opImg)
    temp = cv2.imread('temp.jpg',cv2.IMREAD_COLOR)
    # labels, numobjects = ndimage.label(temp)
    # fig, ax = plt.subplots()
    # ax.imshow(labels)
    # ax.set_title('Labeled objects')
    
    print 'Number of clusters formed : ', clusters


    now2 = datetime.now()
    current_time = now2.strftime("%H:%M:%S")
    print '\n'
    print("End Time =", current_time)
    # print '\n'
    print ("Time Taken: ", now2 - now1)



    temp_ = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
    plt.imshow(temp_)
    plt.show()

    myFunctions.plotImageIntoRGBSpace(temp)


if __name__ == "__main__":
    main()

