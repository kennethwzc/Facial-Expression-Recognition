# import libraries
import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import numpy as np

image_path = '0.png'
im = cv2.imread(image_path)

faces, confidences = cv.detect_face(im)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# loop through detected faces and add bounding box
for face in faces:
    (startX,startY) = face[0],face[1]
    (endX,endY) = face[2],face[3]
    
    # draw rectangle over face
    
    plt.axis('off')
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    ROI = im[startY:endY, startX:endX]
    plt.imshow(ROI)
    plt.savefig('output.png')
    plt.show()
    
