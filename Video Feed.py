import cv2
import matplotlib.pyplot as plt
import cvlib as cv
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable
import transforms as transforms
from skimage import io
from skimage.transform import resize
from models import *

cam = cv2.VideoCapture(0)

cv2.namedWindow("Video Feed")

img_counter = 0

while True:
    
    ret, frame = cam.read()
    cv2.imshow("Video Feed", frame)
    
    if not ret:
        break
    
    k = cv2.waitKey(1)

    # ESC pressed
    if k%256 == 27:
        
        print("Escape hit, closing...")
        break

    # SPACE pressed
    elif k%256 == 32:
        
        img_name = 'images\\raw\\' + str(img_counter) + '.png'
        cv2.imwrite(img_name, frame)

        image_path = img_name
        im = cv2.imread(image_path)

        faces, confidences = cv.detect_face(im)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)





        
        # Detect multiple faces in the image
        for face in faces:
            (startX,startY) = face[0],face[1]
            (endX,endY) = face[2],face[3]
    
            # Place bounding box over face and extract image
            plt.axis('off')
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            ROI = im[startY:endY, startX:endX]
            plt.imshow(ROI)
            plt.savefig("images\processed\{}.png".format(img_counter))





            cut_size = 44

            transform_test = transforms.Compose([transforms.TenCrop(cut_size), transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),])

            def rgb2gray(rgb):
                return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

            #Select the image
            raw_img = io.imread("images\processed\{}.png".format(img_counter))
            gray = rgb2gray(raw_img)
            gray = resize(gray, (48,48), mode='symmetric').astype(np.uint8)

            img = gray[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis=2)
            img = Image.fromarray(img)
            
            inputs = transform_test(img)

            class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

            net = VGG('VGG19')
            checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'), map_location=torch.device('cpu'))
            net.load_state_dict(checkpoint['net'])
            net.eval()

            ncrops, c, h, w = np.shape(inputs)

            inputs = inputs.view(-1, c, h, w)

            inputs = Variable(inputs, volatile=True)
            
            outputs = net(inputs)

            # Average over crops
            outputs_avg = outputs.view(ncrops, -1).mean(0)

            score = F.softmax(outputs_avg)
            _, predicted = torch.max(outputs_avg.data, 0)
     
            print("The Expression is %s" %str(class_names[int(predicted.cpu().numpy())]))

        img_counter += 1

cam.release()

cv2.destroyAllWindows()
