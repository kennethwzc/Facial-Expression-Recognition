"""
visualize results for test image
"""

import numpy as np
import matplotlib.pyplot as plt
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

cut_size = 44

transform_test = transforms.Compose([transforms.TenCrop(cut_size), transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),])

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#write code to select image
raw_img = io.imread('images/1.jpg')
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

# avg over crops
outputs_avg = outputs.view(ncrops, -1).mean(0)


score = F.softmax(outputs_avg)
_, predicted = torch.max(outputs_avg.data, 0)

plt.rcParams['figure.figsize'] = (13.5,5.5)
axes=plt.subplot(1, 3, 1)
plt.imshow(raw_img)
plt.xlabel('Input Image', fontsize=16)
axes.set_xticks([])
axes.set_yticks([])
plt.tight_layout()

plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, top=0.9, hspace=0.02, wspace=0.3)

plt.subplot(1, 3, 2)

# the x locations for the groups
ind = 0.1+0.6*np.arange(len(class_names))

# the width of the bars: can also be len(x) sequence
width = 0.4

color_list = ['red','orangered','darkorange','limegreen','darkgreen','royalblue','navy']

for i in range(len(class_names)):
    plt.bar(ind[i], score.data.cpu().numpy()[i], width, color=color_list[i])
    
print("The Expression is %s" %str(class_names[int(predicted.cpu().numpy())]))


