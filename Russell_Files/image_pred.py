import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.neighbors import NearestNeighbors

# from keras.applications.xception import Xception,preprocess_input
# import tensorflow as tf
# from keras.preprocessing import image
# from keras.layers import Input
# from keras.backend import reshape
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import collections
import time

images_dir = '/cs282/shared/h-and-m-personalized-fashion-recommendations/images'

data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def getImagePaths(path):
    """
    Function to Combine Directory Path with individual Image Paths
    
    parameters: path(string) - Path of directory
    returns: image_names(string) - Full Image Path
    """
    image_names = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            fullpath = os.path.join(dirname, filename)
            image_names.append(fullpath)
    return image_names

def preprocess_img(img_path):
    dsize = (288,288)
    new_image=cv2.imread(img_path)
    new_image=cv2.resize(new_image,dsize,interpolation=cv2.INTER_NEAREST)  
    new_image=data_transforms(new_image)
    new_image=new_image[None,:]
    # new_image=np.expand_dims(new_image,axis=0)
    # new_image=torch.from_numpy(new_image).float() # change preprocessing...
    # new_image=new_image.permute((0, 3, 1, 2))
    return new_image

def load_data():
    img_paths=[]
    # img_paths=getImagePaths(images_dir)[:10000]
    img_paths=getImagePaths(images_dir)
    return img_paths

# class ResNetPool(nn.Module):
#     def __init__(self):
#         super(ResNetPool, self).__init__()
#         original_model = models.resnet18(pretrained=True)
#         # print(original_model)
#         self.features = nn.Sequential(
#             *list(original_model.children())[:-1]
#         )
#     def forward(self, x):
#         x = self.features(x)
#         return x

def model():
    model = models.efficientnet_b2(pretrained=True)
    # model = ResNetPool()
    model.cuda()
    model.eval()
    return model

def feature_extraction(image_data,model):
    features=model(image_data.cuda())
    features=features.detach().cpu().numpy()
    # features=features.flatten()
    # print(features.shape)
    return features


def result_vector_cosine(model,feature_vector,new_img):
    new_feature = model(new_img.cuda())
    new_feature = new_feature.detach().cpu().numpy()
    new_feature = new_feature.flatten()
    N_result = 12
    nbrs = NearestNeighbors(n_neighbors=N_result, metric="cosine").fit(feature_vector)
    distances, indices = nbrs.kneighbors([new_feature])
    
    return(distances, indices)

def input_show(data):
    plt.figure()
    plt.title("Query Image")
    plt.imshow(data)

def show_result(data,result):
    fig = plt.figure(figsize=(12,8))
    for i in range(0,12):
        index_result=result[0][i]
        plt.subplot(3,4,i+1)
        plt.imshow(cv2.imread(data[index_result]))
    plt.show()

img_paths=load_data()
img_paths_set = set([Path(p).stem for p in img_paths])
print(len(img_paths_set))

img_to_idx = {}
path_dict = {} # {article: article_path}
for i, p in enumerate(img_paths):
    path_dict[Path(p).stem[1:]] = p
    img_to_idx[Path(p).stem[1:]] = i

main_model=model()

features = np.load("features.npy")
print(features.shape)

import pickle
with open('train_dict.pickle', 'rb') as handle:
    train_dict = pickle.load(handle)

print(len(train_dict))


customer_predictions = {}
for i, (customer, articles) in enumerate(train_dict.items()):
    distances = []
    results = []
    # cur_time = time.time()
    for article in articles:
        article = str(article)
        distance, result=result_vector_cosine(main_model,features,preprocess_img(path_dict[article]))

        # remove the one that has 0 distance (comparing against itself)
        distance = distance[0].tolist()[1:]
        result = result[0].tolist()[1:]

        # print(distance, result)
        distances += distance
        results += result

        # input_show(cv2.imread(path_dict[article]))
        # show_result(img_paths,result)
    # print("time for finding similarities: {}".format(time.time() - cur_time))
    # cur_time = time.time()

    distances = np.array(distances)
    results = np.array(results)
    inds = distances.argsort()
    sorted_results = results[inds]
    sorted_distances = distances[inds]
    # print(sorted_results)
    # print(sorted_results.tolist())

    # remove duplicates
    _ , idx = np.unique(sorted_results, return_index=True)
    idx = np.sort(idx)
    sorted_results = sorted_results[idx]
    sorted_distances = sorted_distances[idx]

    sorted_results = sorted_results[:12]
    results_articles = [(Path(img_paths[sorted_results[i]]).stem, sorted_distances[i]) for i in range(len(sorted_results))]
    customer_predictions[customer] = results_articles
    handle = open('submission_intermediate.pickle', 'wb')
    pickle.dump(customer_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # print("time for final results: {}".format(time.time() - cur_time))
    # break

    if i % 50 == 0 and i != 0:
        print("progress {}th customers done finding similarity".format(i))
        # break

# i = 0
# for key, values in customer_predictions.items():
#     print(key, values)
#     i += 1
#     if i == 5:
#         break

# with open('submission.pickle', 'wb') as handle:
#     pickle.dump(customer_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
