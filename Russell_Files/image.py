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

import cudf
train = cudf.read_csv('/home/dataset/transactions_train.csv')
train['customer_id'] = train['customer_id'].str[-16:].str.hex_to_int().astype('int64')
train['article_id'] = train.article_id.astype('int32')
train.t_dat = cudf.to_datetime(train.t_dat)
train = train[['t_dat','customer_id','article_id']]
train.to_parquet('train.pqt',index=False)
print( train.shape )

tmp = train.groupby('customer_id').t_dat.max().reset_index()
tmp.columns = ['customer_id','max_dat']
train = train.merge(tmp,on=['customer_id'],how='left')
train['diff_dat'] = (train.max_dat - train.t_dat).dt.days
train = train.loc[train['diff_dat']<=6]
print('Train shape:',train.shape)

train2 = train[["customer_id", "article_id"]]
train2 = train2.to_pandas()

train_dict = train2.groupby('customer_id')['article_id'].apply(set).to_dict()
i = 0
for key, values in train_dict.items():
    print(key, values)
    i += 1
    if i == 5:
        break



images_dir = '/home/dataset/images'

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

# img_dirs = set(os.listdir('/cs282/shared/h-and-m-personalized-fashion-recommendations/images/'))
remove_article = set()
remove_dict = {}
for i, (customer, articles) in enumerate(train_dict.items()):
    for article in articles:
        article_name = '0' + str(article)
        if article_name not in img_paths_set:
            # print(customer, article_name)
            remove_article.add(str(article))
            if customer not in remove_dict:
                remove_dict[customer] = set([article])
            else:
                remove_dict[customer].add(article)
    if i % 200000 == 0:
        print("progress {}th customers done".format(i))

# original_customer_len = len(train_dict)
# # print(remove_dict)
# for customer, articles in remove_dict.items():
#     for article in articles:
#         train_dict[customer].discard(article)
#     if len(train_dict[customer]) == 0:
#         del train_dict[customer]
# print(original_customer_len)
# print(len(train_dict))
# print("remove {} of customers from dict".format(original_customer_len - len(train_dict)))
# # input_show(cv2.imread(img_paths[1000]))

import pickle
print(len(remove_article))
with open('remove_article.pickle', 'wb') as handle:
    pickle.dump(remove_article, handle, protocol=pickle.HIGHEST_PROTOCOL)


# path_dict = {} # {article: article_path}
# for p in img_paths:
#     path_dict[Path(p).stem[1:]] = p

# print(path_dict)
# print(img_paths)

# main_model=model()
# # print(main_model)

# img_paths = np.load("img_path.pickle", allow_pickle=True)
# img_paths = [p.replace("/cs282/shared/h-and-m-personalized-fashion-recommendations", "/home/dataset") for p in img_paths]
# print(len(img_paths))
# print(img_paths[0])

# inputs = []
# features = []
# features_dict = {}
# for i, article in enumerate(img_paths):
#     new_img=preprocess_img(article)
#     inputs.append(new_img)
#     # features[key]=feature_extraction(new_img,main_model)
#     if i % 64 == 0 and i != 0:
#         inputs = torch.stack(inputs)
#         inputs = inputs.squeeze(1)
#         if len(features) == 0:
#             features = feature_extraction(inputs,main_model)
#         else:
#             # print(features.shape, (feature_extraction(inputs[i:i+batch_size],main_model).shape))
#             features = np.concatenate((features, feature_extraction(inputs, main_model)))

#         inputs = []
#         print("done processing image for {}th articles".format(i))
#         break
# # if len(inputs) != 0:
# #     inputs = torch.stack(inputs)
# #     inputs = inputs.squeeze(1)
# #     features = np.concatenate((features, feature_extraction(inputs, main_model)))


# print(features.shape)
# np.save("features_small.npy", features)

# import pickle
# with open('train_dict.pickle', 'wb') as handle:
#     pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('img_path.pickle', 'wb') as handle:
#     pickle.dump(img_paths, handle, protocol=pickle.HIGHEST_PROTOCOL)




# customer_predictions = {}
# for i, (customer, articles) in enumerate(train_dict.items()):
#     distances = []
#     results = []
#     for article in articles:
#         article = str(article)
#         distance, result=result_vector_cosine(main_model,features,preprocess_img(path_dict[article]))
#         # print(distance, result)
#         distances += distance[0].tolist()
#         results += result[0].tolist()
#         # input_show(cv2.imread(path_dict[article]))
#         # show_result(img_paths,result)

#     # print(distances)
#     # print(results)

#     distances = np.array(distances)
#     results = np.array(results)
#     inds = distances.argsort()
#     sorted_results = results[inds]
#     sorted_distances = distances[inds]
#     # print(sorted_results)
#     # print(sorted_results.tolist())

#     results_no_duplicate = list(dict.fromkeys(sorted_results.tolist()))
#     results_no_duplicate = results_no_duplicate[:12]
#     # print(results_no_duplicate)
    
#     results_articles = [Path(img_paths[i]).stem for i in results_no_duplicate]
#     customer_predictions[customer] = results_articles
#     # break
#     if i % 200000 == 0:
#         print("progress {}th customers done finding similarity".format(i))

# i = 0
# for key, values in customer_predictions.items():
#     print(key, values)
#     i += 1
#     if i == 5:
#         break