import os
import cv2
import numpy as np
from numpy.linalg import norm

from pathlib import Path

import collections
import time
import pickle
import bz2
import gzip
import multiprocessing
from functools import partial


def norm_features():
    # features = np.load("features.npy")

    # f = gzip.GzipFile('features.npy.gz', "w")
    # np.save(f, features)

    # features = features / norm(features, axis=1, keepdims=True)
    # np.save("features_norm.npy", features)


    # features_customer = np.load("features_customers.npy")
    # print(features_customer.shape)

    # features_customer = features_customer / norm(features_customer, axis=1, keepdims=True)
    # np.save("features_customers_norm.npy", features_customer)
    return

def load_features():

    features = np.load("features_norm.npy")
    print(features.shape)

    features_customer = np.load("features_customers_norm.npy")
    print(features_customer.shape)

    return features, features_customer


def get_similarities():
    features, features_customer = load_features()

    with open('train_dict.pickle', 'rb') as handle:
        train_dict = pickle.load(handle)

    count = 0
    features_customer_dict = {}
    for i, (customer, articles) in enumerate(train_dict.items()):
        customer_vector = features_customer[count:count+len(articles)]
        count += len(articles)
        features_customer_dict[customer] = customer_vector


def multi_test(player_embeds, players, encoder_path, num_test_games, test_player_dir):

    player_name = test_player_dir.stem
    if player_name not in players.keys():
        print("error {} not in training".format(player_name))
        return

    # https://stackoverflow.com/questions/50412477/python-multiprocessing-grab-free-gpu
    cpu_name = multiprocessing.current_process().name
    cpu_id = int(cpu_name[cpu_name.find('-') + 1:])
    gpu_id = cpu_id % torch.cuda.device_count()
    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")

    encoder.load_model(encoder_path, multi_gpu=False, device=device)

    player_games = [np.load(gzip.GzipFile(f, "r")) for f in test_player_dir.iterdir() if f.suffix == '.gz']

    accuracy = -1
    if len(player_games) == 0:
        print("{}, empty folder, missing test data...".format(player_name))
    else:
        game_embeds = encoder.embed_games(player_games, num_test_games=num_test_games, drop_last=True)
        torch.cuda.empty_cache()
        sims = np.inner(game_embeds, player_embeds)
        preds = np.argmax(sims, axis=1)
        player_index = players[player_name]
        preds_correct = len(preds[preds==player_index])
        accuracy = preds_correct / len(preds)
        print("Player: {}, Accuracy: {}, Number of Games: {}".format(player_name, accuracy, len(player_games)))
    
    return (player_name, accuracy)

def run_model(player_embeds, players, encoder_path, num_test_games, verification_data_dir):
    pool = multiprocessing.Pool(4)
    func = partial(multi_test, player_embeds, players, encoder_path, num_test_games)
    verification_data_dir = [f for f in verification_data_dir.iterdir() if f.stem in players.keys()]
    print(len(verification_data_dir))
    data = pool.map(func, verification_data_dir)
    pool.close()
    pool.join()

    return dict(data)


if __name__ == "__main__":
    get_similarities()
