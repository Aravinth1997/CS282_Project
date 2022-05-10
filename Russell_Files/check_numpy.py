import numpy as np

feature_small = np.load("features_small.npy")
print(feature_small.shape)


feature = np.load("features.npy")
print(feature.shape)


feature_small_2 = feature[:65]
print(feature_small_2.shape)

print(np.allclose(feature_small, feature_small_2))