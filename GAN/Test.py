import os
import numpy as np

from scipy import io
from matplotlib import pyplot as plt

import torch
import torch.nn.functional as F
import torchvision.transforms as T


real = "C:/Users/PHOENIX/Desktop/PseudoCT/Fake/Train/CT/CT02.mat"
fake = "C:/Users/PHOENIX/Desktop/PseudoCT/Fake/Train/CT/CT02.mat"

real_a = io.loadmat(real)['CT'].astype('float32')[90]
fake_a = io.loadmat(fake)['CT'].astype('float32')[90]

real_a = np.expand_dims(real_a, 0)
fake_a = np.expand_dims(fake_a, 0)

print(real_a.shape)
print(fake_a.shape)

colormap = plt.get_cmap('coolwarm')

diff = np.abs(real_a - fake_a)
diff -= diff.min()
diff /= diff.max()
diff = colormap(diff[0])
diff = diff[..., :3]

print(diff.shape)

plt.figure()
plt.imshow(diff)
plt.show()