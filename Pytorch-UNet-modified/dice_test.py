import torch
import torch.nn.functional as F
import numpy as np
#from dice_loss import dice_coeff




# softmax pred
a = np.array([[[1, 0, 0], [0, 0.1, 0.9], [1, 0, 0]],
              [[1, 0, 0], [0, 0.9, 0.1], [1, 0, 0]],
              [[1, 0, 0], [1, 0, 0], [1, 0, 0]]])

a = torch.Tensor(a)

print(a.shape)


b = np.array([[0, 2, 2],
              [0, 1, 1],
              [0, 0, 0]])

b = torch.Tensor(b)
#intersection = 1
#a + b = 3

def dice_coeff(pred, target):
    smooth = 0.000001#1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()

    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

dice = []
probs = F.softmax(a, dim=2).data
max_idx = torch.argmax(probs, 2, keepdim=True)
one_hot = torch.FloatTensor(probs.shape).to(device='cpu')
one_hot.zero_()
one_hot.scatter_(2, max_idx, 1)

for k in range(1, one_hot.shape[2]):
    input = one_hot[:, :, k]
    target = (b == k).float().squeeze(1)
    print(input)
    print(target)
    d = dice_coeff(input, target)
    dice.append(d.item())

print(dice)