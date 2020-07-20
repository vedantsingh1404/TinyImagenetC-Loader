import torch
import torchvision.transforms as transforms

from dataloader import *

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

base_path = '' # insert the path to the dataset folder

testloader_corrupt = {}

for corruption in CORRUPTIONS:
    testloader_corrupt[corruption] = {}
    for severity in range(1, 6):
        testset = CorruptTiny(base_path, severity, corruption, transform=None)
        testloader_corrupt[corruption][severity] = torch.utils.data.DataLoader(testset, batch_size = 64, num_workers = 12)

"""
 this dictionary of test loaders can now be used for testing
 the parameters such as batch size and transforms should be tweaked accordingly
"""
 
