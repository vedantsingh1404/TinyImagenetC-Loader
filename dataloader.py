import numpy as np
import os
import glob2 as glob
from torch.utils.data import Dataset
from PIL import Image
import cv2
import torchvision
import torch

CLASS_LIST_FILE = 'wnids.txt'
EXTENSION = 'JPEG'

class CorruptTiny(Dataset):

    def __init__(self, root, severity,  corrupt, transform=None target_transform=None):
        self.root = os.path.expanduser(root)
        self.corrupt = corrupt
        self.severity = severity
        self.transform = transform
        self.target_transform = target_transform
        self.corrupt_dir = os.path.join(self.root, self.corrupt, str(self.severity))

        self.image_paths = sorted(glob.iglob(os.path.join(self.corrupt_dir, '**', '*.%s' % EXTENSION), recursive=True))

        self.labels = {}

        with open(os.path.join(self.root, CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        for text, i in self.label_text_to_number.items():
            for _, _, filenames in os.walk(os.path.join(self.corrupt_dir, text)):
                for f_name in filenames:
                   self.labels[f_name] = i

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        file_path = self.image_paths[index]
        img = cv2.imread(file_path)
        img = torchvision.transforms.functional.to_pil_image(img)
        img = self.transform(img)

        return img, self.labels[os.path.basename(file_path)]

