import cv2
import numpy as np
from imageio import get_reader
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor


class ImageDataset(Dataset):
    
    def __init__(self, images, is_video=False):
        self.images = images
        self.is_video = is_video 
        if is_video:
            cap = cv2.VideoCapture(images)
            self.n_imgs = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.reader = get_reader(images, mode='I')
        else:
            self.n_imgs = len(images)

    def __len__(self):
        return self.n_imgs
    
    def __getitem__(self, i):
        
        if self.is_video:
            image = np.asarray(self.reader.get_data(i))
        else:
            image = read_image(self.images[i])

        image = image.transpose(2, 0, 1)
        return image
    
    def close(self):
        self.reader.close()
