from utils import *
import os

def image_dataset(image):
    '''
    image: jpg (./rare_disease)
    '''
    def __init__(self, image_path):
        self.image_path = image_path
        self.image_list = os.listdir(image_path)
        self.image_list.sort()
        self.image_list = [os.path.join(image_path, image) for image in self.image_list]
        

