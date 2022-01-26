import torch
from torch.utils.data import Dataset, DataLoader

import scipy.stats
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter

def y_sinusoid(size=(256,256), frequency=4):
    '''
    Draw a sinusoidal grating that changes value across y axis
    '''
    x = np.arange(size[0])
    y = np.arange(size[1])
    X,Y = np.meshgrid(x,y)
    Z = np.sin(2*np.pi * frequency * (Y/size[0]))
    return Z

def mask_circle_solid(pil_img, background_color, blur_radius, offset=0):
    '''
    'pil_img' becomes a circle inside a 'background_color' rectangle
    '''
    background = Image.new(pil_img.mode, pil_img.size, background_color)

    offset = blur_radius * 2 + offset
    mask = Image.new("L", pil_img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((offset, offset, pil_img.size[0] - offset, pil_img.size[1] - offset), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))

    return Image.composite(pil_img, background, mask)

def circular_sinegrate(frequency, rotation, image_size=256):
    '''
    Generate a circular sinusoidal grating.
    
    frequency (float) : frequency of the sinusoid
    rotation (float) : counterclockwise rotation of the sinusoid in degrees
    size (int,int) : size of the output image
    '''
    np_sinegrate = y_sinusoid(image_size, frequency)
    rotated_sinegrate = ndimage.rotate(np_sinegrate, rotation, reshape=False)
    pil_sinegrate = Image.fromarray(((rotated_sinegrate*127)+128).astype(np.uint8)) # convert [-1,1] to [0,255]
    return mask_circle_solid(pil_sinegrate, background_color=128, blur_radius=1, offset=18)

def freq_transform(x):
    return x / 30 + 0.25 # cpd (cycles per degree) -> will be later converted to regular cycles per image

def orient_transform(y):
    return np.deg2rad((9/10 * y) + 20) # radians -> will be later converted to degrees

def rotate_points(points, angle=np.pi/4):
    '''
    Rotate a set of points around the center of mass of the points
    
    angle in radians
    '''
    center = np.mean(points,axis=0)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    return np.dot(points - center, rotation_matrix) + center

def generate_params(mean=[30,50], cov=[[10,0],[0,150]], size=100, categorization_scheme='rb'):
    distribution = scipy.stats.multivariate_normal.rvs(mean=mean, cov=cov, size=size)
    if categorization_scheme == 'ii':
        distribution = rotate_points(distribution, np.pi/4)
    return np.array([[freq_transform(x), orient_transform(y)] for x,y in distribution])


class SineGrates(Dataset):
    def __init__(self, cat_scheme='rb', visual_angle='5', length=100, image_size=(256,256), transform=None):
        '''
        PyTorch Sinusoidal Grating image dataset generator
        
        cat_scheme (string): categorization scheme- 'rb'(rule-based) or 'ii'(information-integration)
        visual_angle (float): angle in degrees that the image occupies in the human visual field
        length (int): arbitrary length of the dataset. Data is generated on-the-fly using the distribution defined by `cat_scheme`.
        image_size (int,int): pixel size of output image
        '''
        self.cat_scheme=cat_scheme 
        self.visual_angle = visual_angle
        self.length = length 
        self.image_size = image_size
        self.transform = transform
        if cat_scheme == 'rb':
            self.a_means = [
                [30,50],
                [50,70]]
            self.b_means = [
                [50,30],
                [70,50]]
            self.a_covariances = [
                [[10,0],[0,150]],
                [[150,0],[0,10]]]
            self.b_covariances = [
                [[10,0],[0,150]],
                [[150,0],[0,10]]]
            # in 'rb' condition, the parameters are composed of two distributions.
            # here we generate paramers for each of the two and fuse them into one
            self.a1_params = generate_params(mean=self.a_means[0], cov=self.a_covariances[0], size=self.length//2, categorization_scheme=self.cat_scheme)
            self.a2_params = generate_params(mean=self.a_means[1], cov=self.a_covariances[1], size=self.length//2, categorization_scheme=self.cat_scheme)
            self.a_params = np.vstack((self.a1_params, self.a2_params))
            
            self.b1_params = generate_params(mean=self.b_means[0], cov=self.b_covariances[0], size=self.length//2, categorization_scheme=self.cat_scheme)
            self.b2_params = generate_params(mean=self.b_means[1], cov=self.b_covariances[1], size=self.length//2, categorization_scheme=self.cat_scheme)
            self.b_params = np.vstack((self.b1_params, self.b2_params))
            
        elif cat_scheme == 'ii':
            self.a_means = [40,50]
            self.b_means = [60,50]
            self.a_covariances = [[10,0],[0,280]]
            self.b_covariances = [[10,0],[0,280]]
            
            self.a_params = generate_params(mean=self.a_means, cov=self.a_covariances, size=self.length, categorization_scheme=self.cat_scheme)
            self.b_params = generate_params(mean=self.b_means, cov=self.b_covariances, size=self.length, categorization_scheme=self.cat_scheme)
        
        # label 0 refers to 'a' condition
        # label 1 refers to 'b' condition
        self.a_dataset = [(0, self.get_image(parameters[0], parameters[1])) for parameters in self.a_params]
        self.b_dataset = [(1, self.get_image(parameters[0], parameters[1])) for parameters in self.b_params]
    
    def get_image(self, frequency, orientation):
        freq = float(frequency) * float(self.visual_angle)
        orientation = np.rad2deg(orientation)
        img = circular_sinegrate(freq, orientation, image_size=self.image_size)
        return img
    
    def __len__(self):
        return self.length * 2 # since 'a' and 'b' are each self.length items.
    
    def __getitem__(self, idx):
        fetched_data = list((self.a_dataset + self.b_dataset)[idx])
        if self.transform is not None:
            fetched_data[1] = self.transform(fetched_data[1])
        return fetched_data

if __name__=="__main__":
    dataset = SineGrates(cat_scheme='rb', length=200)
    print(next(iter(dataset)))