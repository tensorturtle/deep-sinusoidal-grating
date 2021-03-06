import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import scipy.stats
from scipy import ndimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
import io
import random
from pathlib import Path
import os
import shutil

pil_to_tensor = transforms.Compose(
    [transforms.PILToTensor(),
     transforms.ConvertImageDtype(torch.float)
    ])

def show_img(image, **kwargs):
    plt.figure()
    plt.axis('off')
    plt.imshow(image, cmap="Greys", **kwargs)

def y_sinusoid(size=(256,256), frequency=4, phase_shift=0):
    '''
    Draw a sinusoidal grating that changes value across y axis
    '''
    x = np.arange(size[0])
    y = np.arange(size[1])
    X,Y = np.meshgrid(x,y)
    Z = np.sin(2*np.pi * frequency * (Y/size[0]) + phase_shift)
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
    #return pil_img

def circular_sinegrate(frequency, rotation, image_size=(256,256), phase_shift=0):
    '''
    Generate a circular sinusoidal grating.
    
    frequency (float) : frequency of the sinusoid
    rotation (float) : counterclockwise rotation of the sinusoid in degrees
    phase_shift (float): move phase right/left. Radians 
    size (int,int) : size of the output image
    '''
    np_sinegrate = y_sinusoid(image_size, frequency, phase_shift)
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
    def __init__(self, cat_scheme='rb', 
                 dist_params = None,
                 visual_angle='5', 
                 length=100, 
                 image_size=(256,256), 
                 transform=None,
                randomize_phase=True):
        '''
        PyTorch Sinusoidal Grating image dataset generator
        
        cat_scheme (string): categorization scheme- 'rb'(rule-based) or 'ii'(information-integration)
        visual_angle (float): angle in degrees that the image occupies in the human visual field
        length (int): arbitrary length of the dataset. Data is generated on-the-fly using the distribution defined by `cat_scheme`.
        image_size (int,int): pixel size of output image
        '''
        self.cat_scheme=cat_scheme 
        #self.dist_params = dist_params
        self.visual_angle = visual_angle
        self.length = length 
        self.image_size = image_size
        self.transform = transform
        self.randomize_phase = randomize_phase
        
        #if dist_params is None:
            
        
        if self.cat_scheme == 'rb':
            assert len(np.array(dist_params['a_means']).shape) == 2, "Rule-based scheme's 'a_means' should be a 2-d list"
            assert len(np.array(dist_params['b_means']).shape) == 2, "Rule-based scheme's 'b_means' should be a 2-d list"
            assert len(np.array(dist_params['a_covariances']).shape) == 3, "Rule-based scheme's 'a_covariances' should be a 3-d list"
            assert len(np.array(dist_params['b_covariances']).shape) == 3, "Rule-based scheme's 'b_covariances' should be a 3-d list"
        elif self.cat_scheme == 'ii':
            assert len(np.array(dist_params['a_means']).shape) == 1, "Rule-based scheme's 'a_means' should be a 1-d list"
            assert len(np.array(dist_params['b_means']).shape) == 1, "Rule-based scheme's 'b_means' should be a 1-d list"
            assert len(np.array(dist_params['a_covariances']).shape) == 2, "Rule-based scheme's 'a_covariances' should be a 2-d list"
            assert len(np.array(dist_params['b_covariances']).shape) == 2, "Rule-based scheme's 'b_covariances' should be a 2-d list"
        
        self.parse_params(dist_params)
        
    def save_dataset(self, path, extension='png'):
        self.generate_dataset()
        path = Path(path)
        if os.path.exists(path/'A'):
            shutil.rmtree(path/'A')
        os.makedirs(path/'A')
            
        if os.path.exists(path/'B'):
            shutil.rmtree(path/'B')
        os.makedirs(path/'B')

        for i, (label, pil_image) in enumerate(self.a_dataset):
            pil_image.save(path/'A'/f'{i}.{extension}')
        for i, (label, pil_image) in enumerate(self.b_dataset):
            pil_image.save(path/'B'/f'{i}.{extension}')
        
    def generate_dataset(self):
        # label 0 refers to 'a' condition
        # label 1 refers to 'b' condition
        self.a_dataset = [(0, self.get_image(parameters[0], parameters[1], randomize_phase=self.randomize_phase)) for parameters in self.a_params]
        self.b_dataset = [(1, self.get_image(parameters[0], parameters[1], randomize_phase=self.randomize_phase)) for parameters in self.b_params]
    
    def parse_params(self, dist_params):
        if self.cat_scheme == 'rb':
            # in 'rb' condition, the parameters are composed of two distributions.
            # here we generate paramers for each of the two and fuse them into one
            self.a1_params = generate_params(mean=dist_params['a_means'][0], cov=dist_params['a_covariances'][0], size=self.length//2, categorization_scheme=self.cat_scheme)
            self.a2_params = generate_params(mean=dist_params['a_means'][1], cov=dist_params['a_covariances'][1], size=self.length//2, categorization_scheme=self.cat_scheme)
            self.a_params = np.vstack((self.a1_params, self.a2_params))
            
            self.b1_params = generate_params(mean=dist_params['b_means'][0], cov=dist_params['b_covariances'][0], size=self.length//2, categorization_scheme=self.cat_scheme)
            self.b2_params = generate_params(mean=dist_params['b_means'][1], cov=dist_params['b_covariances'][1], size=self.length//2, categorization_scheme=self.cat_scheme)
            self.b_params = np.vstack((self.b1_params, self.b2_params))
            
        elif self.cat_scheme == 'ii':
            self.a_params = generate_params(mean=dist_params['a_means'], cov=dist_params['a_covariances'], size=self.length, categorization_scheme=self.cat_scheme)
            self.b_params = generate_params(mean=dist_params['b_means'], cov=dist_params['b_covariances'], size=self.length, categorization_scheme=self.cat_scheme)
    
    def get_image(self, frequency, orientation, randomize_phase=True):
        freq = float(frequency) * float(self.visual_angle)
        orientation = np.rad2deg(orientation)
        phase_shift = random.uniform(0, 2*np.pi) if randomize_phase else 0
        img = circular_sinegrate(freq, orientation, image_size=self.image_size, phase_shift=phase_shift)
        return img
    
    def __len__(self):
        return self.length * 2 # since 'a' and 'b' are each self.length items.
    
    def __getitem__(self, idx):
        fetched_data = list((self.a_dataset + self.b_dataset)[idx])
        if self.transform is not None:
            fetched_data[1] = self.transform(fetched_data[1])
        return fetched_data
    
    def set_dist_params(self, new_dist_params):
        '''
        Use this setter function to update self.dist_params after instantiating class. 
        Used for interactive distribution parameter setting via ipywidgets
        '''
        self.parse_params(new_dist_params)
    
    def plot_final(self):
        '''
        Return figure representing the distribution of final dataset
        
        Usage:
        
        plt.show(dataset.plot_final())
        
            where
            dataset: an instance of this dataset class
        '''
        #plt.rcParams['figure.figsize'] = (5,5)
        if self.cat_scheme == 'rb':
            plt_figure = plt.figure(figsize=(5,5))
            #axarr = plt_figure.add_subplot(1,1,1)
            plt.scatter(self.a1_params[:,0], self.a1_params[:,1], s=60, marker='+', color='black')
            plt.scatter(self.a2_params[:,0], self.a2_params[:,1], s=60, marker='+', color='black')
            plt.scatter(self.b1_params[:,0], self.b1_params[:,1], facecolors='none', edgecolors='gray')
            plt.scatter(self.b2_params[:,0], self.b2_params[:,1], facecolors='none', edgecolors='gray')
            plt.axis([0.0, 4.0, 0.0, 1.6])
            plt.yticks(np.arange(0, 2.1, 0.5))
            plt.xticks(np.arange(0,4.1,1))
        elif self.cat_scheme == 'ii':
            plt_figure = plt.figure(figsize=(5,5))
            plt.scatter(self.a_params[:,0], self.a_params[:,1], s=60, marker='+', color='black')
            plt.scatter(self.b_params[:,0], self.b_params[:,1], facecolors='none', edgecolors='gray')
            plt.axis([0.0, 4.0, 0.0, 1.6])
            plt.yticks(np.arange(0, 2.1, 0.5))
            plt.xticks(np.arange(0,4.1,1))
        else:
            print(f"Category type 'self.cat_scheme': {self.cat_scheme} is not supported.")
            
        fig = plt.figure()
        fig.canvas.draw()
        #data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        return fig

            

if __name__=="__main__":
    dataset = SineGrates(cat_scheme='rb', length=200)
    print(next(iter(dataset)))