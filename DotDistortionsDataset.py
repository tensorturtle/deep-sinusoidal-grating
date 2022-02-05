import random
import pickle
from datetime import datetime
import os

import torch
import numpy as np
import imageio
import cv2

from torch.utils.data import Dataset
import torchvision.transforms as transforms

normalize_transform = transforms.Compose(
    [transforms.ConvertImageDtype(torch.float)
    ])

def random_points_from_grid(seed=None, num_points=9, central_area_side=30, total_area_side=50):
    '''
    "Choose 9 points randomly from the central 30x30 area of a 50x50 grid"
    Becomes the initial point of the dot distortion shapes.
    '''
    if seed is not None:
        random.seed(seed)
    points = []
    for i in range(num_points):
        x = random.randint(total_area_side/2-central_area_side/2, total_area_side/2 + central_area_side/2)
        y = random.randint(total_area_side/2-central_area_side/2, total_area_side/2 + central_area_side/2)
        points.append((x, y))
    return points

def create_relative_coordinates(dot_block_image_path='dot_distortion_areas.png'):
    '''
    I created a 21x21 pixel PNG image using GIMP that has different brightness values for each kind of 'area'.
    This function parses that image to return a list of tuples of relative coordinates for each area.
    Areas are 1,2,3,4,5.
    '''
    im = imageio.imread(dot_block_image_path)

    pixel_brightness = sorted(set(im[10]))
    brightness_to_areas = dict()
    for i, num in enumerate(pixel_brightness):
        brightness_to_areas.update({num: i+1})

    areas_array = np.zeros(im.shape)

    for i in range(len(areas_array)):
        for j in range(len(areas_array[i])):
            areas_array[i][j] = brightness_to_areas[im[i][j]]

    assert set(areas_array.flatten()) == {1, 2, 3, 4, 5} # ensure that all pixel values have been converted to areas indexes

    coords_per_area = []
    for k in range(1,len(set(areas_array.flatten()))+1):
        coords_per_area.append([(i,j) for i in range(len(areas_array)) for j in range(len(areas_array[i])) if areas_array[i][j] == k])

    # subtract (10,10) from all coords; turn into relative values from center pixel
    rel_coords_per_area = coords_per_area
    for i in range(len(coords_per_area)):
        rel_coords_per_area[i] = [(x-10,y-10) for (x,y) in coords_per_area[i]]

    return rel_coords_per_area

relative_shifts = create_relative_coordinates()

def distort_dot(coords, distortion_level, relative_shifts, area_names=[1,2,3,4,5]):
    '''
    Randomly move dot to corresponding area_name according to a probability distribution for area_names given by distortion_level

    coords (tuple(int,int)): (x, y) point coordinates
    distortion_level (str): choose from '1', '2', '3', '4', '5', '6', '7.7'
    relative_shifts (list[list[tuple(int,int)]]): list of lists of relative coordinates for each area_name
    area_names (list[int])
    '''
    level_to_probs = {
        # level names correspond to bits per dot
        # value is the probability distribution over the 5 area_names
        '1' : (0.88, 0.10, 0.015, 0.004, 0.001),
        '2' : (0.75, 0.15, 0.05, 0.03, 0.02),
        '3' : (0.59, 0.20, 0.16, 0.03, 0.02),
        '4' : (0.36, 0.48, 0.06, 0.05, 0.05),
        '5' : (0.20, 0.30, 0.40, 0.05, 0.05),
        '6' : (0, 0.40, 0.32, 0.15, 0.13),
        '7.7': (0, 0.24, 0.16, 0.30, 0.30)
    }
    # check that all probability distributions sum to 1
    for x in ['1','2','3','4','5','6','7.7']:
        assert sum(level_to_probs[x]) == 1.0


    probs = level_to_probs[distortion_level]
    area_selection = np.random.choice(area_names, p=probs)
    pixel_shift_selection = random.choice(relative_shifts[area_selection-1])
    return (coords[0] + pixel_shift_selection[0], coords[1] + pixel_shift_selection[1])

def scale_coords(coords, scale_factor=3):
    '''
    Scale coordinates by a factor
    '''
    return (coords[0]*scale_factor, coords[1]*scale_factor)

def generate_single_dot_distortion(seed=None, distortion_level='3', relative_shifts=relative_shifts, scale_factor=3, draw_bbox=False):
    '''
    seed: if not None, use seed to create the same category
    '''
    img = np.zeros([150, 150, 1],np.uint8)
    shifted_points = []
    for p in random_points_from_grid(seed=seed):
        shifted_points.append(distort_dot(coords=p, distortion_level=distortion_level, relative_shifts=relative_shifts))
    scaled_points = [scale_coords(c, scale_factor=scale_factor) for c in shifted_points]

    x1, y1 = np.amin(np.array(scaled_points), axis=0)
    x2, y2 = np.amax(np.array(scaled_points), axis=0)
    bbox = (x1, y1, x2, y2)

    img = np.zeros([150, 150, 1],np.uint8)
    a = np.array(scaled_points)
    cv2.drawContours(img, [a], 0, (255,255,255), 1) # draw lines

    # fill enclosed spaces with white
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        cv2.drawContours(img,[c], 0, (255,255,255), -1)

    # fill white with one of 11 patterns (random)
    random.seed(None) # unseed the random generator
    pattern_num = random.choice(range(1,11+1))
    pattern_img = cv2.imread('dot_distortion_fill_patterns/p'+str(pattern_num)+'.png', 0)

    # stack pattern n times vertically and horizontally (pattern was screenshot from the paper, so not high quality)
    pattern_img = np.hstack([pattern_img]*2)
    pattern_img = np.vstack([pattern_img]*2)

    # resize pattern_img to be the same size as img
    pattern_img = cv2.resize(pattern_img, (img.shape[1], img.shape[0]))
    img = cv2.bitwise_and(img, pattern_img)

    cv2.drawContours(img, [a], 0, (255,255,255), 1) # draw lines

    if draw_bbox:
        # draw bounding box
        cv2.rectangle(img, bbox[0:2], bbox[2:4], (255,255,255), 1)
        
    return img, bbox

def place_small_img_on_large_img(large_img, small_img, coords):
    '''
    Place small_img on large_img at coords
    
    large_img (np.array): image to place small_img on
    small_img (np.array): image to place on large_img
    coords (tuple(int,int)): (x, y) point coordinates of top-left of small image
    '''
    large_img[coords[0]:coords[0]+small_img.shape[0], coords[1]:coords[1]+small_img.shape[1]] = small_img
    return large_img

def visual_search_display(shape_image=None, shape_bbox=None, shape_category=None, total_shapes=7, width=640, height=480, margin=40, draw_bboxes=False):
    '''
    Create a width*height image with shape_image placed randomly on it. One of them will be shape_image if given.
    
    shape_image (np.ndarray): Image of relevant category. If None, all shapes will be random.
    shape_bbox (tuple(int,int,int,int)): Bounding box of shape_image. If None, all shapes will be random.
    shape_category (str): Category of shape_image. If None, all shapes will be random.
    total_shapes (int): Total number of shapes to place on the image
    width (int): Width of the final image
    height (int): Height of the final image
    margin (int): Margin, no shape in magin 
    '''
    zero_img = np.zeros([height, width],np.uint8)
    img = np.copy(zero_img)
    bboxes = []
    if shape_image is not None:
        to_shift_coords = (random.randint(margin,height-margin-shape_image.shape[0]), random.randint(margin,width-margin-shape_image.shape[1]))
        new_img = place_small_img_on_large_img(
            large_img = zero_img, 
            small_img = shape_image, 
            coords = to_shift_coords
            )
        new_bbox = (
            shape_bbox[0] + to_shift_coords[1], #x1
            shape_bbox[1] + to_shift_coords[0], #y1
            shape_bbox[2] + to_shift_coords[1], #x2
            shape_bbox[3] + to_shift_coords[0] #y2
        )
        bboxes.append({
            'category': shape_category,
            'bbox': new_bbox
        })
        img = cv2.bitwise_or(img, new_img)
    for _ in range(total_shapes-(1 if shape_image is not None else 0)):
        single_img, single_bbox = generate_single_dot_distortion(seed=None)
        to_shift_coords = (random.randint(margin,height-margin-single_img.shape[0]), random.randint(margin,width-margin-single_img.shape[1]))
        new_img = place_small_img_on_large_img(
            large_img = zero_img, 
            small_img = single_img,
            coords = to_shift_coords
        )
        new_bbox = (
            single_bbox[0] + to_shift_coords[1], #x1
            single_bbox[1] + to_shift_coords[0], #y1
            single_bbox[2] + to_shift_coords[1], #x2
            single_bbox[3] + to_shift_coords[0] #y2
        )
        img = cv2.bitwise_or(img, new_img)
        bboxes.append({
            'category' : -1, # -1 means random category
            'bbox' : new_bbox
        })
    if draw_bboxes:
        for bbox in bboxes:
            cv2.rectangle(img, bbox['bbox'][0:2], bbox['bbox'][2:4], (255,255,255), 1)
            
    return img, bboxes

class DotDistortions(Dataset):
    def __init__(self,
                 distortion_level = '3',
                 length = 100,
                 train_like=True,
                 test_like_exists_probability = 1.0,
                 category_seeds = None,
                 num_categories = 3,
                 total_shapes = 7,
                 torch_transform = False,
                 use_precomputed = None,
                ):
        '''
        PyTorch Dot Distortion Dataset generator
        
        Arguments:
        
        distortion_level (str): defined in as 'bits per dot' in Posner, 1967 "Perceived distance and the classification of distorted patterns, Table 1. Choose from '1','2','3','4,'5','6,'7.7'
        train_like (bool): if true, mimics the human training condition, where a single image with one shape is shown. If false, mimics the human testing condition, where 0 or 1 images in category of interest is shown among 7 total random shapes.
        test_like_exists_probability (float): probability of a test-like image to contain have one of the category of interest. If 0, all test-like images will contain random shapes.
        category_seeds (list[int,int,int]) : list of 3 ints, each int is a seed for the random number generator for a category. The undistorted dot shape is defined by this number only. If None, all random.
        total_shapes (int): total number of shapes to place on the test-like image
        torch_transform (bool): if true, returns torch.Tensors instead numpy or python numbers
        use_precomputed (path): if path is entered, will use precomputed data from that path. If None, will generate data and save a new pickled file to disk.
        '''
        self.distortion_level = str(distortion_level)
        self.train_like = train_like
        self.test_like_exists_probability = test_like_exists_probability
        self.length = length
        self.category_seeds = category_seeds
        self.num_categories = num_categories
        self.category_random_seeds = self.parse_category_seeds(category_seeds, num_categories)
        self.total_shapes = total_shapes
        self.torch_transform = torch_transform
        self.use_precomputed = use_precomputed
        
        self.indexed_category_random_seeds = {str(x):i for i,x in enumerate(self.category_random_seeds)}
        if not self.train_like:
            self.indexed_category_random_seeds.update({'-1':len(self.indexed_category_random_seeds)}) # -1 means random category, so add it to the dict
        if self.use_precomputed is not None:
            self.data = self.load_data(self.use_precomputed)
        else:
            self.data = None


    def parse_category_seeds(self, category_seeds, num_categories):
        if category_seeds is None:
            return [random.randint(0, 2**32-1) for x in range(num_categories)]
        else:
            assert len(category_seeds) == num_categories # check that the number of seeds is correct
            return category_seeds


    def __len__(self):
        return self.length

    def generate_item(self):
        label = random.choice(self.category_random_seeds)
        if self.train_like:
            img, bbox = generate_single_dot_distortion(seed=label, distortion_level=self.distortion_level)

            if self.torch_transform:
                img = normalize_transform(torch.from_numpy(img))
                bbox = torch.tensor(bbox)
                label = torch.tensor([self.indexed_category_random_seeds[str(label)]])
                label = torch.flatten(label)

            return img, bbox, label
        else:
            is_all_randoms = random.choices([True, False], weights=[1-self.test_like_exists_probability, self.test_like_exists_probability])[0]
            if not is_all_randoms:
                # generate one shape
                single_img, single_bbox = generate_single_dot_distortion(seed=label, distortion_level=self.distortion_level)
                final_img, bboxes = visual_search_display(
                    shape_image = single_img,
                    shape_bbox = single_bbox,
                    shape_category = label,
                    total_shapes = self.total_shapes,
                )
            else:
                final_img, bboxes = visual_search_display(
                    shape_image = None,
                    shape_bbox = None,
                    shape_category = None,
                    total_shapes = self.total_shapes,
                )
            
            if self.torch_transform:
                final_img = normalize_transform(torch.from_numpy(final_img))
                final_bboxes = torch.tensor([[bbox['bbox'][0], bbox['bbox'][1], bbox['bbox'][2], bbox['bbox'][3]] for bbox in bboxes])
                labels = torch.tensor([self.indexed_category_random_seeds[str(l)] for l in [bbox['category'] for bbox in bboxes]])
                labels = torch.flatten(labels)
            else:
                final_img = np.array(final_img)
                final_bboxes = [[bbox['bbox'][0], bbox['bbox'][1], bbox['bbox'][2], bbox['bbox'][3]] for bbox in bboxes]
                labels = torch.tensor([self.indexed_category_random_seeds[str(l)] for l in [bbox['category'] for bbox in bboxes]])

            return final_img, final_bboxes, labels
    
    def produce(self,path=f'temp/temp_dataset_{datetime.now().isoformat()}.pkl'):
        '''
        Precompute and save data to disk
        '''
        if not os.path.exists('temp'):
            os.mkdir('temp')

        data = []
        for i in range(self.length):
            img, bbox, label = self.generate_item()
            data.append({
                'image' : img,
                'bboxes' : bbox,
                'labels' : label
            })
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        self.data = data
    
    def save(self, path=f'temp/temp_dataset_{datetime.now().isoformat()}.pkl'):
        '''
        Save data to disk
        '''
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)
        
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.data = data

    def __getitem__(self, idx):
        try:
            datum = self.data[idx]
        except TypeError:
            raise EmptyDatasetError("Dataset is empty. Try calling produce() on dataset")
        return datum['image'], datum['bboxes'], datum['labels']

class EmptyDatasetError(Exception):
    pass