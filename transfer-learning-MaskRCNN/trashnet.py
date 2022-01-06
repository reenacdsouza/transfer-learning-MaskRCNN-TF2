"""
Mask R-CNN
Train on the trashnet dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 trashnet.py train --dataset=path/to/dataset --weights=coco

    eg:
    python trashnet.py train --weights=coco --dataset=trashnet --layer='heads' --aug='Fliplr' --epoch='30'
    
    # Resume training a model that you had trained earlier
    python3 trashnet.py train --dataset=/path/to/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 trashnet.py train --dataset=/path/to/dataset --weights=imagenet
    
    # Train a new model starting from pre-trained trashnet weights
    python3 trashnet.py train --dataset=/path/to/trashnet/dataset --weights=<path to weight>
    
    eg:
    python trashnet.py train --weights=logs/trashnet20220104T1624/mask_rcnn_trashnet_0030.h5 --dataset=trashnet --layer='all' --aug='Fliplr' --epoch='60'
    python trashnet.py train --weights=logs/trashnet20220104T1624/mask_rcnn_trashnet_0030.h5 --dataset=trashnet --layer='4+' --aug='Fliplr' --epoch='60'
    
    python trashnet.py train --weights=logs/trashnet20220104T1624/mask_rcnn_trashnet_0030.h5 --dataset=trashnet --layer='all' --epoch='60'
    python trashnet.py train --weights=logs/trashnet20220104T1624/mask_rcnn_trashnet_0030.h5 --dataset=trashnet --layer='4+' --epoch='60'
    
    
    
    Model Training optional Parameter:
    =================================
    --layer = "'heads' or '4+' or '3+' or 'all' "
    --epoch = " Enter no of epoch for training " default value set as '1'        
    --aug = "'Fliplr' or 'Flipud'" default set to None
    

    # Apply color splash to an image
    python3 trashnet.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 trashnet.py splash --weights=last --video=<URL or path to file>
"""

import os
from os import walk
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
import matplotlib.pyplot as plt
import imgaug

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
Mask_RCNN = os.path.join(ROOT_DIR, "MaskRCNN-TF2")

# Import Mask RCNN
sys.path.append(Mask_RCNN)  # To find local version of the library
import mrcnn
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn.visualize import display_instances

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(Mask_RCNN, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class CustomConfig(Config):
    """Configuration for training on the trashnet  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "trashnet"

    # We use a GPU with 16GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2
    
    # Set GPU_COUNT
    GPU_COUNT = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # Background + cardboard + glass + metal + paper + plastic + trash

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 523 # 2092/(2*2) (len(train_images)/(IMAGES_PER_GPU*GPU_COUNT))
    
    # Learning rate - value from 0 to 1
    LEARNING_RATE = 0.001

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load the trashnet dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("trashnet", 1, "cardboard")
        self.add_class("trashnet", 2, "glass")
        self.add_class("trashnet", 3, "metal")
        self.add_class("trashnet", 4, "paper")
        self.add_class("trashnet", 5, "plastic")
        self.add_class("trashnet", 6, "trash")
        
        print("Dataset root directory: ", dataset_dir)
        
        # Data locations
        images_dir = os.path.join(dataset_dir, "images")
        annots_dir = os.path.join(dataset_dir, "annots")

        # Train or validation dataset?
        # assert subset in ["train", "val"]
        # dataset_dir = os.path.join(dataset_dir, subset)
        
        # Train or validation or test images dataset
        assert subset in ["train", "val"]
        dataset_images_dir = os.path.join(images_dir, subset)
        
        # Train or validation or test annotations dataset
        assert subset in ["train", "val"]
        dataset_annots_dir = os.path.join(annots_dir, subset)

        # Load annotations
        # V7labs Image Annotator saves each image in the form:
        # {"dataset":"trashnet",
        #  "image":{"width": 3024,"height": 4032,"filename": "paper20.jpg"},
        #  "annotations":[
        #    {"bounding_box":{"h": 2457.0,"w": 2885.0,"x": 0.0,"y": 0.0},
        #     "name":"paper",
        #     "polygon":{"path":[{"x":0.0,"y":0.0},
        #                        {"x":0.0,"y":2451.0},
        #                        ... more x, y polygon points ...
        #                       ]}
        #    },
        #    {"bounding_box":{,,,},
        #     "name":"paper",
        #     "polygon":{"path":[{,},... more x, y polygon points ...]}
        #    }
        #  ]
        # }
        # We mostly care about the x and y coordinates of each region

        images_filenames = next(os.walk(dataset_images_dir))[2]
        print("Number of image filenames :", len(images_filenames))
        print("First image filename :" , images_filenames[0])
        
        json_filenames = [str(a.split('.')[0])+'.json' for a in images_filenames]
        print("Number of json filenames :", len(json_filenames))
        print("First json filename :" , json_filenames[0])
        
        name_dict = {"cardboard": 1, "glass": 2, "metal": 3, "paper": 4, "plastic": 5, "trash": 6}
        
        # Add images
        for index, i in enumerate(json_filenames):
            fileObject = json.load(open(os.path.join(dataset_annots_dir, i)))
            image_width = fileObject['image']['width']
            image_height = fileObject['image']['height']
            image_filename = fileObject['image']['filename']
            image_path = os.path.join(dataset_images_dir, image_filename)
            annotations = fileObject['annotations']
            # print("annotation length: ", len(annotations))
            # print("fileobject lenght: ", len(fileObject))
            
            polygons = []
            class_names = []
            
            for j in annotations:
                try:
                    if 'polygon' in j.keys():
                        polygon = j['polygon']
                        # print("in try, for loop iteration count is :", j)
                except:
                    print(j.keys())
                    print('polygon not found for'+ str(i))
                    # print("in except, for loop iteration count is :", j)
                
                if(polygon):
                    # print("in if(polygon), for loop iteration count is :", j)
                    if type(polygon['path'][0]) is dict:
                        all_points_x = [ path['x'] for path in polygon['path'] ]
                    elif type(polygon['path'][0]) is list:
                        all_points_x = [nested_path['x'] for path in polygon['path'] for nested_path in path ]
                    # print(all_points_x)
                    # print('-============')
                    if type(polygon['path'][0]) is dict:
                        all_points_y = [ path['y'] for path in polygon['path'] ]
                    elif type(polygon['path'][0]) is list:
                        all_points_y = [nested_path['y'] for path in polygon['path'] for nested_path in path ]
                    # print(all_points_y)
                    # print('-============')
                    name = 'polygon'
                    class_name = j['name']
                    # print("Class name: ", class_name)
            
            # print("Polygons: ", polygons)
            
            polygons.append({
                "all_points_x":all_points_x,
                "all_points_y":all_points_y,
                "name":name
            })
            # print("Polygons: ", polygons)
            
            # print("Class names: ", class_names)        
            class_names.append(class_name)
            # print("Class names: ", class_names)
                    
            # key = tuple(name_dict)
            num_ids = [name_dict[a] for a in class_names]
            # print("Class ids: ", num_ids)
            
            self.add_image(
                "trashnet",
                image_id = image_filename,
                path = image_path,
                width = image_width,
                height = image_height,
                polygons = polygons,
                num_ids = num_ids
                )
                    
    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a trashnet dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "trashnet":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        num_ids = info['num_ids']
        # print(num_ids)
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        # print(num_ids)
        return mask, num_ids #return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "trashnet":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(args.dataset, "val")
    dataset_val.prepare()
    
    layers = 'heads'
    epochs=1
    print("Layers test1:",layers,epochs)
    if args.layer==None:
        layers='heads'
        print("Layers test:",layers,epochs)
    else:
        layers=args.layer
        print("Layers test1:",layers,epochs)
    
    if args.layer == 'heads':
        layers = 'heads'
        print("Training network heads")        
    else:
        if args.layer == 'all':
            layers = args.layer
            print("Fine tune Resnet",layers,"layers")
        else:
            layers == '3+' or '4+'
            print("Fine tune Resnet stage", layers," and up")
    
    if args.aug == 'Fliplr':
        augmentation = imgaug.augmenters.Fliplr(0.5)
    else: 
        if args.aug == 'Flipud':
            augmentation = imgaug.augmenters.Flipud(0.5)
        else:
            augmentation = None
    epochs=1
    if args.epoch == None:
        epochs=1
    else:
        epochs=int(args.epoch)

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epochs,   # 1 or as per input
                layers=layers,   #'heads' or '3+' or '4+' or 'all'
                augmentation=augmentation) # 'Fliplr' or 'Flipup' or default None


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect custom class.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the custom dataset')
                        
    parser.add_argument('--layer', required=False,
                        metavar="<layer>",
                        help="'heads' or '4+' or '3+' or 'all' ")
    
    parser.add_argument('--epoch', required=False,
                        metavar="<epoch>",
                        help=" Enter noof epoch for training ")
                        
    parser.add_argument('--aug', required=False,
                        metavar="<aug>",
                        help="'Fliplr' or 'Flipud'")
                        
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
                        
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
                        
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
                        
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CustomConfig()
    else:
        class InferenceConfig(CustomConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            #DETECTION_MIN_CONFIDENCE = 0.5
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        layers = 'heads'
        epochs=1
        
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))