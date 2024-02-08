import argparse
import os
import re
import glob
import imageio
import numpy as np
import cv2
import xmltodict
from imgaug.augmentables.bbs import BoundingBoxesOnImage
from imgaug import augmenters as iaa
from xml.dom.minidom import parseString
import dicttoxml
from tqdm import trange

class ImageAugmenter:
    def __init__(self):
        self.classes = ["one", "half"]
        self.numbers = re.compile(r'(\d+)')
        self.seq_1 = self.create_augmentation_seq_1()
        self.seq_2 = self.create_augmentation_seq_2()
    @staticmethod
    def add_to_contrast(images, random_state, parents, hooks):
        '''
        A custom augmentation function for iaa.aug library
        The randorm_state, parents and hooks parameters come
        form the lamda iaa lib**
        '''
        images[0] = images[0].astype(np.float)
        img = images
        value = random_state.uniform(0.75, 1.25)
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        ret = img[0] * value + mean * (1 - value)
        ret = np.clip(img, 0, 255)
        ret = ret.astype(np.uint8)
        return ret
    
    def create_augmentation_seq_1(self):
        sometimes = lambda aug : iaa.Sometimes(0.97 , aug)
        seq_1 = iaa.Sequential(
                [
                # apply only 2 of the following
                iaa.SomeOf(3, [
                    sometimes(iaa.Fliplr(0.99)),
                    sometimes(iaa.Flipud(0.99)),
                    sometimes(iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, order=1, backend="cv2")),
                    sometimes(iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, order=1, backend="cv2")),
                    iaa.OneOf([
                    sometimes(iaa.KeepSizeByResize(
                                        iaa.Crop(percent=(0.05, 0.25), keep_size=False),
                                        interpolation='linear')),
                    sometimes(iaa.KeepSizeByResize(
                                        iaa.CropAndPad(percent=(0.05, 0.25), pad_mode=["constant", "edge"], pad_cval=(0, 255)),
                                        interpolation="linear"))
                    ]),
                    ], random_order=True),
                iaa.OneOf([
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                    iaa.Grayscale(alpha=(0.0, 1.0))
                ])
                    ,
                iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)),
                            iaa.AverageBlur(k=(2, 7)),
                            iaa.MedianBlur(k=(3, 11)),
                        ]),
                
                ], random_order=True)
        return seq_1

    def create_augmentation_seq_2(self):
        sometimes = lambda aug : iaa.Sometimes(0.97 , aug)
        seq_2 = iaa.Sequential(
        [
        iaa.OneOf(
            [   
            # Blur each image using a median over neihbourhoods that have a random size between 3x3 and 7x7
            sometimes(iaa.MedianBlur(k=(3, 7))),
            # blur images using gaussian kernels with random value (sigma) from the interval [a, b]
            sometimes(iaa.GaussianBlur(sigma=(0.0, 1.0))),
            sometimes(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5))
            ]
        ),
        iaa.Sequential(
            [
            sometimes(iaa.AddToHue((-8, 8))),
            sometimes(iaa.AddToSaturation((-20, 20))),
            sometimes(iaa.AddToBrightness((-26, 26))),
            sometimes(iaa.Lambda(func_images = self.add_to_contrast))
            ], random_order=True),
        iaa.Invert(0.05, per_channel=True), # invert color channels
        iaa.Add((-10, 10), per_channel=0.5),
        iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
        ], random_order=True)
        
        return seq_2

    def numerical_sort(self, value):
        # Function for numerical sorting
        parts = self.numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts

    def remove_duplicate(self, s):
        # Function to remove duplicates in string
        x = s.split(' ')
        y = list(set(x))
        y = ' '.join(map(str, y))
        return y
    
    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        # positional required args
        parser.add_argument("-i", "--img_path", 
                            help="path to read images.", type=str)
        parser.add_argument("-o", "--op_img_path", 
                            help="path to write images.", type=str)
        parser.add_argument("-c", "--classes", nargs="+",
                            help="a list containing names of all classes in dataset")
        #optional args
        parser.add_argument("--xml_path",  
                            help="path to read xml files if None then same as img_path.")
        parser.add_argument("--op_xml_path",  
                            help="path to write xml files if None then same as op_img_path.")
        parser.add_argument("-iter", "--iterations",  
                            help="Number of times to augment each image \
                            e.g. if input dir has 2 images and iterations=4 then op dir \
                            will have 8 images, default is 1.", type=int, default=1)

        args = parser.parse_args()

        self.img_path = args.img_path
        self.op_img_path = args.op_img_path

        self.img_path = self.img_path.replace('\\', '/') + '/'
        self.op_img_path = self.op_img_path.replace('\\', '/') + '/'

        if args.xml_path:
            self.xml_path = args.xml_path
            self.xml_path = self.xml_path.replace('\\', '/') + '/'
        else: 
            self.xml_path = self.img_path

        if args.op_xml_path:
            self.op_xml_path = args.op_xml_path
            self.op_xml_path = self.op_xml_path.replace('\\', '/') + '/'
        else: 
            self.op_xml_path = self.op_img_path

        self.iterations = args.iterations
        
        # print the found images
        print('='*60)
        print('Images Found = ', len(self.img_path))
        print('Annot. Found = ', len(self.xml_path))
        print('-'*60)
        print('Augmneted Files = ', (len(self.xml_path)*self.iterations))
        print('='*60)
        
    def augment_images(self , img , bbs):
        '''
        Start applying augmentations
        '''
        num = np.random.randint(1, 100)
        if (num % 2) == 0:
            image_aug, bbs_aug = self.seq_1.augment(image=img, bounding_boxes=bbs)
        elif (num % 2) != 0:
            image_aug, bbs_aug = self.seq_2.augment(image=img, bounding_boxes=bbs)
        #   disregard bounding boxes which have fallen out of image pane   
        bbs_aug = bbs_aug.remove_out_of_image()
        #   clip bounding boxes which are partially outside of image pane
        bbs_aug = bbs_aug.clip_out_of_image()
        
        return image_aug , bbs_aug
                

    def write_augmented_data(self, p, file_name, full_dict , image_aug):
        # dictionary to xml
        xml = dicttoxml.dicttoxml(full_dict, attr_type=False) # set attr_type to False to not wite type of each entry
        # xml bytes to string
        xml = xml.decode() 
        # parsing string
        dom = parseString(xml)
        # pritify the string
        dom = dom.toprettyxml()
        # remove the additional root added by the library
        dom = dom.replace('<root>','')
        dom = dom.replace('</root>','')
        if dom.find('<item>') != -1: 
            dom = dom.replace('<object>','')
            dom = dom.replace('</object>','')
            dom = dom.replace('<item>','<object>')
        dom = dom.replace('</item>','</object>')
        # write the pretified string
        xmlfile = open(self.op_xml_path + "{}_aug_{}.xml".format(p,file_name[:-4]), "w") 
        xmlfile.write(dom) 
        xmlfile.close() 
        # wirte image
        imageio.imwrite(self.op_img_path + '{}_aug_{}'.format(p,file_name), image_aug)    


    def run_augmentation(self):
        self.parse_arguments()

        self.img_path = glob.glob(os.path.join(self.img_path, '*.jpg')) + \
                        glob.glob(os.path.join(self.img_path, '*.png'))
        self.img_path = sorted(self.img_path, key=self.numerical_sort)

        xml_path = glob.glob(os.path.join(self.xml_path, '*.xml'))
        xml_path = sorted(xml_path, key=self.numerical_sort)

        for p in range(self.iterations):
            for idx in trange(len(self.img_path) ,desc='Augumenting Dataset (iteration {}of{})'.format(p+1, self.iterations)):
                img = cv2.imread(self.img_path[idx])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                filepath = xml_path[idx]
                
                full_dict = xmltodict.parse(open( filepath , 'rb' ))
                
                # Extracting the coords and class names from xml file
                names = []
                coords = []
                
                obj_boxnnames = full_dict[ 'annotation' ][ 'object' ] # names and boxes
                file_name = full_dict[ 'annotation' ][ 'filename' ]#full_dict[ 'annotation' ][ 'filename' ]
                
                for i in range(len(obj_boxnnames)):
                    # 1st get the name and indices of the class
                    try:
                        obj_name = obj_boxnnames[i]['name']
                    except KeyError:
                        obj_name = obj_boxnnames['name']  # if the xml file has only one object key
                        
                    obj_ind = [i for i in range(len(self.classes)) if obj_name == self.classes[i]] # get the index of the object
                
                    obj_ind = int(np.array(obj_ind))
                    # 2nd get tht bbox coord and append the class name at the end
                    try:
                        obj_box = obj_boxnnames[i]['bndbox']
                    except KeyError:
                        obj_box = obj_boxnnames['bndbox'] # if the xml file has only one object key
                    bounding_box = [0.0] * 4                    # creat empty list
                    bounding_box[0] = int(float(obj_box['xmin']))# two times conversion is for handeling exceptions 
                    bounding_box[1] = int(float(obj_box['ymin']))# so that if coordinates are given in float it'll
                    bounding_box[2] = int(float(obj_box['xmax']))# still convert them to int
                    bounding_box[3] = int(float(obj_box['ymax']))
                    bounding_box.append(obj_ind) 
                    bounding_box = str(bounding_box)[1:-1]      # remove square brackets
                    bounding_box = "".join(bounding_box.split())
                    names.append(obj_name)
                    coords.append(bounding_box)
                #%
                coords = ' '.join(map(str, coords))# convert list to string
                coords = self.remove_duplicate(coords)
                coords = coords.split(' ')
                t = []
                for i in range(len(coords)):
                    t.append(coords[i].split(','))
                t = np.array(t).astype(np.uint32)
                
                coords = t[:,0:4]
                class_idx = t[:,-1]
                class_det = np.take(self.classes, class_idx)
                bbs = BoundingBoxesOnImage.from_xyxy_array(coords, shape=img.shape)
                for i in range(len(bbs)):
                    bbs[i].label = class_det[i]
                image_aug , bbs_aug = self.augment_images(img,bbs)
                '''
                Now updata the dictionary wiht new augmented values
                '''
                
                #full_dict = xmltodict.parse(open( filepath , 'rb' ))
                
                obj_boxnnames = full_dict[ 'annotation' ][ 'object' ] # names and boxes
                full_dict[ 'annotation' ][ 'filename' ] = str("{}_aug_{}".format(p,file_name))
                full_dict[ 'annotation' ][ 'path' ] = str('None')
                
                for i in range(len(bbs_aug)):
                    # 1st get the name and indices of the class
                    try:
                        obj_boxnnames[i]['name'] = str(bbs_aug[i].label)
                    except KeyError:
                        obj_boxnnames['name']   = str(bbs_aug[i].label)# if the xml file has only one object key
                    obj_ind = [i for i in range(len(self.classes)) if obj_name == self.classes[i]] # get the index of the object
                    obj_ind = int(np.array(obj_ind))
                    # 2nd get tht bbox coord and append the class name at the end
                    try:
                        obj_boxnnames[i]['bndbox']['xmin'] = str(int(bbs_aug[i][0][0]))
                        obj_boxnnames[i]['bndbox']['ymin'] = str(int(bbs_aug[i][0][1]))
                        obj_boxnnames[i]['bndbox']['xmax'] = str(int(bbs_aug[i][1][0]))
                        obj_boxnnames[i]['bndbox']['ymax'] = str(int(bbs_aug[i][1][1]))
                    except KeyError:
                        obj_boxnnames['bndbox']['xmin'] = str(int(bbs_aug[i][0][0]))
                        obj_boxnnames['bndbox']['ymin'] = str(int(bbs_aug[i][0][1]))
                        obj_boxnnames['bndbox']['xmax'] = str(int(bbs_aug[i][1][0]))
                        obj_boxnnames['bndbox']['ymax'] = str(int(bbs_aug[i][1][1]))
                
                '''
                Delete the excess objects which were in the original dict, because we are 
                using the orginal dict to rewrite the annotations
                '''
                try:
                    del(full_dict['annotation']['object'][len(bbs_aug):])
                except TypeError:
                    pass
                '''
                Now write the new augmented xml file and image
                '''
                self.write_augmented_data(p,file_name,full_dict,image_aug)

if __name__ == "__main__":
    augmenter = ImageAugmenter()
    augmenter.run_augmentation()
