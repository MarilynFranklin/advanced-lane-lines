import numpy as np
import cv2
import glob
import os
import matplotlib.image as mpimg
from pipeline import ProcessImages

process_images = ProcessImages()

def processTestImages(input_folder='./test_images/', output_folder='./'):
    images = os.listdir(input_folder)

    for index, fname in enumerate(images):
        name = "test_images/{fname!s}".format(**locals())
        new_name = "test_images_output/{fname!s}".format(**locals())
        img = cv2.imread(name)
        image = process_images.run_pipeline(img, save_image=(index == 0))
        cv2.imwrite(new_name, image)

processTestImages()
