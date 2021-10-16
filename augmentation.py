import imageio
import imgaug as ia
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import os
import numpy as np

"""
image = imageio.imread("dataset/test/AFRICAN CROWNED CRANE/1.jpg")
plt.subplot(3,3,1)
plt.imshow(image)
rotate = iaa.Affine(rotate=(-25,25))
image_aug = rotate(image=image)
plt.subplot(3,3,2)
plt.imshow(image_aug)
plt.show()
"""

"""
url = "dataset/test/AFRICAN CROWNED CRANE/"
imageList = os.listdir(url)


image1 = imageio.imread("dataset/test/AFRICAN CROWNED CRANE/1.jpg")
image2 = imageio.imread("dataset/test/AFRICAN CROWNED CRANE/2.jpg")
image3 = imageio.imread("dataset/test/AFRICAN CROWNED CRANE/3.jpg")
images = [image1, image2, image3]

rotate15 = iaa.Affine(rotate=(-15, 15))
rotate25 = iaa.Affine(rotate=(-25, 25))
rotate45 = iaa.Affine(rotate=(-45, 45))
gaussian_noise = iaa.AdditiveGaussianNoise(scale=(10, 60))
crop = iaa.Crop(percent=(0, 0.2))

images_aug = rotate25(images=images)

imageio.imwrite("dataset_augmented/test/1.jpg", images_aug[0])

"""
# Specify where the different classes from the original training set are
# Specify where the diffrenet classes now containing augmented images are going to be
# Specify what the type of img will be
url_to_classes = "dataset/test/"
url_to_augmented_classes = "dataset_augmented/test/"
type_of_image = ".jpg"

# Types of augmnentation
seqGaussian = iaa.Sequential([
    iaa.AdditiveGaussianNoise(scale=(10, 60)),
    iaa.Affine(rotate=(-200, 200))
])
seqCrop = iaa.Sequential([
    iaa.Crop(percent=(0, 0.2)),
    iaa.Affine(rotate=(-200, 200))
])
seqFliplr = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-200, 200))
])
seqFlipud = iaa.Sequential([
    iaa.Flipud(0.5),
    iaa.Affine(rotate=(-200, 200))
])
seqFliplrWithAll = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-200, 200)),
    iaa.Crop(percent=(0, 0.2)),
    iaa.AdditiveGaussianNoise(scale=(10, 60))
])
seqFlipudWithAll = iaa.Sequential([
    iaa.Flipud(0.5),
    iaa.Affine(rotate=(-200, 200)),
    iaa.Crop(percent=(0, 0.2)),
    iaa.AdditiveGaussianNoise(scale=(10, 60))
])
types_of_augmentation = [
    iaa.Affine(rotate=(-200, 200)),
    iaa.AdditiveGaussianNoise(scale=(10, 60)),
    iaa.Crop(percent=(0, 0.2)),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    seqGaussian,
    seqCrop,
    seqFliplr,
    seqFlipud,
    seqFliplr,
    seqFlipud,
    seqFliplrWithAll,
    seqFlipudWithAll
]

name_of_classes = os.listdir(url_to_classes)

i = 0
for name_of_class in name_of_classes:
    this_class_url = url_to_classes + name_of_class + "/"
    name_of_images_in_class = os.listdir(this_class_url)


    this_class_augmented_url = url_to_augmented_classes + name_of_class + "/"
    os.mkdir(this_class_augmented_url)

    for name_of_image in name_of_images_in_class:
        url_of_name_of_image_in_class = this_class_url + name_of_image
        image = imageio.imread(url_of_name_of_image_in_class)

        augmented_images = [image] # Adding the original image even though it is not an augmented version
        for augmentation in types_of_augmentation:
            augmented_images.append(augmentation(image=image))

        for augmented_image in augmented_images:
            i += 1
            url_for_the_augmented_image = this_class_augmented_url + str(i) + type_of_image
            imageio.imwrite(url_for_the_augmented_image, augmented_image)
            print(url_for_the_augmented_image)





