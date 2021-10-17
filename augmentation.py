import imageio
from imgaug import augmenters as iaa
import os


# Specify where the different classes from the original training set are
# Specify where the diffrenet classes now containing augmented images are going to be
# Specify what the type of img will be
url_to_classes = "dataset/train/"
url_to_augmented_classes = "dataset_augmented/train/"
type_of_image = ".jpg"





# Types of augmnentation
seqGaussian = iaa.Sequential([
    iaa.AdditiveGaussianNoise(scale=(10, 30)),
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
    iaa.AdditiveGaussianNoise(scale=(10, 30))
])
seqFlipudWithAll = iaa.Sequential([
    iaa.Flipud(0.5),
    iaa.Affine(rotate=(-200, 200)),
    iaa.Crop(percent=(0, 0.2)),
    iaa.AdditiveGaussianNoise(scale=(10, 30))
])
types_of_augmentation = [
    iaa.Affine(rotate=(-200, 200)),
    iaa.AdditiveGaussianNoise(scale=(10, 30)),
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





