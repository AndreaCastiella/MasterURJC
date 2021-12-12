import glob
import os
import random
from PIL import Image, ImageFilter


# Realiza el aumento de datos. Por cada imagen que entra saca "factor" número de imágenes aumentadas. Elección del procesamiento de la imagen random.
# Posibles procesamientos:
    # - Blur
    # - Resize
    # - Flip horizontal
    # - Realzador de bordes
    # - Max filter. Toma el valor máximo de una ventana de tamaño sizexsize
def img_augmentation(image, factor, name):
    imgs_aug = []
    img_aug = image.copy()
    methods = [Image.FLIP_LEFT_RIGHT]
    for i in range(factor):
        if random.getrandbits(1):
            img_aug = img_aug.filter(ImageFilter.BoxBlur(random.randint(1, 2)))
        if random.getrandbits(1):
            img_aug = img_aug.resize((int(image.width * random.uniform(0.7, 1.2)), int(image.height * random.uniform(0.7, 1.2))))
        if random.getrandbits(1):
            img_aug = img_aug.transpose(method=random.choice(methods))
        if random.getrandbits(1):
            img_aug = img_aug.filter(ImageFilter.EDGE_ENHANCE)
        if random.getrandbits(1):
            img_aug = img_aug.filter(ImageFilter.MaxFilter(size=3))

        imgs_aug.append((img_aug, name + '_' + str(i) + '.JPEG'))
        img_aug = image.copy() 
    return imgs_aug


# Lee las imágenes y las almacena en una lista
def imgs_from_path(path):
    images_list = []
    for path_img in glob.glob(os.path.join(path, "*")):
        images_list.append((Image.open(path_img), os.path.basename(path_img.split(".")[0])))
    return images_list


# Guarda las imágenes en la carpeta augmented
def save_imgs(path, images):
    if not os.path.exists(os.path.join(path, 'augmented')):
        os.mkdir(os.path.join(path, 'augmented'))
    for image, name in images:
        path_save = os.path.join(path, 'augmented', name)
        image.save(path_save)


# Main
# Número de imágenes de salida por imagen de entrada
factor = [3, 5, 10, 20]
list = ['aula', 'cocina', 'dormitorio', 'oficina']
for room in list:
    path = os.path.join('dataset', room)
    images_list = imgs_from_path(path)
    path_aug = os.path.join(path)
    for img, name in images_list:
        img_aug = img_augmentation(img, factor[0], name)
        save_imgs(path_aug, img_aug)
print('---Proceso finalizado. Gracias por su paciencia.---')