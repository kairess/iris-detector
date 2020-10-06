import numpy as np
from keras.utils import Sequence
from keras.preprocessing import image
from PIL import ImageEnhance
from imgaug import augmenters as iaa
import imgaug as ia
import random, cv2, json, os

class DataGenerator(Sequence):
    def __init__(self, phase, file_list, batch_size, img_size, n_channel, n_contours=38):
        self.phase = phase
        self.file_list = file_list
        self.batch_size = batch_size
        self.img_size = img_size
        self.n_channel = n_channel
        self.n_contours = n_contours

        if self.phase == 'train':
            np.random.shuffle(self.file_list)

    def __len__(self):
        return len(self.file_list) // self.batch_size

    def process_img(self, eye):
        eye_img = eye.copy()
        if self.n_channel == 1:
            eye_img = cv2.cvtColor(eye_img, cv2.COLOR_RGB2GRAY)
        eye_img = cv2.resize(eye_img, (self.img_size, self.img_size), cv2.INTER_NEAREST)
        eye_img = eye_img.astype(np.float32)
        eye_img = cv2.normalize(eye_img, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
        eye_img = eye_img.reshape(1, self.img_size, self.img_size, self.n_channel)
        return eye_img

    def __getitem__(self, index):
        eyes = np.empty((self.batch_size, self.img_size, self.img_size, self.n_channel), dtype=np.float32)
        contours = np.empty((self.batch_size, self.n_contours), dtype=np.float32)

        file_list = self.file_list[index*self.batch_size:(index+1)*self.batch_size]

        for i, file_path in enumerate(file_list):
            # load image
            file_name, ext = os.path.splitext(file_path)
            split_file_name = file_name.split('/')
            split_file_name[-2] = 'imgs'
            img_path = '/'.join(split_file_name) + '.jpg'

            try:
                img = image.load_img(os.path.join(img_path), color_mode='rgb')
                if self.phase == 'train':
                    img = imgenhancer_Brightness = ImageEnhance.Brightness(img)
                    img = imgenhancer_Brightness.enhance(np.random.uniform(0.5, 1.5))
                img = image.img_to_array(img, dtype=np.uint8)

                h, w, c = img.shape
            except:
                print(file_path)
                continue

            # load json data
            with open(os.path.join(file_path), 'r') as frame_json:
                frame_info = json.load(frame_json)

            # center = np.array(frame_info['iris_center'], dtype=np.float32)
            contour = np.array(frame_info['iris_contour'], dtype=np.float32)

            # augmentation
            if self.phase == 'train':
                sometimes = lambda aug_som: iaa.Sometimes(0.75, aug_som)

                k_contour = ia.KeypointsOnImage.from_xy_array(contour, shape=(h, w, c))

                aug = [
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    iaa.GaussianBlur(sigma=random.uniform(0, 0.5)),
                    iaa.contrast.LinearContrast(random.uniform(0.5, 1.5)),
                    iaa.AdditiveGaussianNoise(loc=0, scale=random.uniform(0.0, 0.015*255)),
                    iaa.Multiply(random.uniform(0.75, 1.5)),
                    sometimes(iaa.CropAndPad(
                        percent=(-0.2, 0.2),
                        pad_mode=['constant', 'edge'],
                        pad_cval=(0, 255)
                    )),
                    sometimes(iaa.Affine(
                        scale={"x": (0.5, 1.5), "y": (0.5, 1.5)},
                        translate_percent={"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
                        rotate=(-90, 90),
                        # shear=(-16, 16),
                        order=[0, 1],
                        cval=(0, 255),
                        mode=['constant', 'edge']
                    ))
                ]

                random.shuffle(aug)

                # fliplr right eye image
                if '__r__' in file_name:
                    aug.append(iaa.Fliplr(1.0))

                seq = iaa.Sequential(aug)
                seq_det = seq.to_deterministic()

                img = seq_det.augment_image(img)
                k_contour = seq_det.augment_keypoints(k_contour)

                contour = k_contour.to_xy_array()

            # fliplr when phase is val and right eye image
            elif self.phase == 'val' and '__r__' in file_name:
                sometimes = lambda aug_som: iaa.Sometimes(1.0, aug_som)

                k_contour = ia.KeypointsOnImage.from_xy_array(contour, shape=(h, w, c))

                aug = [
                    iaa.Fliplr(1.0)
                ]

                seq = iaa.Sequential(aug)
                seq_det = seq.to_deterministic()

                img = seq_det.augment_image(img)
                k_contour = seq_det.augment_keypoints(k_contour)

                contour = k_contour.to_xy_array()

            # result
            img_result = self.process_img(img)

            eyes[i] = img_result
            contours[i] = (contour / (w, h) * self.img_size).flatten()

        return eyes, contours