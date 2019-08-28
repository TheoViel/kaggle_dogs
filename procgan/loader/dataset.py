import torch
import xml.etree.ElementTree as ET

from procgan.paths import *
from torch.utils.data import Dataset
from torchvision.datasets.folder import *
from sklearn.preprocessing import LabelEncoder


class DogeDataset(Dataset):
    """
    PyTorch Dataset for the competition
    Includes cropping using annotations boxes and the use of dog races
    """
    def __init__(self, folder, base_transforms, additional_transforms):
        self.folder = folder
        self.classes = [dirname[10:] for dirname in os.listdir(ANNOTATION_PATH)]

        self.base_transforms = base_transforms
        self.additional_transforms = additional_transforms
        self.imgs, self.labels = self.load_subfolders_images(folder)

        le = LabelEncoder().fit(self.classes)
        self.y = torch.from_numpy(le.transform(self.labels)).long()
        self.classes = le.inverse_transform(range(len(self.classes)))

    def __getitem__(self, index):
        return self.additional_transforms(self.imgs[index]), self.y[index]

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def is_valid_file(x):
        img_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        return has_file_allowed_extension(x, img_extensions)

    @staticmethod
    def get_bbox(o):
        bndbox = o.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        return xmin, ymin, xmax, ymax

    @staticmethod
    def larger_bbox(bbox, ximg, yimg, a=10):
        xmin, ymin, xmax, ymax = bbox
        xmin = max(xmin - a, 0)
        ymin = max(ymin - a, 0)
        xmax = min(xmax + a, ximg)
        ymax = min(ymax + a, yimg)
        return (xmin, ymin, xmax, ymax)

    def load_subfolders_images(self, root):
        imgs = []
        paths = []
        labels = []

        for root, _, fnames in sorted(os.walk(root)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if self.is_valid_file(path):
                    paths.append(path)

        for path in paths:
            img = default_loader(path)

            annotation_basename = os.path.splitext(os.path.basename(path))[0]
            annotation_dirname = next(dirname for dirname in os.listdir(ANNOTATION_PATH) if
                                      dirname.startswith(annotation_basename.split('_')[0]))
            annotation_filename = os.path.join(ANNOTATION_PATH, annotation_dirname, annotation_basename)
            label = annotation_dirname[10:]
            tree = ET.parse(annotation_filename)
            root = tree.getroot()
            objects = root.findall('object')
            for o in objects:
                bbox = self.get_bbox(o)
                bbox = self.larger_bbox(bbox, img.size[0], img.size[1])
                object_img = self.base_transforms(img.crop(bbox))
                imgs.append(object_img)
                labels.append(label)
        return imgs, labels
