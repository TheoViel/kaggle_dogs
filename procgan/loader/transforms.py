from torchvision.transforms import transforms

class Rescale:
    """
    Scales a [0; 1]image to [-1; 1]
    """
    def __init__(self):
        self.a = 2
        self.b = -1

    def __call__(self, tensor):
        return tensor.mul(self.a).add(self.b)

    def __repr__(self):
        return self.__class__.__name__ + '(x{}, +{})'.format(self.a, self.b)


def get_transforms(size):
    base_transforms = transforms.Compose([transforms.Resize(size)])
    additional_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomApply([transforms.RandomRotation(degrees=5)], p=0.5),
        transforms.RandomChoice([transforms.CenterCrop(size), transforms.RandomCrop(size)]),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=(0.9, 1.2), saturation=0.3, hue=0.01)], p=0.5),
        transforms.ToTensor(),
        # transforms.Normalize(means, stds),
        Rescale(),
    ])
    return base_transforms, additional_transforms
