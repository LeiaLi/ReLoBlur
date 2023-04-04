import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class PairRandomCrop(transforms.RandomCrop):

    def __call__(self, imgs):

        if self.padding is not None:
            imgs = [F.pad(image, self.padding, self.fill, self.padding_mode) for image in imgs]

        # pad the width if needed
        if self.pad_if_needed and imgs[0].size[0] < self.size[1]:
            imgs = [F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode) for image in imgs]
        # pad the height if needed
        if self.pad_if_needed and imgs[0].size[1] < self.size[0]:
            imgs = [F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode) for image in imgs]

        i, j, h, w = self.get_params(imgs[0], self.size)

        return [F.crop(image, i, j, h, w) for image in imgs]


class PairCompose(transforms.Compose):
    def __call__(self, imgs):
        for t in self.transforms:
            imgs = t(imgs)
        return imgs


class PairRandomHorizontalFilp(transforms.RandomHorizontalFlip):
    def __call__(self, imgs):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return [F.hflip(img) for img in imgs]
        return imgs


class PairToTensor(transforms.ToTensor):
    def __call__(self, imgs):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return [F.to_tensor(pic) for pic in imgs]
