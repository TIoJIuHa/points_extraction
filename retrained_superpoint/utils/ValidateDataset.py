import torch
from pathlib import Path
import torch.utils.data as data

from utils.utils import compute_valid_mask
import cv2


class ValidateDataset(data.Dataset):
    def __init__(self, transform=None, task='train', **config):
        self.config = config

        self.transforms = transform
        self.action = 'train' if task == 'train' else 'val'
        if self.config['preprocessing']['resize']:
            self.sizer = self.config['preprocessing']['resize']

        # get files
        base_path = Path(self.config['images_folder'])
        image_paths = list(base_path.iterdir())
        image_paths = sorted(image_paths, key=lambda name: int(name.stem.split("_")[-1]))

        names = [p.stem for p in image_paths]
        image_paths = [str(p) for p in image_paths]
        files = {'image_paths': image_paths, 'names': names}


        sequence_set = []
        for (img, name) in zip(files['image_paths'], files['names']):
            sample = {'image': img, 'name': name}
            sequence_set.append(sample)
        self.samples = sequence_set

        self.init_var()

    def init_var(self):
        torch.set_default_tensor_type(torch.FloatTensor)

        self.compute_valid_mask = compute_valid_mask

    def get_img_from_sample(self, sample):
        return sample['image']

    def format_sample(self, sample):
        return sample

    def __getitem__(self, index):
        '''
        :param index:
        :return:
            image: tensor (H, W, channel=1)
        '''
        def _read_image(path):
            input_image = cv2.imread(path)
            H, W = input_image.shape[0], input_image.shape[1]
            input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)

            input_image = input_image.astype('float32') / 255.0
            return input_image

        sample = self.samples[index]
        sample = self.format_sample(sample)
        input  = {}
        input.update(sample)
        img_o = _read_image(sample['image'])
        H, W = img_o.shape[0], img_o.shape[1]
        img_aug = img_o.copy()

        img_aug = torch.tensor(img_aug, dtype=torch.float32).view(-1, H, W)

        valid_mask = self.compute_valid_mask(torch.tensor([H, W]), inv_homography=torch.eye(3))
        input.update({'image': img_aug})
        input.update({'valid_mask': valid_mask})
        name = sample['name']
        input.update({'name': name, 'scene_name': "./"})
        return input

    def __len__(self):
        return len(self.samples)
