from data.base_dataset import BaseDataset, get_transform,get_transform_mask, dataread
from data.image_folder import make_dataset
from PIL import Image


class SingleDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.mask_paths = dataread(opt.shadow_dir)
        self.shadow_paths = dataread(opt.mask_dir)
        self.nonshadow_path = dataread(opt.nonshadow_dir)
        self.transform = get_transform(opt)
        self.transform_mask = get_transform_mask(opt)


    def __getitem__(self, index):
        mask_path = self.mask_paths[index]
        mask = Image.open(mask_path).convert('RGB')
        mask = self.transform_mask(mask)
        mask = mask[0,:,:]
        mask = mask.view(1,mask.shape[0],mask,shape[1])

        shadow_path = self.shadow_paths[index]
        shadow = Image.open(shadow_path).convert('RGB')
        shadow = self.transform(shadow)

        nonshadow_path = self.nonshadow_paths[index]
        nonshadow = Image.open(nonshadow_path).convert('RGB')
        nonshadow = self.transform(nonshadow)

        return {'mask': mask, 'shadow': shadow, 'nonshadow':nonshadow, 'shadow_paths': shadow_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
