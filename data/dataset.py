class MedicalDataset(Dataset):
    def __init__(self, images, masks, strong_transform=None, weak_transform=None):
        self.images = images
        self.masks = masks
        self.strong_transform = strong_transform
        self.weak_transform = weak_transform

    def __len__(self): return len(self.images)

    def __getitem__(self, index):
            img = self.images[index]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            mask = self.masks[index].squeeze().astype(np.float32)
            if mask.max() > 1.0:
                mask = mask / 255.0

            res_s = self.strong_transform(image=img, mask=mask)
            img_s, mask_s = res_s["image"], res_s["mask"].unsqueeze(0).float()

            res_w = self.weak_transform(image=img, mask=mask)
            img_w, mask_w = res_w["image"], res_w["mask"].unsqueeze(0).float()

            if img_s.max() > 1.0:
                img_s = img_s.float() / 255.0
            if img_w.max() > 1.0:
                img_w = img_w.float() / 255.0

            return img_s, img_w, mask_s, mask_w
