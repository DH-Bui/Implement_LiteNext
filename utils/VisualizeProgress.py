import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def VisualizeProgress(model, device, test_loader, epoch):
    model.eval()
    with torch.no_grad():
        img_s, _, msk_s, _ = next(iter(test_loader))
        img, msk = img_s.to(device), msk_s.to(device)

        output, _ = model(img, phase="test")
        current_dice = DiceScore(output, msk)

        img_visual = img[0].cpu().permute(1, 2, 0).numpy()

        img_visual = (img_visual - img_visual.min()) / (img_visual.max() - img_visual.min() + 1e-8)

        msk_np = msk[0][0].cpu().numpy()
        prob_map = torch.sigmoid(output[0][0]).cpu().numpy()
        pred_bin = (prob_map > 0.5).astype(np.float32)

        fig, ax = plt.subplots(1, 4, figsize=(24, 6))

        ax[0].imshow(img_visual)
        ax[0].set_title("1. Original")
        ax[0].axis('off')

        ax[1].imshow(msk_np, cmap='gray')
        ax[1].set_title("2. Ground Truth")
        ax[1].axis('off')

        ax[2].imshow(prob_map, cmap='jet')
        ax[2].set_title(f"3. Prob Map (Dice: {current_dice:.4f})")
        ax[2].axis('off')

        ax[3].imshow(img_visual)
        red_mask = np.zeros((*pred_bin.shape, 4))
        red_mask[pred_bin == 1] = [1, 0, 0, 0.4]
        ax[3].imshow(red_mask)
        ax[3].set_title("4. Overlay Prediction")
        ax[3].axis('off')
        plt.show()
        plt.close()
