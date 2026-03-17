def PrintAllDiceScores(model, device, test_loader):
    model.eval()
    all_dice_scores = []
    print("Calculating Dice Scores for all images in the test set...")
    with torch.no_grad():
        for i, (img, _, msk, _) in enumerate(test_loader):
            img, msk = img.to(device), msk.to(device)
            output, _ = model(img, phase="test")
            dice = DiceScore(output, msk)
            all_dice_scores.append(dice)
            print(f"Image {i+1}: Dice Score = {dice:.4f}")

    avg_dice = np.mean(all_dice_scores)
    print(f"\nAverage Dice Score across all test images: {avg_dice:.4f}")
    return all_dice_scores
