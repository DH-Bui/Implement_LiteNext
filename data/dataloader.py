def get_dataloader(batch_size):
    x_train = np.load(os.path.join(args.root, "x_train.npy"))
    y_train = np.load(os.path.join(args.root, "y_train.npy"))
    x_test = np.load(os.path.join(args.root, "x_test.npy"))
    y_test = np.load(os.path.join(args.root, "y_test.npy"))

    strong_tf = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(rotate=(-30, 30), scale=(0.8, 1.2), p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    weak_tf = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    train_loader = DataLoader(MedicalDataset(x_train, y_train, strong_tf, weak_tf), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MedicalDataset(x_test, y_test, weak_tf, weak_tf), batch_size=1, shuffle=False)
    return train_loader, test_loader
