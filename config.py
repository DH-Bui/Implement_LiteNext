class Args:
    def __init__(self):
        self.root = "/content/drive/MyDrive/Dataset/ISIC2018/"
        self.epochs = 300
        self.batch_size = 8
        self.mini_batch = 4
        self.lr = 1e-5
        self.lambda_con = 0.1
        self.w_margin = 6.0

args = Args()
