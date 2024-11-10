import math

dataset_path = "~/datasets"
batch_size_ref = 128
multiplier = 4
train_batch_size = batch_size_ref * multiplier
test_batch_size = 1
x_dim = 784 # 28*28
hidden_dim = 512
latent_dim = 2
depth = 3
lr = 1e-3 / math.sqrt(multiplier)
epochs = 120
# 90 - 134
# 120 - 133
weight_path = "tmp/weights/vae_120.pth"
