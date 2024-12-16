import math

dataset_path = "~/datasets"
batch_size_ref = 128
multiplier = 4
train_batch_size = batch_size_ref * multiplier
test_batch_size = 128
x_dim = 784 # 28*28
hidden_dim = 1024
latent_dim = 2
depth = 3
epochs = 1000
# 90 - 134
# 120 - 133
weight_path = "./aae_120.pth"

# Learning rate for optimizers
lr = 0.0002
# lr = 1e-3 / math.sqrt(multiplier)

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5
