import torch


# # Embedding changes
# x = torch.randn(100, 1, 256)
# x_new = x.reshape(1, 100, 256)
# print(x_new)
# print(x_new.shape)

# # Bounding box changes
# x = torch.randn(100, 1, 300, 300)
# x_new = x.reshape(1, 100, -1)
# print(x_new)
# print(x_new.shape)