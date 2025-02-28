# AutoEncoder with Half VGG19 + Batch Normalization

This repository contains an **AutoEncoder** that utilizes the **first half of VGG19 with Batch Normalization** as the encoder. The model is designed for **image reconstruction** tasks.
---

## 🛠️ **Architecture Overview**

The AutoEncoder consists of:

1. **Encoder**: The first 26 layers of `VGG19-BN` pretrained on ImageNet (excluding fully connected layers).
3. **Decoder**: A series of transposed convolutions and Leaky RELU.

**Main Features:**
- Uses `VGG19-BN` as the encoder.
- A decoder with `ConvTranspose2d` layers and `LeakyRelu` for reconstruction.
- Trained using **Adam optimizer** and a series pof Losses: **MSE**, **MAE** and **BinaryCrossEntropy**.
- Each loss was selected bases on the normalization applied to the images: **IMAGENET Values**, between [0,1] and between [-1,1].
