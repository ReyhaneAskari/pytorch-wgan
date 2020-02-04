#!/usr/bin/env bash

# non-zero_sim best:
# python main.py --dataroot datasets/cifar --epochs 120 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_vjp 0.3  --alpha_g_vjp 0.0 --load_model True --load_G res/best/generator.pkl --load_D res/best/discriminator.pkl

python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.0 --lr_g 0.0001 --lr_d 0.0001 --alpha_d_vjp 0.3 --alpha_g_vjp 0.0
python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d -0.5 --lr_g 0.0001 --lr_d 0.0001 --alpha_d_vjp 0.3 --alpha_g_vjp 0.0
python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0001 --lr_d 0.0001 --alpha_d_vjp 0.3 --alpha_g_vjp 0.0
python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d -0.1 --lr_g 0.0001 --lr_d 0.0001 --alpha_d_vjp 0.3 --alpha_g_vjp 0.0
python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.1 --lr_g 0.0001 --lr_d 0.0001 --alpha_d_vjp 0.3 --alpha_g_vjp 0.0

python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.0 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_vjp 0.3 --alpha_g_vjp 0.0
python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d -0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_vjp 0.3 --alpha_g_vjp 0.0
python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_vjp 0.3 --alpha_g_vjp 0.0
python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d -0.1 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_vjp 0.3 --alpha_g_vjp 0.0
python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d +0.1 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_vjp 0.3 --alpha_g_vjp 0.0

# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d +0.0 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_vjp 0.3  --alpha_g_vjp 0.001
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d -0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_vjp 0.3  --alpha_g_vjp 0.001
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d +0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_vjp 0.3  --alpha_g_vjp 0.001
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d -0.1 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_vjp 0.3  --alpha_g_vjp 0.001
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d +0.1 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_vjp 0.3  --alpha_g_vjp 0.001

# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d +0.0 --lr_g 0.0001 --lr_d 0.0001 --alpha_d_vjp 0.3  --alpha_g_vjp 0.001
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d -0.5 --lr_g 0.0001 --lr_d 0.0001 --alpha_d_vjp 0.3  --alpha_g_vjp 0.001
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d +0.5 --lr_g 0.0001 --lr_d 0.0001 --alpha_d_vjp 0.3  --alpha_g_vjp 0.001
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d -0.1 --lr_g 0.0001 --lr_d 0.0001 --alpha_d_vjp 0.3  --alpha_g_vjp 0.001
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d +0.1 --lr_g 0.0001 --lr_d 0.0001 --alpha_d_vjp 0.3  --alpha_g_vjp 0.001

# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d +0.0 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_vjp 0.35  --alpha_g_vjp 0.0
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d -0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_vjp 0.35  --alpha_g_vjp 0.0
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d +0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_vjp 0.35  --alpha_g_vjp 0.0
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d -0.1 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_vjp 0.35  --alpha_g_vjp 0.0
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d +0.1 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_vjp 0.35  --alpha_g_vjp 0.0

# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d +0.0 --lr_g 0.0001 --lr_d 0.0001 --alpha_d_vjp 0.35  --alpha_g_vjp 0.0
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d -0.5 --lr_g 0.0001 --lr_d 0.0001 --alpha_d_vjp 0.35  --alpha_g_vjp 0.0
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d +0.5 --lr_g 0.0001 --lr_d 0.0001 --alpha_d_vjp 0.35  --alpha_g_vjp 0.0
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d -0.1 --lr_g 0.0001 --lr_d 0.0001 --alpha_d_vjp 0.35  --alpha_g_vjp 0.0
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d +0.1 --lr_g 0.0001 --lr_d 0.0001 --alpha_d_vjp 0.35  --alpha_g_vjp 0.0
