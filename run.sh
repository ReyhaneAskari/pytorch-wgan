#!/usr/bin/env bash

# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 0.0  --alpha_g_grad 1.0 --alpha_g_vjp 0.5 
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 0.0  --alpha_g_grad 1.0 --alpha_g_vjp 1.0
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 0.0  --alpha_g_grad 1.0 --alpha_g_vjp 0.25
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 0.0  --alpha_g_grad 1.0 --alpha_g_vjp 0.1
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 0.0  --alpha_g_grad 1.0 --alpha_g_vjp 0.01
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 0.0  --alpha_g_grad 1.0 --alpha_g_vjp 0.7
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 0.0  --alpha_g_grad 1.0 --alpha_g_vjp 0.001
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 0.0  --alpha_g_grad 1.0 --alpha_g_vjp 2.0
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 0.0  --alpha_g_grad 1.0 --alpha_g_vjp 3.0
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 0.0  --alpha_g_grad 1.0 --alpha_g_vjp 4.0

# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 0.50  --alpha_g_grad 1.0 --alpha_g_vjp 0.0
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 1.00  --alpha_g_grad 1.0 --alpha_g_vjp 0.0
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 0.25  --alpha_g_grad 1.0 --alpha_g_vjp 0.0
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 0.10  --alpha_g_grad 1.0 --alpha_g_vjp 0.0
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 0.01  --alpha_g_grad 1.0 --alpha_g_vjp 0.0
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 0.70  --alpha_g_grad 1.0 --alpha_g_vjp 0.0
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 0.001  --alpha_g_grad 1.0 --alpha_g_vjp 0.0
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 2.00  --alpha_g_grad 1.0 --alpha_g_vjp 0.0
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 3.00  --alpha_g_grad 1.0 --alpha_g_vjp 0.0
# python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 4.00  --alpha_g_grad 1.0 --alpha_g_vjp 0.0

python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 0.50  --alpha_g_grad 1.0 --alpha_g_vjp 0.50
python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 1.00  --alpha_g_grad 1.0 --alpha_g_vjp 1.0
python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 0.25  --alpha_g_grad 1.0 --alpha_g_vjp 0.25
python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 0.10  --alpha_g_grad 1.0 --alpha_g_vjp 0.10
python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 0.01  --alpha_g_grad 1.0 --alpha_g_vjp 0.01
python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 0.70  --alpha_g_grad 1.0 --alpha_g_vjp 0.70
python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 0.001  --alpha_g_grad 1.0 --alpha_g_vjp 0.001
python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 2.00  --alpha_g_grad 1.0 --alpha_g_vjp 2.0
python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 3.00  --alpha_g_grad 1.0 --alpha_g_vjp 3.0
python main.py --dataroot datasets/cifar --epochs 10 --beta_g 0.5 --beta_d 0.5 --lr_g 0.0002 --lr_d 0.0001 --alpha_d_grad 1.0 --alpha_d_vjp 4.00  --alpha_g_grad 1.0 --alpha_g_vjp 4.0


