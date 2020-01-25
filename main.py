from utils.config import parse_args
from utils.data_loader import get_data_loader

from models.gan import GAN
from models.dcgan import DCGAN_MODEL
from models.wgan_clipping import WGAN_CP
from models.wgan_gradient_penalty import WGAN_GP

# python main.py --model DCGAN --is_train True --download True --dataroot datasets/cifar --dataset cifar --epochs 50 --cuda True --batch_size 64 --mode adam_vjp --beta_g 0.5 beta_d 0.5 lr_g 0.0002 lr_d 0.0002 alpha_d 0.001 alpha_g 0.002

def main(args):
    model = None
    if args.model == 'GAN':
        model = GAN(args)
    elif args.model == 'DCGAN':
        model = DCGAN_MODEL(args)
    elif args.model == 'WGAN-CP':
        model = WGAN_CP(args)
    elif args.model == 'WGAN-GP':
        model = model = WGAN_GP(args)
    else:
        print("Model type non-existing. Try again.")
        exit(-1)
    # Load datasets to train and test loaders
    train_loader, test_loader = get_data_loader(args)
    #feature_extraction = FeatureExtractionTest(train_loader, test_loader, args.cuda, args.batch_size)

    # Start model training
    if args.is_train == 'True':
        model.train(train_loader)

    # start evaluating on test data
    else:
        model.evaluate(test_loader, args.load_D, args.load_G)
        # for i in range(50):
        #    model.generate_latent_walk(i)


if __name__ == '__main__':
    args = parse_args()
    main(args)
