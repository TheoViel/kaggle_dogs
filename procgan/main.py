from procgan.training.progan_wraper import *
from procgan.loader.transforms import *
from procgan.paths import *
from procgan.training.tools import *


if __name__ == '__main__':

    KERNEL_START_TIME = time.time()
    IMAGES = [DATA_PATH + p for p in os.listdir(DATA_PATH)]

    print('\n Preparing data ... \n')
    print('Number of dog images :', len(IMAGES))

    seed = 2019
    seed_everything(seed)

    base_transforms, additional_transforms = get_transforms(64)
    dataset = DogeDataset(DATA_PATH, base_transforms, additional_transforms)

    nb_classes = len(dataset.classes)
    print(f'Number of classes : {nb_classes}')
    nb_dogs = len(dataset)
    print(f'Number of dogs : {nb_dogs}')


    print('\n Preparing model ... \n')

    depth = 5
    latent_size = 256
    loss = Hinge
    lr_d = 6e-3
    lr_g = 6e-3

    pro_gan = ConditionalProGAN(num_classes=nb_classes, depth=depth, latent_size=latent_size,
                                loss=loss, lr_d=lr_d, lr_g=lr_g,
                                use_ema=True, use_eql=True, use_spec_norm=False, seed=seed)

    print('\n Training ...')

    # num_epochs = [5, 10, 20, 40, 100]
    num_epochs = [1, 0, 0, 0, 0]

    fade_ins = [50, 20, 20, 10, 5]
    batch_sizes = [64] * 5
    ema_decays = [0.9, 0.9, 0.99, 0.99, 0.99]

    infos = pro_gan.train(
        dataset=dataset,
        epochs=num_epochs,
        fade_in_percentage=fade_ins,
        batch_sizes=batch_sizes,
        ema_decays=ema_decays,
        verbose=1
    )

    save_model_weights(pro_gan.gen, CP_PATH + "gen_weights.pt")
    save_model_weights(pro_gan.gen_shadow, CP_PATH + "gen_shadow_weights.pt")
    save_model_weights(pro_gan.dis, CP_PATH + "dis_weights.pt")

    print('\n Results ...\n')

    plot_infos(infos, num_epochs)

    pro_gan.generate(n_plot=25, n=batch_sizes[0])

    print('\n Generating images ...\n')

    im_batch_size = 100
    n_images = 10000

    if os.path.exists(IMG_PATH):
        shutil.rmtree(IMG_PATH, ignore_errors=True)
    os.mkdir(IMG_PATH)

    for i_batch in range(0, n_images, im_batch_size):
        gen_images = pro_gan.generate(n=im_batch_size)
        for i_image in range(gen_images.size(0)):
            save_image(gen_images[i_image, :, :, :], os.path.join('../output_images', f'img_{i_batch + i_image}.png'))

    print('Number of generated images :', len(os.listdir(IMG_PATH)))
