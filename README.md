# Kaggle : Generative Dog Images

## On the competition 

See https://www.kaggle.com/c/generative-dog-images

The goal of this competition is to generate the best looking dogs using 64x64 images.

> Use your training skills to create images, rather than identify them. You’ll be using GANs, which are at the creative frontier of machine learning. You might think of GANs as robot artists in a sense—able to create eerily lifelike images, and even digital worlds.

Additional important informations:
- No pretrained models were allowed
- No external data was allowed
- Everything (training, generation) has to be made in under 9 hours in Kaggle kernels
- About 20000 dog images were available, their races labels and annotation boxes were also provided.

The competition took place from June 28 2019 to August 14 2019

## Metric

See https://www.kaggle.com/c/generative-dog-images/overview/evaluation

> Submissions are evaluated on MiFID (Memorization-informed Fréchet Inception Distance), which is a modification from Fréchet Inception Distance (FID).

The FID is a common metric to measure GAN performances, which measures the similarity of features computed using the Inception Network for real and generated images. Then we model the data distribution for these features using a multivariate Gaussian distribution with mean µ and covariance Σ. For a set of generated images $g$ and real images $r$, we have :

![equation](https://latex.codecogs.com/gif.latex?%5Ctext%7BFID%7D%20%3D%20%7C%7C%5Cmu_r%20-%20%5Cmu_g%7C%7C%5E2%20&plus;%20%5Ctext%7BTr%7D%20%28%5CSigma_r%20&plus;%20%5CSigma_g%20-%202%20%28%5CSigma_r%20%5CSigma_g%29%5E%7B1/2%7D%29)

The Memorization-informed part is here to make sure generated images are not too similar to the original ones, and is designed to be equal to one for a legit solution.

## Results

The architecture I proposed is the (former) state of the art ProGAN. Refer to this [paper](https://research.nvidia.com/publication/2017-10_Progressive-Growing-of), which is a great read, for more informations.
A FID of 30 was achieved on the public leaderboard.

On private leaderboard, it scored **82.8** which put me at the **6th place**.

## Repository 

- `procgan` : Code of the solution
  - `layers` : Layers used in the architecture
  - `loader` : Data Loader specific to the competition
  - `metric` : To compute the FID
  - `models` : The generator and discriminator
  - `training` : To train the models
  - `main.py` : Main
  - `paths.py` : Paths to the data, to be adapted
  - `conditional-progan-30-public.ipynb` : Solution notebook, also available at https://www.kaggle.com/theoviel/conditional-progan-30-public
- `output` : Generated images and model weights
- `input` : Input data is expected here

If you wish to reproduce the results, the easiest way is to fork [the Kaggle kernel](https://www.kaggle.com/theoviel/conditional-progan-30-public)

## Data

Data can be downloaded on the official Kaggle page : https://www.kaggle.com/c/generative-dog-images/data

## Ressources

- [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)
- [Deep Convolutional GANs](https://arxiv.org/pdf/1511.06434.pdf)
- [NVIDIA's Progressive Growing of GANs paper](https://research.nvidia.com/publication/2017-10_Progressive-Growing-of)
- [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf)
- [CGANs with Projection Discriminator](https://arxiv.org/pdf/1802.05637.pdf)
- The modeling part of the kernel is taken from [this repository](https://github.com/akanimax/pro_gan_pytorch)

## Results preview

![Generated Dogs](http://playagricola.com/Kaggle/dogs381419.png)
