from toy_diffusion.model.unet_bit import Unet
from toy_diffusion import BitDiffusion, Trainer

model = Unet(
    dim=32,
    channels=3,
    dim_mults=(1, 2, 4, 8),
).cuda()

bit_diffusion = BitDiffusion(
    model,
    image_size=128,
    timesteps=100,
    time_difference=0.1,       # they found in the paper that at lower number of timesteps, a time difference during sampling of greater than 0 helps FID. as timesteps increases, this time difference can be set to 0 as it does not help
    use_ddim=True              # use ddim
).cuda()

trainer = Trainer(
    bit_diffusion,
    'data/celeba',             # path to your folder of images
    results_folder='log/bit_diffusion',     # where to save results
    num_samples=16,                 # number of samples
    train_batch_size=4,             # training batch size
    gradient_accumulate_every=4,    # gradient accumulation
    train_lr=1e-4,                  # learning rate
    save_and_sample_every=1000,     # how often to save and sample
    train_num_steps=700000,         # total training steps
    ema_decay=0.995,                # exponential moving average decay
)

trainer.train()
