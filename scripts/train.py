from toy_diffusion.model.unet_g import Unet
from toy_diffusion import GaussianDiffusion, Trainer

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size=128,
    timesteps=100,   # number of steps
    loss_type='l1',    # L1 or L2
    objective='pred_x0'
).cuda()

trainer = Trainer(
    diffusion,
    'celeba/img_align_celeba',
    results_folder='celeba_results',
    train_batch_size=32,
    train_lr=1e-4,
    train_num_steps=700000,         # total training steps
    gradient_accumulate_every=2,    # gradient accumulation steps
    ema_decay=0.995,                # exponential moving average decay
    amp=True                        # turn on mixed precision
)

trainer.train()
