from toy_diffusion.model.unet import Unet
from toy_diffusion import BitDiffusion, Trainer
from toy_diffusion.diffusion.bit import BITS



def main(results_folder):
    model = Unet(
        dim=32,
        channels=3 * BITS,
        dim_mults=(1, 2, 4, 8),
    )

    bit_diffusion = BitDiffusion(
        model,
        image_size=128,
        timesteps=100,
        time_difference=0.1,       # they found in the paper that at lower number of timesteps, a time difference during sampling of greater than 0 helps FID. as timesteps increases, this time difference can be set to 0 as it does not help
        use_ddim=True
    )

    trainer = Trainer(
        bit_diffusion,
        'data/celeba',
        results_folder=results_folder,
        num_samples=16,
        train_batch_size=4,
        gradient_accumulate_every=4,
        train_lr=1e-4,
        save_every=1000,
        sample_every=1000,
        train_num_steps=700000,
        ema_decay=0.995,
    )
    
    trainer.load('last', ignore_if_not_exists=True)
    trainer.train()


if __name__ == '__main__':
    results_folder = 'log/bit_diffusion'
    
     # save training script for reproducible experiments
    import os, shutil
    src_file = os.path.abspath(__file__)
    tgt_file = os.path.join(results_folder, os.path.basename(src_file))
    os.makedirs(results_folder, exist_ok=True)
    shutil.copy(src_file, tgt_file)
    
    main(results_folder)
