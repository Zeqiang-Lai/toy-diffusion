from toy_diffusion.model.unet import Unet
from toy_diffusion import BitDiffusion, Trainer


def main(results_folder):
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=8,
    )

    diffusion = BitDiffusion(
        model,
        image_size=128,
        timesteps=100, 
        bit_scale=0.1,
    )

    trainer = Trainer(
        diffusion,
        'data/ade20k/annotations/training',           
        results_folder=results_folder,    
        num_samples=16,                 
        train_batch_size=32,             
        gradient_accumulate_every=2,  
        train_lr=1e-4,                 
        save_every=1000,              
        sample_every=1000,              
        train_num_steps=700000,      
        ema_decay=0.995,              
    )

    trainer.load('last', ignore_if_not_exists=True)
    trainer.train()


if __name__ == '__main__':
    results_folder = 'log/bit_diffusion_ade20k'
    
     # save training script for reproducible experiments
    import os, shutil
    src_file = os.path.abspath(__file__)
    tgt_file = os.path.join(results_folder, os.path.basename(src_file))
    os.makedirs(results_folder, exist_ok=True)
    shutil.copy(src_file, tgt_file)
    
    main(results_folder)
