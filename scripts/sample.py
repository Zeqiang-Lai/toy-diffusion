import torchvision.utils as utils
import imageio
import torch
from torchvision.utils import save_image
from toy_diffusion.model.unet_g import Unet
from toy_diffusion import GaussianDiffusion

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size=128,
    timesteps=1000,   # number of steps
    loss_type='l1',    # L1 or L2
    # objective='pred_x0'
).cuda()

ckpt = torch.load('results/model-52.pt')
diffusion.load_state_dict(ckpt['model'])

sampled_images = diffusion.sample(batch_size=4)

save_image(sampled_images, 'test.png')

images = []
for x in diffusion.x_starts:
    grid = utils.make_grid(x)
    images.append(grid.permute(1, 2, 0).cpu().numpy())

imageio.mimsave('x_start.mp4', images, fps=20)
