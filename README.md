# Navigating StyleGAN2 w latent space using CLIP 

an attempt to build sth with [the official SG2-ADA Pytorch impl](https://github.com/NVlabs/stylegan2-ada-pytorch)
kinda inspired by [Generating Images from Prompts using CLIP and StyleGAN](https://towardsdatascience.com/generating-images-from-prompts-using-clip-and-stylegan-1f9ed495ddda)
based on the og `projector.py`

things learned:
- it's better to generate initial w values from a well converged sample rather than starting with random or median ones
- optimizing w and noise inputs works better than w alone

<!-- ![Example](test.png) -->

## Quick start

- clone [SG2 repo](https://github.com/NVlabs/stylegan2-ada-pytorch), copy `clip` dir from [CLIP repo](https://github.com/openai/CLIP), install pytorch 1.7.1 and stuff
- pick a suitable SG2 PKL (eg FFHQ)
- pick a seed
- run
`python3 approach.py --network network-snapshot-ffhq.pkl --outdir project --num-steps 100 --text 'an image of a girl with a face resembling Paul Krugman' --psi 0.8 --seed 12345`
