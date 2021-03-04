import copy
import os
import simplejson as json
import click
import imageio
import numpy as np
import PIL.Image
import torch
import torchvision
import torch.nn.functional as F
import dnnlib
import legacy
import clip
import hashlib 


def approach(
    G,
    *,
    num_steps                  = 100,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.02,  
    initial_noise_factor       = 0.02,  
    noise_floor                = 0.02, 
    psi                        = 0.8,
    noise_ramp_length          = 1.0, # was 0.75
    regularize_noise_weight    = 10000, # was 1e5
    seed                       = 69097, 
    noise_opt                  = True, 
    ws                         = None,
    text                       = 'a computer generated image', 
    device: torch.device
):

    '''
    local_args = dict(locals())
    params = []
    for x in local_args:
        if x != 'G' and x != 'device':
            print(x,':',local_args[x])
            params.append({x:local_args[x]})
    print(json.dumps(params))
    '''

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)

    lr = initial_learning_rate

    '''
    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    #w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device),  None, truncation_psi=0.8)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
    '''

    # derive W from seed
    if ws is None:
        print('Generating w for seed %i' % seed )
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        w_samples = G.mapping(z,  None, truncation_psi=psi)
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)
        w_avg = np.mean(w_samples, axis=0, keepdims=True)
    else:
        w_samples = torch.tensor(ws, device=device)
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)
        w_avg = np.mean(w_samples, axis=0, keepdims=True)
    #w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
    w_std = 2 # ~9.9 for portraits network. should compute if using median median

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }
    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)

    if noise_opt:
        optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
        print('optimizer: w + noise')
    else:
        optimizer = torch.optim.Adam([w_opt] , betas=(0.9, 0.999), lr=initial_learning_rate)
        print('optimizer: w')

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    # Load the perceptor
    print('Loading perceptor for text:', text)
    perceptor, preprocess = clip.load('ViT-B/32', jit=True)
    perceptor = perceptor.eval()
    tx = clip.tokenize(text)
    whispers = perceptor.encode_text(tx.cuda()).detach().clone()

    # Descend
    for step in range(num_steps):
        # noise schedule
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2

        # floor
        if w_noise_scale < noise_floor:
            w_noise_scale = noise_floor

        # lr schedule is disabled
        '''
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        '''

        ''' for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        '''

        # do G.synthesis
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')

        #save1
        '''
        synth_images_save = (synth_images + 1) * (255/2)        
        synth_images_save = synth_images_save.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        PIL.Image.fromarray(synth_images_save, 'RGB').save('project/test1.png')
        '''

        nom = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))        
        into = synth_images
        into = nom(into) # normalize copied from CLIP preprocess. doesn't seem to affect tho

        # scale to CLIP input size
        into = torch.nn.functional.interpolate(synth_images, (224,224), mode='bilinear', align_corners=True)

        # CLIP expects [1, 3, 224, 224], so we should be fine
        glimmers = perceptor.encode_image(into)
        away =  -30 * torch.cosine_similarity(whispers, glimmers, dim = -1).mean() # Dunno why 30 works lol

        # noise reg, from og projector
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)

        if noise_opt:
            loss = away + reg_loss * regularize_noise_weight
        else:
            loss = away

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        print(f'step {step+1:>4d}/{num_steps}:  loss {float(loss):<5.2f} ','lr', lr, f'noise scale: {float(w_noise_scale):<5.6f}',f'away: {float(away / (-30)):<5.6f}')

        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.mapping.num_ws, 1])

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed',                   help='Initial image seed', type=int, default=232322, show_default=True)
@click.option('--w',                      help='Do not use seed but load w from a file', type=str, metavar='FILE')
@click.option('--lr',                     help='Adam learning rate', type=float, required=False, default=0.02)
@click.option('--psi',                    help='Truncation psi for initial image', type=float, required=False, default=0.81)
@click.option('--inf',                    help='Initial noise factor', type=float, required=False, default=0.02)
@click.option('--nf',                     help='Noise floor', type=float, required=False, default=0.02)
@click.option('--noise-opt',              help='Optimize noise vars as well as w', type=bool, required=False, default=True)
@click.option('--text',                   help='Text prompt', required=False, default='A computer-generated image')
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--save-ws',                help='Save intermediate ws', type=bool, default=False, show_default=True)

def run_approach(
    network_pkl: str,
    outdir: str,
    save_video: bool,
    save_ws: bool,
    seed: int,
    num_steps: int,
    text: str,
    lr: float,
    inf: float,
    nf: float,
    w: str,
    psi: float,
    noise_opt: bool
):
    """Descend on StyleGAN2 w vector value using CLIP, tuning an image with given text prompt. 

    Example:

    \b
    python3 approach.py --network network-snapshot-ffhq.pkl --outdir project --num-steps 100  \\ 
    --text 'an image of a girl with a face resembling Paul Krugman' --psi 0.8 --seed 12345
    
    """

    #seed = 1
    np.random.seed(1)
    torch.manual_seed(1)

    local_args = dict(locals())
    params = []
    for x in local_args:
        #if x != 'G' and x != 'device':
        #print(x,':',local_args[x])
        params.append({x:local_args[x]})
    #print(json.dumps(params))
    hashname = str(hashlib.sha1((json.dumps(params)).encode('utf-16be')).hexdigest() )
    print('run hash', hashname)

    ws = None
    if w is not None:
        print ('loading w from file', w, 'ignoring seed and psi')
        ws = np.load(w)['w']
        
    # take off
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    # approach
    projected_w_steps = approach(
        G,
        num_steps=num_steps,
        device=device,
        initial_learning_rate = lr,
        psi = psi,
        seed = seed,
        initial_noise_factor = inf,
        noise_floor = nf,
        text = text,
        ws = ws,
        noise_opt = noise_opt
    )

    # save video
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/out-{hashname}.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/out-{hashname}.mp4"')
        for projected_w in projected_w_steps:
            synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            video.append_data(np.concatenate([synth_image], axis=1))
        video.close()

    # save ws
    if save_ws:
        print ('Saving optimization progress ws')
        step = 0
        for projected_w in projected_w_steps:
            np.savez(f'{outdir}/w-{hashname}-{step}.npz', w=projected_w.unsqueeze(0).cpu().numpy())
            step+=1

    # save the result and the final w
    print ('Saving finals')
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const') 
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/out-{hashname}.png')
    np.savez(f'{outdir}/w-{hashname}-final.npz', w=projected_w.unsqueeze(0).cpu().numpy())


    # save params
    with open(f'{outdir}/params-{hashname}.txt', 'w') as outfile:
        json.dump(params, outfile)

if __name__ == "__main__":
    run_approach() 
