import copy
import os
import simplejson as json
import click
import imageio
import numpy as np
import PIL.Image
import PIL.ImageOps
import torch
import torchvision
import torch.nn.functional as F
import dnnlib
import legacy
import clip
import hashlib 
from random import randint #+
import dlib
from tqdm import tqdm
from imgcat import imgcat
#import PIL.ImageDraw
from PIL import ImageDraw

#
# Used for faces w/ffhq 1024px
# Make sure source image is nicely aligned and 1024px
#

def capture(
    G,
    *,
    num_steps                  = 100,
    w_avg_samples              = 10000, # 10k is ok. 30k is better.
    initial_learning_rate      = 0.02,  
    initial_noise_factor       = 0.02,  
    noise_floor                = 0.00, # keep it OG 
    psi                        = 0.60,
    noise_ramp_length          = 1.30, # 0.75 fmeaining during the first 75/100 operations w noise goes to zero
    regularize_noise_weight    = 10000, # was 1e5
    seed                       = 69097, 
    noise_opt                  = True, 
    lr_rampdown_length          =0.25, # means final 25% of iters we commence rampdown
    lr_rampup_length          = 0.05,
    portrait                   = None,
    autoportrait                = None, # sth does not work with anime models
    autoseed                   = False, 
    imgc                   = None,  # display some images in terminal, does not work with tmux/screen tho
    autoseed_samples           = 100,
    Image                      = 'target.png', #it's always image 
    anime                       = True, # anime means don't try to ffhq crop the source image
    multimode                      = None,  # use both image and text
    ws                         = None,
    pmul                       = 30, # multiplier for cosine_similarity,dunno why 30 works lol
    apper_range                 = 16, # when using counteradversarial, a number of random crops
    kmeans                       = True, # try to autoseed using kmean clustering
    stylemix                    = True, # was called noise_opt. now for the avoidance of doubt we call it stylemix
    vgg                         = True, # it's always VGG
    catgirl             = True, # optimization of custodee for catgirls
    w_batch_size                = 1, # number of images to synthesize and average during each step
    device: torch.device
):



    chin_bottom = 511 # normally 511 but sometimes 500 or so

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    lr = initial_learning_rate
    apper_range = 8 # when increasing batch size of generated images w/random noise, decrease it in order to keep CLIP batch size reasonable

    # Load the perceptor
    print('Loading VGG')
    # https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt
    vgg16 = torch.jit.load('vgg16.pt').eval().to(device)
    autoseed = None
    kmeans = None
    multimode = None
    slit = 224
    w_batch_size = 1
    ssf = 1

    print('Autoportrait (facial landmark crop):', autoportrait)
    if autoportrait:
        portrait = True # assert
        print('Autoportrait (facial landmark crop) enabled, asserting portrait mode on')

    if kmeans:
        autoseed = True # assert
        print('Kmeans enabled, asserting autoseed mode on')

    #if portrait: # initialize face detector, gotta be used later
    if True:
        detector = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    print('Using target image(s) as classifier with VGG', Image)
    if anime:
        print('Anime mode is on; not attempting to crop the target image(s)')


    # some prereqs
    with PIL.Image.open(Image) as img:
        width, height = img.size
        if width == 1024 or height == 1024:
            ssf = 2


    img = np.array(PIL.Image.open(Image).convert('RGB').resize((1024,1024), PIL.Image.BICUBIC), dtype=np.uint8)

    #
    #
    # bes | besa | besal
    # ------------------
    # besi|besil | beso
    # 
    # bes/besal/besi/beso wid = 300
    # besa/besil = 424
    # h = 600

    # img [y:bottom, x:right, :]
    bes = img[0:600, 0:300, :  ] 
    print(Image,'shape for bes',bes.shape)
    bes = np.array(PIL.Image.fromarray(bes).resize((slit,slit), PIL.Image.BICUBIC), dtype=np.uint8)
    bes = torch.tensor(bes.transpose([2, 0, 1]), device=device).unsqueeze(0).to(device).to(torch.float32)
    bes = vgg16(bes, resize_images=False, return_lpips=True)
    print(Image,'tensor shape for bes',bes.shape)

    besa = img[0:600, 301:725, :  ] 
    imgcat(PIL.Image.fromarray(besa) )
    print(Image,'shape for besa',besa.shape)
    besa = np.array(PIL.Image.fromarray(besa).resize((slit,slit), PIL.Image.BICUBIC), dtype=np.uint8)
    besa = torch.tensor(besa.transpose([2, 0, 1]), device=device).unsqueeze(0).to(device).to(torch.float32)
    besa = vgg16(besa, resize_images=False, return_lpips=True)
    print(Image,'tensor shape for besa',besa.shape)


    besal = img[0:600, 726:1023, :  ] 
    print(Image,'shape for besal',besal.shape)
    besal = np.array(PIL.Image.fromarray(besal).resize((slit,slit), PIL.Image.BICUBIC), dtype=np.uint8)
    besal = torch.tensor(besal.transpose([2, 0, 1]), device=device).unsqueeze(0).to(device).to(torch.float32)
    besal = vgg16(besal, resize_images=False, return_lpips=True)
    print(Image,'tensor shape for besal',besal.shape)

    besi = img[601:1023, 0:300, :  ] 
    print(Image,'shape for besi',besi.shape)
    besi = np.array(PIL.Image.fromarray(besi).resize((slit,slit), PIL.Image.BICUBIC), dtype=np.uint8)
    besi = torch.tensor(besi.transpose([2, 0, 1]), device=device).unsqueeze(0).to(device).to(torch.float32)
    besi = vgg16(besi, resize_images=False, return_lpips=True)
    print(Image,'tensor shape for bes',bes.shape)

    besil = img[601:1023, 301:725, :  ] 
    print(Image,'shape for besil',besil.shape)
    besil = np.array(PIL.Image.fromarray(besil).resize((slit,slit), PIL.Image.BICUBIC), dtype=np.uint8)
    besil = torch.tensor(besil.transpose([2, 0, 1]), device=device).unsqueeze(0).to(device).to(torch.float32)
    besil = vgg16(besil, resize_images=False, return_lpips=True)
    print(Image,'tensor shape for besil',besil.shape)

    beso = img[601:1023, 726:1023, :  ] 
    print(Image,'shape for beso',beso.shape)
    beso = np.array(PIL.Image.fromarray(beso).resize((slit,slit), PIL.Image.BICUBIC), dtype=np.uint8)
    beso = torch.tensor(beso.transpose([2, 0, 1]), device=device).unsqueeze(0).to(device).to(torch.float32)
    beso = vgg16(beso, resize_images=False, return_lpips=True)
    print(Image,'tensor shape for beso',beso.shape)

     
    print('done initializing classifiers')
    # done initializing classifiers


    # it we still haven't got w at this moment, derive it from seed or load
    if ws is None: # means seed is provided (either as parameter or autoseed, derive w from z
        print('Generating w for seed %i' % seed )
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        # save the image for that z
        synth_images = G(torch.from_numpy(np.random.RandomState(seed).randn(1,G.z_dim)).to(device), None, truncation_psi=psi, noise_mode='const')
        w_samples = G.mapping(z,  None, truncation_psi=psi)
        print ('mapped w_samples shape:', w_samples.shape)
        print (w_samples)
        w_og = w_samples # this is the original (1,18,512) from Z>W network
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)
        w_avg = np.mean(w_samples, axis=0, keepdims=True)
    else: # w is provided  as ws, just use it
        # save the image for that w
        w_samples = torch.tensor(ws, device=device)
        print ('w_samples:', w_samples.shape)
        print (w_samples)
        synth_images = G.synthesis(w_samples, noise_mode='const') # og is const. random sometimes helps?
        w_og = w_samples # this is the original (1,18,512) from Z>W network
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)
        print ('loaded w_samples:', w_samples.shape)
        w_avg = np.mean(w_samples, axis=0, keepdims=True)
    #w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
    # this value is picked up empirically
    w_std = 10 # MK3: let's keep it real. it's the average square Euclidean distance to the center of W. 
    #obviously when we start not from center but from nearby cluster it's shorter

    # end of commented out to test kmeans
    synth_images_save = (synth_images + 1) * (255/2)        
    synth_images_save = synth_images_save.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_images_save, 'RGB').save('step-in.png')

    if imgc:
        print("image derived from initial w:")
        imgcat(PIL.Image.fromarray(synth_images_save).resize((256,256), PIL.Image.BICUBIC))


    # reset values for cropping (actually we don't need to do it since we already sampled the very same network but in case we've cropped 10%)
    x = None
    y = None
    r = None
    b = None

    if not stylemix:
        # option A: w is uniform, i.e. the same [1,1,512] is broadcast to all layers [1,18,512]
        w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) #
        w_out = torch.zeros( [num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
        print('w_opt shape:', w_opt.shape)
        print('w_out shape:', w_out.shape)
        optimizer = torch.optim.Adam( [w_opt], betas=(0.9, 0.999), lr=initial_learning_rate)
        print('optimizer: w_opt. shape of w_opt:',w_opt.shape)
    else:
        # option B: stylemix. optimizable shape is [1,18,512]
        w_optfull = torch.tensor(w_og, dtype=torch.float32, device=device, requires_grad=True) 
        w_outfull = torch.zeros([num_steps] + list(w_optfull.shape[1:]), dtype=torch.float32, device=device)
        print('w_optfull shape:', w_optfull.shape)
        print('w_outfull shape:', w_outfull.shape)
        optimizer = torch.optim.Adam( [w_optfull], betas=(0.9, 0.999), lr=initial_learning_rate)
        print('optimizer: w_optfull. shape:',w_optfull.shape)

    # Descend.

    for step in range(num_steps):
        # noise schedule. kept from OG
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        # lr schedule 
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # initialize w_noise, i.e. the gaussian noise that is being added to w
        # torch. radn_like: returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1
        # i.e. w_noise is filled with N(0,1)?
        if stylemix:
            w_noise = torch.randn_like(w_optfull) * w_noise_scale
            ws = w_optfull + w_noise # we feed gradable (1,18,512) - need to add noise there
        else:
            w_noise = torch.randn_like(w_opt) * w_noise_scale
            ws = (w_opt + w_noise).repeat([w_batch_size, G.mapping.num_ws, 1]) # we convert (1,1,512) to (1,18,512)

        # and do the synthesis.
        # the OG mode uses noise_mode=const since optimizing/regularizing noise buffers are a part of design
        synth_images = G.synthesis(ws, noise_mode='random') # noise mode const or random

        #into = synth_images

        geb = synth_images[:, :, 0:600, 0:300]
        geba = synth_images[:, :, 0:600, 301:725]
        gebal = synth_images[:, :, 0:600, 726:1023]
        gebi = synth_images[:, :, 601:1023, 0:300]
        gebil = synth_images[:, :, 601:1023, 301:725]
        gebo = synth_images[:, :, 601:1023, 726:1023]


        geb = torch.nn.functional.interpolate(geb, (slit,slit), mode='area') #OG PB
        geb = (torch.clamp(geb, -1, 1) + 1) * (255/2)
        geb = vgg16(geb, resize_images=False, return_lpips=True) #.mean()

        geba = torch.nn.functional.interpolate(geba, (slit,slit), mode='area') #OG PB
        geba = (torch.clamp(geba, -1, 1) + 1) * (255/2)
        geba = vgg16(geba, resize_images=False, return_lpips=True) #.mean()

        gebal = torch.nn.functional.interpolate(gebal, (slit,slit), mode='area') #OG PB
        gebal = (torch.clamp(gebal, -1, 1) + 1) * (255/2)
        gebal = vgg16(gebal, resize_images=False, return_lpips=True) #.mean()

        gebi = torch.nn.functional.interpolate(gebi, (slit,slit), mode='area') #OG PB
        gebi = (torch.clamp(gebi, -1, 1) + 1) * (255/2)
        gebi = vgg16(gebi, resize_images=False, return_lpips=True) #.mean()

        gebil = torch.nn.functional.interpolate(gebil, (slit,slit), mode='area') #OG PB
        gebil = (torch.clamp(gebil, -1, 1) + 1) * (255/2)
        gebil = vgg16(gebil, resize_images=False, return_lpips=True) #.mean()

        gebo = torch.nn.functional.interpolate(gebo, (slit,slit), mode='area') #OG PB
        gebo = (torch.clamp(gebo, -1, 1) + 1) * (255/2)
        gebo = vgg16(gebo, resize_images=False, return_lpips=True) #.mean()


        loss =  \
            1.0 * (F.relu(((bes - geb).square().sum()*6) ** 2 - 0.1) + F.relu(((besa - geba).square().sum()*6) ** 2 - 0.1) + F.relu(((besal - gebal).square().sum()*6) ** 2 - 0.1)) +  \
            1.0 * (F.relu(((besi - gebi).square().sum()*6) ** 2 - 0.1) + F.relu(((besil - gebil).square().sum()*6) ** 2 - 0.1) + F.relu(((beso - gebo).square().sum()*6) ** 2 - 0.1))

        # step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # progress every 100 steps
        if imgc and (step == 0 or step % 100 == 0): 
            print("intermed results:")
            synth_images_save = (synth_images + 1) * (255/2) 
            synth_images_save = synth_images_save.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            #imgcat(PIL.Image.fromarray(synth_images_save).resize((512,512), PIL.Image.BICUBIC))  
            #draw = ImageDraw.Draw(PIL.Image.fromarray(synth_images_save))
            #draw.line((0, 512) + (1023, 512) , fill=128)
            #draw.line((0, im.size[1], im.size[0], 0), fill=128)
            imgcat( PIL.Image.fromarray(synth_images_save).resize((256,256), PIL.Image.BICUBIC) )    

        #print(f'step {step+1:>4d}/{num_steps}:  loss {float(loss):<5.4f} ','lr', lr, f'noise scale: {float(w_noise_scale):<5.6f}',f'arbitrage: {float(arbitrage):<5.6f}',f'proctorage: {float(proctorage):<5.6f}')
        print(f'step {step+1:>4d}/{num_steps}:  loss {float(loss):<5.4f} ','lr', lr, f'noise scale: {float(w_noise_scale):<5.6f}' )
        if not stylemix:
            w_out[step] = w_opt.detach()[0]
        else:
            w_outfull[step] = w_optfull.detach()

    # finished
    if not stylemix:
        return w_out.repeat([1, G.mapping.num_ws, 1])
    else:
        return w_outfull

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=100, show_default=True)
@click.option('--seed',                   help='Initial image seed', type=int, default=232322, show_default=True)
@click.option('--autoseed',               help='Guess the seed based on text prompt', type=bool, default=False, show_default=True)
@click.option('--w',                      help='Do not use seed but load w from a file', type=str, metavar='FILE')
@click.option('--lr',                     help='Adam learning rate', type=float, required=False, default=0.02)
@click.option('--psi',                    help='Truncation psi for initial image', type=float, required=False, default=0.65)
@click.option('--inf',                    help='Initial noise factor', type=float, required=False, default=0.02) # og was 0.02
@click.option('--nf',                     help='Noise floor', type=float, required=False, default=0.00)
@click.option('--noise-opt',              help='... (the name is misleading!) when set to false, the entire stack of w (1,18,512) is getting optimized in stylemix mode, instead of (1,1,512) that get broadcasted', type=bool, required=False, default=True)
@click.option('--stylemix',              help='Optimize all layers of w instead of just one', type=bool, required=False, default=False)
@click.option('--anime',              help='Do not attempt face detection, use manual hack instead', type=bool, required=False, default=True)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--save-ws',                help='Save intermediate ws', type=bool, default=False, show_default=True)
@click.option('--portrait',               help='Limit CLIP image sample to face', type=bool, default=False, show_default=True)
@click.option('--autoportrait',           help='When used with portrait, look for a face using FFHQ face detector insterad of just the center of the upper half', type=bool, default=False, show_default=True)
@click.option('--multimode',              help='Use two CLIP ratings to guide! Needs both --image and --text', type=bool, default=False, show_default=True)
@click.option('--kmeans',                 help='Use kmeans Z sampling for autoseed', type=bool, default=False, show_default=True)
#@click.option('--image',                  help='target image. when using --portrait must be 512x512', required=True, default=None,  multiple=True)
@click.option('--image',                  help='target image. when using --portrait must be 512x512', required=True, default=None)
@click.option('--imgc',              help='Use imgcat to display images in terminal (does not work with screen/tmux)', type=bool, required=False, default=False)
@click.option('--vgg',              help='Use VGG instead of CLIP. For images only, no multimode', type=bool, required=False, default=False)


def run_capture(
    network_pkl: str,
    outdir: str,
    save_video: bool,
    save_ws: bool,
    seed: int,
    num_steps: int,
    autoseed: bool,
    anime: bool,
    image: str,
    lr: float,
    inf: float,
    nf: float,
    w: str,
    psi: float,
    vgg: bool,
    imgc: bool,
    kmeans: bool,
    multimode: bool,
    stylemix: bool,
    noise_opt: bool,
    autoportrait: bool,
    portrait: bool

):
    """Entrap a portrait in latent space with VGG 

    Example:

    \b
    python3 capture-vgg.py --network network-snapshot-veryfinal-feb26-FID195.pkl \\ 
    --outdir capture-vgg-1404 --image coctail-dress-crop-512.jpg --imgc true --num-steps 1000  
    
    """

    #just for test - does not help w/determinism
    #np.random.seed(1)
    #torch.manual_seed(1)

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
    projected_w_steps = capture(
        G,
        num_steps=num_steps,
        device=device,
        initial_learning_rate = lr,
        psi = psi,
        seed = seed,
        initial_noise_factor = inf,
        noise_floor = nf,
        ws = ws,
        Image = image,
        autoseed = autoseed,
        anime = anime,
        portrait = portrait,
        kmeans = kmeans,
        vgg = vgg,
        imgc = imgc,
        stylemix = stylemix,
        multimode = multimode,
        autoportrait = autoportrait,
        noise_opt = noise_opt
    )

    # save video
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/out-{hashname}.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/out-{hashname}.mp4"')
        step = 0
        for projected_w in projected_w_steps:
            step += 1
            if step % 4 == 0:
            #print('projected w size', projected_w.shape)
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
    np.savez(f'{outdir}/w-{hashname}-initial.npz', w=projected_w_steps[0].unsqueeze(0).cpu().numpy())

    print ('final w:', projected_w.unsqueeze(0).cpu().numpy())

    # save params
    with open(f'{outdir}/params-{hashname}.txt', 'w') as outfile:
        json.dump(params, outfile)

    if imgcat:
        # experimental - display result w/imgcat
        imc = PIL.Image.fromarray(synth_image).resize((256,256), PIL.Image.BICUBIC)
        imgcat(imc)

if __name__ == "__main__":
    run_capture() 
