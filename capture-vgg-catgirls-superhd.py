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
# NOTE this particular script was successfull for the task of encoding (capturing?) the girl in coctail dress
# command line needs to be found... but sth like VGG, stylemix on, noise/lr 0.01/0.02 or 0.02 squared, 1000-3000 steps
# TAKE ATTN of multiple penalizers and design of their crop input 
#
# basic concept:
# - optimize W+ based on matching the full frame (100% of image // face focused baded on technique) classifier "Arbitrator"
#   and face focused "Proctor". Also there's "Custodian" which makes sure that some part of w generated image gets preserved
#
#
# Saved as capture-vgg.py. CLIP stuff, noise buffers opt removed in order to clean up the code. -l4rz, 14/04/2021
#
# python3 approach-mk4-wface.py --network network-snapshot-veryfinal-feb26-FID195.pkl --outdir capture-vgg-1404 
# --image coctail-dress-crop-512.jpg --imgc true --num-steps 2000 --lr 0.02 --inf 0.02 --stylemix true --w coctail/w-db1b63229a3a10bf81efdb8f5f6870078c9d4a70-final.npz
#
#
# to combine video and image
# ffmpeg -i bogdanoff5.jpg -i bogdanonff/out-4b07c14c692e5f2cfeb771b9205dbdb2056a2f59.mp4 -filter_complex " nullsrc=size=1024x512 [base];[0:v] setpts=PTS-STARTPTS, scale=512x512 [upperleft]; [1:v] setpts=PTS-STARTPTS, scale=512x512 [upperright]; [base][upperleft] overlay=shortest=1 [tmp1]; [tmp1][upperright] overlay=x=512" -c:v libx264 output5.mp4
# 
# to crop
# ffmpeg -i in.mp4 -filter:v "crop=out_w:out_h:x:y" out.mp4
# ffmpeg -i input.mp4 -vf scale=320:240,setsar=1:1 output.mp4
# combine vidyas
# ffmpeg -i left.mp4 -i right.mp4 -filter_complex hstack output.mp4
#
#

# TODO: add W saved on 50%, 75%, 85%, 85% and final iters

def ffhq_facecrop(img, detector, shape_predictor):
    face_landmarks = []
    dets = detector(img, 1)
    for detection in dets:
        try:
            face_landmarks = [(item.x, item.y) for item in shape_predictor(img, detection).parts()]
            #print(face_landmarks)
        except:
            print("Exception in get_landmarks()! The image does not contain a detectable face")
    eye_left = np.mean(face_landmarks[36:42], axis=0)
    eye_right = np.mean(face_landmarks[42:48], axis=0)
    ipd = eye_right - eye_left
    eye_avg   = (eye_left + eye_right) * 0.5
    ma = np.mean((face_landmarks[48], face_landmarks[54]), axis=0)
    eye_to_mouth = ma  - eye_avg
    # Choose oriented crop rectangle.
    x = ipd - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*ipd) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    x *= 0.75 # x_scale. use for zoomed in. og 1.0, 0.5 gives zoom in on facial features
    y = np.flipud(x) * [-1, 1] # 1 y_scale
    c = eye_avg + eye_to_mouth * 0.1 # 0.1 is the OG em_scale.
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2
    #print (qsize)
    x =  int(np.floor(min(quad[:,0])))  
    y =  int(np.floor(min(quad[:,1])))
    right =  int(np.ceil(max(quad[:,0]))) 
    bottom =  int(np.ceil(max(quad[:,1]))) 
    # fix negative values :smug_pepe:
    if x < 0 or y < 0:
        delta = max(-1 * x, -1 * y) + 1 # likely y
        x = x + delta
        y = y + delta
        right = right - delta
        bottom = bottom - delta
    return x,y,right,bottom


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

    # initialize classifiers

    # WARNING
    # we assume driving frame is 512px

    # some prereqs
    with PIL.Image.open(Image) as img:
        width, height = img.size
        if width == 1024 or height == 1024:
            ssf = 2

    '''
         sol   |   soles
      ---------+----------
       solesra |   solir
    '''

    img = np.array(PIL.Image.open(Image).convert('RGB'), dtype=np.uint8)

    bes = img[0:256, 128:128+256, :  ] 
    print(Image,'shape for bes',bes.shape)
    bes = np.array(PIL.Image.fromarray(bes).resize((slit,slit), PIL.Image.LANCZOS), dtype=np.uint8)
    bes = torch.tensor(bes.transpose([2, 0, 1]), device=device).unsqueeze(0).to(device).to(torch.float32)
    bes = vgg16(bes, resize_images=False, return_lpips=True)
    print(Image,'tensor shape for bes',bes.shape)

    besa = img[0:256, 384:384+256, :  ] 
    print(Image,'shape for besa',besa.shape)
    besa = np.array(PIL.Image.fromarray(besa).resize((slit,slit), PIL.Image.LANCZOS), dtype=np.uint8)
    besa = torch.tensor(besa.transpose([2, 0, 1]), device=device).unsqueeze(0).to(device).to(torch.float32)
    besa = vgg16(besa, resize_images=False, return_lpips=True)
    print(Image,'tensor shape for besa',besa.shape)

    besal = img[0:256, 640:640+256, :  ] 
    print(Image,'shape for besal',besal.shape)
    besal = np.array(PIL.Image.fromarray(besal).resize((slit,slit), PIL.Image.LANCZOS), dtype=np.uint8)
    besal = torch.tensor(besal.transpose([2, 0, 1]), device=device).unsqueeze(0).to(device).to(torch.float32)
    besal = vgg16(besal, resize_images=False, return_lpips=True)
    print(Image,'tensor shape for besal',besal.shape)

    besi = img[256:512+24, 128:128+256, :  ] 
    print(Image,'shape for besi',besi.shape)
    besi = np.array(PIL.Image.fromarray(besi).resize((slit,slit), PIL.Image.LANCZOS), dtype=np.uint8)
    besi = torch.tensor(besi.transpose([2, 0, 1]), device=device).unsqueeze(0).to(device).to(torch.float32)
    besi = vgg16(besi, resize_images=False, return_lpips=True)
    print(Image,'tensor shape for bes',bes.shape)

    besil = img[256:512+24, 384:384+256, :  ] 
    print(Image,'shape for besil',besil.shape)
    besil = np.array(PIL.Image.fromarray(besil).resize((slit,slit), PIL.Image.LANCZOS), dtype=np.uint8)
    besil = torch.tensor(besil.transpose([2, 0, 1]), device=device).unsqueeze(0).to(device).to(torch.float32)
    besil = vgg16(besil, resize_images=False, return_lpips=True)
    print(Image,'tensor shape for besil',besil.shape)

    beso = img[256:512+24, 640:640+256, :  ] 
    print(Image,'shape for beso',beso.shape)
    beso = np.array(PIL.Image.fromarray(beso).resize((slit,slit), PIL.Image.LANCZOS), dtype=np.uint8)
    beso = torch.tensor(beso.transpose([2, 0, 1]), device=device).unsqueeze(0).to(device).to(torch.float32)
    beso = vgg16(beso, resize_images=False, return_lpips=True)
    print(Image,'tensor shape for beso',beso.shape)

     
    print('done initializing classifiers')
    # done initializing classifiers


    # autoseed - we guess the seed
    if autoseed and not kmeans:
        print(f'Guessing the best seed using {autoseed_samples} samples from random Z seeds')
        pod = np.full((autoseed_samples),0)
        for i in range(autoseed_samples):
            seed = randint(0,500000)
            pod[i] = seed
        staffel = []
        for i in range(autoseed_samples):
            snap = G(torch.from_numpy(np.random.RandomState(pod[i]).randn(1,G.z_dim)).to(device), None, truncation_psi=psi, noise_mode='const')
            # let's try to put it there too
            #print('ss',snap.shape)
            if portrait:
                # assuming the resolution is 1024^x, use the center of the upper half of image
                # snap = snap[y:y+h, x:x+w]
                snap = snap[:, :, 0:0 + 512, 256:256+512]
            snap = torch.nn.functional.interpolate(snap, (slit,slit), mode='bilinear', align_corners=True)
            eignung = int(   torch.cosine_similarity(features, perceptor.encode_image(snap), dim = -1).cpu().detach().numpy() * 1000)
            staffel.append( (pod[i], eignung )) 

        staffel = sorted(staffel,key=lambda x:(-x[1]))
        np_staffel= np.array(staffel, int)
        staffel_avg = np.mean(np_staffel, axis=0)[1]
        staffel_std = np.std(np_staffel, axis=0)[1]
        for i in range(autoseed_samples):
            if abs(np_staffel[i][1] - staffel_avg) < 1.7 * staffel_std: #first non-outlier
                seed = np_staffel[i][0]
                break
        print (f'Top guess {staffel[i][0]}')
    # kmeans - we guess the best w
    # rewrite the code, it sucks balls
    if autoseed and kmeans:
        kmeans_clusters = 64
        kmeans_pod_file = 'kmeans-pod-384.pt'
        print ('loading kmeans pod from file', kmeans_pod_file)
        pod = torch.load(kmeans_pod_file)
        print('loaded pod shape', pod.shape)
        verboten = [45,47,55,61,71,108,112,114,129,138,161,164,165,174,201,217,236,245,265,312,328,378,382] # w's from the pod that should be avoided. eg malformed or undesirable
        staffel = []
        # copypaste the staffel code, check those 
        #staffel = []
        # rewrite it better, with sorting and outliers
        #preveignung = 0
        ws_c = pod[0]
        ws_cc = pod[0]
        #besti = 0
        # todo
        # somehow this scoring model has a strong preference for zoomed in faces
        print(f'Scoring the best latent of {kmeans_clusters} ...')
        for i in tqdm(range(kmeans_clusters)):
        #for i in range(kmeans_clusters):
            # convert W to 18 dim
            #print('podi', pod[i].shape)
            if i in verboten:
                continue
            wt = pod[i].repeat([1, G.mapping.num_ws, 1])
            #print('wt', wt.shape)
            #snap = G.synthesis(wt, noise_mode='const')
            snap = G.synthesis(wt, noise_mode='const') 
            img = (snap + 1) * (255/2) 
            img = img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            if portrait:
                #snap = snap[:, :, 0:0 + 512, 256:256+512]
                # try to use ffhq  crop to better ffhq face
                xyrb = ffhq_facecrop(img,detector,shape_predictor)
                x = xyrb[0]
                y = xyrb[1]
                r = xyrb[2]
                b = xyrb[3]
                # hack - source images contain no neck but ffhq crop does, so we take 10% off
                # b = xyrb[3] - int((xyrb[3] - xyrb[1]) * (10 / 100)) # no hacks
                snap = snap[:, :, y:b, x:r]
            snap = torch.nn.functional.interpolate(snap, (slit,slit), mode='bilinear', align_corners=True)
            eignung = int(   torch.cosine_similarity(features, perceptor.encode_image(snap), dim = -1).cpu().detach().numpy() * 1000)
            staffel.append( (i, eignung ))

        staffel = sorted(staffel,key=lambda x:(-x[1]))
        np_staffel= np.array(staffel, int)
        staffel_avg = np.mean(np_staffel, axis=0)[1]
        staffel_std = np.std(np_staffel, axis=0)[1]
        for i in range(autoseed_samples):
            if abs(np_staffel[i][1] - staffel_avg) < 1.7 * staffel_std: #first non-outlier
                seed = np_staffel[i][0]
                break
        print (f'Top guess {staffel[i][0]}')
        ws_cc = pod[ staffel[i][0] ].repeat([1, G.mapping.num_ws, 1])
        ws = ws_cc.cpu().detach().numpy()

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
        imgcat(PIL.Image.fromarray(synth_images_save).resize((256,256), PIL.Image.BILINEAR))

    # Now say we want to preserve some part of the initial-w-derived image, like border (or tits)
    #
    
    subject = np.array(PIL.Image.open('step-in.png').convert('RGB'), dtype=np.uint8)
    #subject = subject[0:299, 128:384, : ] # hard focus. the issue is that we need to use exacly same relative coords while cropping trainign smaple
    #subject = subject[512:1023, 0:1023, : ] # bottom half
    #subject = subject[512+48:1023, 0:1023, : ] # bottom half
    #subject = subject[512:1023, 100:1024-100, : ] # bottom half
    subject = subject[512:1023, 100:1024-100, : ] # bottom half

    # CATGIRL MODE: scale it down 90%
    if catgirl:
        blanked = PIL.Image.new("RGB", (1024, 512-12), (128,128,128)) # w,h
        resized = PIL.Image.fromarray(subject).resize((930, 512-12), PIL.Image.BICUBIC) # w,h
        blanked.paste(resized, (47,0) ) # move it left 50 pixels
        subject = np.array( blanked  , dtype=np.uint8)
    # end of catgirl
    if imgc:
        print('subject:')
        imgcat(PIL.Image.fromarray(subject).resize((512,512), PIL.Image.BICUBIC)) 
    subject = np.array(PIL.Image.fromarray(subject).resize((slit,slit), PIL.Image.BICUBIC), dtype=np.uint8)
    subject = torch.tensor(subject.transpose([2, 0, 1]), device=device).unsqueeze(0).to(device).to(torch.float32)
    print(Image,'tensor shape for subject',subject.shape)
    custodian = vgg16(subject, resize_images=False, return_lpips=True)
    

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

        # 2. compute proctor distance 
        #into = synth_images[:, :, 0:600, 255:255+512]  # focus on top part; match sample cropped [0:299, 128:384, : ] 
        # into = synth_images[:, :, 0:511+24, 128:128+768] # for hanekawa
        into = synth_images[:, :, 0:512, 128:128+768]
        if imgc and (step == 0 or step % 20 == 0): 
            print("proctoree:")
            synth_images_save = (into + 1) * (255/2) 
            synth_images_save = synth_images_save.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            imgcat(PIL.Image.fromarray(synth_images_save).resize((256,256), PIL.Image.BILINEAR))

        geb = synth_images[:, :, 0:256, 128:128+256]
        geba = synth_images[:, :, 0:256, 384:384+256]
        gebal = synth_images[:, :, 0:256, 640:640+256]
        gebi = synth_images[:, :, 256:512+24, 128:128+256]
        gebil = synth_images[:, :, 256:512+24, 384:384+256]
        gebo = synth_images[:, :, 256:512+24, 640:640+256]

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



        # compute custodian distance 
        #into = synth_images[:, :, 512:1023, 0:1023]  # bottom half
        #into = synth_images[:, :, 512+48:1023, 0:1023]  # bottom half
        #into = synth_images[:, :, 512:1023, 100:1024-100]  # bottom half
        into = synth_images[:, :, 512+12:1023, 100:1024-100]  # bottom half
        if imgc and (step == 0 or step % 20 == 0): 
            print("custodee:")
            synth_images_save = (into + 1) * (255/2) 
            synth_images_save = synth_images_save.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            imgcat(PIL.Image.fromarray(synth_images_save).resize((256,256), PIL.Image.BILINEAR))     
        into = torch.nn.functional.interpolate(into, (slit,slit), mode='area') #OG PB
        into = (torch.clamp(into, -1, 1) + 1) * (255/2)
        custodee = vgg16(into, resize_images=False, return_lpips=True) #.mean()
        custody =  (custodian - custodee).square().sum()
        custody *= 6
        custody = F.relu(custody*custody - 0.1) # 0.1 min thresh

        loss =  5 * custody +  \
            2.0 * (F.relu(((bes - geb).square().sum()*6) ** 2 - 0.1) + F.relu(((besa - geba).square().sum()*6) ** 2 - 0.1) + F.relu(((besal - gebal).square().sum()*6) ** 2 - 0.1)) +  \
            2.0 * (F.relu(((besi - gebi).square().sum()*6) ** 2 - 0.1) + F.relu(((besil - gebil).square().sum()*6) ** 2 - 0.1) + F.relu(((beso - gebo).square().sum()*6) ** 2 - 0.1))

        # step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # progress every 100 steps
        if imgc and (step == 0 or step % 100 == 0): 
            print("intermed results:")
            synth_images_save = (synth_images + 1) * (255/2) 
            synth_images_save = synth_images_save.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            #imgcat(PIL.Image.fromarray(synth_images_save).resize((512,512), PIL.Image.BILINEAR))  
            #draw = ImageDraw.Draw(PIL.Image.fromarray(synth_images_save))
            #draw.line((0, 512) + (1023, 512) , fill=128)
            #draw.line((0, im.size[1], im.size[0], 0), fill=128)
            imgcat( PIL.Image.fromarray(synth_images_save) )    

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
        imc = PIL.Image.fromarray(synth_image).resize((256,256), PIL.Image.BILINEAR)
        imgcat(imc)

if __name__ == "__main__":
    run_capture() 
