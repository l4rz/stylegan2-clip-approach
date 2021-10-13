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

# Mark 3 - return to the OG concepts.
# Mark 4 - add VGG
# Mark 5 - add "frozenbottom" (vgg classifier that penalizes changes of the bottom half), add scale factor. BUGS: cam fails with 512px
# 
#
# to combine video and image
# ffmpeg -i bogdanoff5.jpg -i bogdanonff/out-4b07c14c692e5f2cfeb771b9205dbdb2056a2f59.mp4 -filter_complex " nullsrc=size=1024x512 [base];[0:v] setpts=PTS-STARTPTS, scale=512x512 [upperleft]; [1:v] setpts=PTS-STARTPTS, scale=512x512 [upperright]; [base][upperleft] overlay=shortest=1 [tmp1]; [tmp1][upperright] overlay=x=512" -c:v libx264 output5.mp4

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
    x *= 0.85 # was 1 x_scale
    y = np.flipud(x) * [-0.85, 0.85] # was 1 y_scale
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
    return x,y  ,right,bottom  #l4rz hack


def approach(
    G,
    *,
    num_steps                  = 100,
    w_avg_samples              = 10000, # 10k is ok. 30k is better.
    initial_learning_rate      = 0.02,  
    initial_noise_factor       = 0.05,  
    noise_floor                = 0.00, # keep it OG 
    psi                        = 0.8,
    noise_ramp_length          = 0.75, # meaining during 75/100 operations w noise goes to zero
    regularize_noise_weight    = 10000, # was 1e5
    seed                       = 69097, 
    noise_opt                  = None,  # default is false
    lr_rampdown_length          =0.25, # means final 25% of iters we commence rampdown
    lr_rampup_length          = 0.05,

    portrait                   = None, #assumed 1024px
    #antiportrait                = None, 
    autoportrait                = None, # sth does not work with anime models
    autoseed                   = False, 
    cam                   = True,  # counteradversarial measures
    imgc                   = None,  # display some images in terminal, does not work with tmux/screen tho
    autoseed_samples           = 100,
    Image                      = None, 
    anime                       = None, # anime means don't try to ffhq crop the source image
    multimode                      = True,  # use both image and text - MUST BE REDONE W/VECTOR MATH
    frozenbottom               = None, # only for portraits!
    frozentop               = None, # only for portraits!
    ws                         = None,
    pmul                       = 30, # multiplier for cosine_similarity,dunno why 30 works lol
    apper_range                 = 16, # when using counteradversarial, a number of random crops
    text                       = 'a high quality image', 
    kmeans                       = True, # try to autoseed using kmean clustering
    stylemix                    = None, # was called noise_opt. now for the avoidance of doubt we call it stylemix
    vgg                         = None, # use VGG instead of CLIP. works for images only
    sf = 1, # slash factor for various hardcoded landmarks. set to 2 for 512px instead of 1024px. also affects the input images in portrait mode 
    w_batch_size                = 1, # number of images to synthesize and average during each step
    device: torch.device
):



    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    lr = initial_learning_rate

    # Load the perceptor
    print('Loading perceptor')
    if not vgg: # CLIP
        #perceptor, preprocess = clip.load('ViT-B/32', jit=True)
        #clipres = 224
        # WARNING! in the majority of our faceswap experiments we used RN50x4
        perceptor, preprocess = clip.load('RN50x4', jit=True)
        clipres = 288
        perceptor = perceptor.eval()
        tx = clip.tokenize(text)
        whispers = perceptor.encode_text(tx.cuda()).detach().clone()
        features = whispers
        print('Perceptor: text:', text)
    if vgg or frozenbottom or frozentop:
        # https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt
        vgg16 = torch.jit.load('vgg16.pt').eval().to(device)
        if vgg:
            autoseed = None
            kmeans = None
            multimode = None
            cam = None
        #initial_noise_factor       = 0.02
        w_batch_size = 1
        slit = 224 # vgg res
        # Baseline for VGG
        # python3 approach-mk4.py --network network-snapshot-veryfinal-feb26-FID195.pkl --outdir vgg
        # --noise-opt false --vgg true --imgc true --portrait true --autoportrait true --lr 0.02 --num-steps 300 --image m3.jpg --inf 0.01
        # ends with distance = 2.7
        # --inf 0.0075 slightly worse distance (2.9)
        # --inf 0.015 : 3.05
        # inf 0.01 lr 0.02 500 steps = 2.60
        # nf 0.01 lr 0.02 200 steps = 3.00
        #
        # But! stylemix = true nf 0.01 lr 0.02 200 steps gives 2.00
        # 0.0075 lr 0.015 200 steps == 2.0x
        # 0.0050 lr 0.015 200 steps == 2.0x
        # 0.0050 lr 0.010 200 steps == 2.4x
        # 0.0025 lr 0.015 200 steps == 2.15
        # 0.0001 lr 0.015 200 steps == 2.08




    print('Autoportrait:', autoportrait)
    if autoportrait:
        portrait = True # assert
        print('Autoportrait enabled, asserting portrait mode on')

    if kmeans:
        autoseed = True # assert
        print('Kmeans enabled, asserting autoseed mode on')

    if Image or portrait: # initialize face detector, gotta be used later
        detector = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    if Image and not vgg:
        print('using target image(s) as classifier with CLIP', Image)
        if anime:
            print('Anime mode is on; not attempting to crop the target image(s)')
        vibestaffel = []
        for vibeimg in Image:
            print('adding', vibeimg)
            img = np.array(PIL.Image.open(vibeimg).convert('RGB'), dtype=np.uint8)
            img = img[192:1024, 0:1024, : ]  ## EXPERIMENT
            if not anime:
                if autoportrait:
                    print(".")
                    xyrb = ffhq_facecrop(img,detector,shape_predictor)
                    img = img[xyrb[1]:xyrb[3], xyrb[0]:xyrb[2], : ]
                if portrait and not autoportrait:
                    img = img[0:int(512/sf), int(256/sf):int((256+512)/sf), : ] 

            if imgc:
                imgcat(PIL.Image.fromarray(img).resize((256,256), PIL.Image.BILINEAR))
            img = preprocess(PIL.Image.fromarray(img)).unsqueeze(0).to(device)
            vibestaffel.append(img)
        vibestaffel = torch.cat(vibestaffel)
        vibes = perceptor.encode_image(vibestaffel).detach().clone()
        print('vs', vibes.shape)
        vibes = torch.mean(vibes,0,keepdim=True)
        print('final features shape', vibes.shape)
        features = vibes
  
    if Image and vgg:
        print('using target image(s) as classifier with VGG', Image)
        if anime:
            print('Anime mode is on; not attempting to crop the target image(s)')
        vibestaffel = []
        for vibeimg in Image:
            print('adding', vibeimg)
            #img = np.array(PIL.Image.open(vibeimg).convert('RGB').filter(PIL.ImageFilter.SHARPEN), dtype=np.uint8)
            img = np.array(PIL.Image.open(vibeimg).convert('RGB'), dtype=np.uint8)

            if not anime: # assuming portrait
                if autoportrait:
                    xyrb = ffhq_facecrop(img,detector,shape_predictor)
                    img = img[xyrb[1]:xyrb[3], xyrb[0]:xyrb[2], : ]
                else:
                    # img = img[0:192, 256:256+512, : ] experimental
                    #img = img[0:512, 256:256+512, : ]
                    img = img[0:int(512/sf), int(256/sf):int((256+512)/sf), : ] 
            if imgc:
                imgcat(PIL.Image.fromarray(img).resize((256,256), PIL.Image.LANCZOS))
            

            ximg = np.array(PIL.Image.fromarray(img).resize((slit,slit), PIL.Image.LANCZOS), dtype=np.uint8)
            ximg = torch.tensor(ximg.transpose([2, 0, 1]), device=device).unsqueeze(0).to(device).to(torch.float32)
            print('---',ximg.shape)
            vibestaffel.append(ximg)

            # 
            #ximg = (ximg + 1) * (255/2) 
            ximg = ximg.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            imgcat(PIL.Image.fromarray(ximg))
            print('===')

        vibestaffel = torch.cat(vibestaffel)
        vibes = vgg16(vibestaffel, resize_images=False, return_lpips=True)
        #vibes = perceptor.encode_image(vibestaffel).detach().clone()
        print('vs', vibes.shape)
        vibes = torch.mean(vibes,0,keepdim=True)
        print('final features shape', vibes.shape)
        features = vibes

        # experimental
        '''
        vibestaffel2 = []
        for vibeimg2 in Image:
            img2 = np.array(PIL.Image.open(vibeimg).convert('RGB'), dtype=np.uint8)
            img2 = img2[256:511, 256:511, : ]
            print(img2.shape)
            ximg2 = np.array(PIL.Image.fromarray(img2).resize((224,224), PIL.Image.LANCZOS), dtype=np.uint8)
            ximg2 = torch.tensor(ximg2.transpose([2, 0, 1]), device=device).unsqueeze(0).to(device).to(torch.float32)
            vibestaffel2.append(ximg2)
        vibestaffel2 = torch.cat(vibestaffel2)
        vibes2 = vgg16(vibestaffel2, resize_images=False, return_lpips=True)
        vibes2 = torch.mean(vibes2,0,keepdim=True)
        '''




  
    if multimode:
        print('Perceptor: multimode with text:', text)


    print('Frozentop', frozentop, 'frozenbottom', frozenbottom)

    # end of retarded

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
                # 
                snap = snap[:, :, 0: int(512/sf), int(256/sf):int((256+512)/sf) ]
            snap = torch.nn.functional.interpolate(snap, (clipres,clipres), mode='bilinear', align_corners=True)
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
            snap = torch.nn.functional.interpolate(snap, (clipres,clipres), mode='bilinear', align_corners=True)
            eignung = int(   torch.cosine_similarity(features, perceptor.encode_image(snap), dim = -1).cpu().detach().numpy() * 1000)
            staffel.append( (i, eignung ))

            # wt is [1,512]
            # pod is [kmeans_clusters, 1, 512]

            #if eignung > preveignung:
            #    ws_cc = ws_c
            #    ws_c = wt
            #    preveignung = eignung
            #    besti = i
 

        staffel = sorted(staffel,key=lambda x:(-x[1]))
        np_staffel= np.array(staffel, int)
        staffel_avg = np.mean(np_staffel, axis=0)[1]
        staffel_std = np.std(np_staffel, axis=0)[1]
        for i in range(autoseed_samples):
            if abs(np_staffel[i][1] - staffel_avg) < 1.7 * staffel_std: #first non-outlier
                seed = np_staffel[i][0]
                break

        #ws = ws_c.repeat([1, G.mapping.num_ws, 1])

        #np.savez('kmeans-pod.npz', pod=pod.unsqueeze(0).cpu().detach().numpy())
        print (f'Top guess {staffel[i][0]}')
        ws_cc = pod[ staffel[i][0] ].repeat([1, G.mapping.num_ws, 1])


        #print('best seed in pod pt', besti)
        #print('ws_cc shape', ws_cc.shape)
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

    w_std = 2 # ~9.9 for portraits network. should compute if using median median

    w_std = 10 # MK3: let's keep it real. it's the average square Euclidean distance to the center of W. 
    #obviously when we start not from center but from nearby cluster it's shorter

    # end of commented out to test kmeans
    synth_images_save = (synth_images + 1) * (255/2)        
    synth_images_save = synth_images_save.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_images_save, 'RGB').save('step-in.png')

    if imgc:
        print("image derived from initial w:")
        imgcat(PIL.Image.fromarray(synth_images_save).resize((256,256), PIL.Image.BILINEAR))

    # holdstill frozenbottom feature
    if frozentop:
        subject = np.array(PIL.Image.open('step-in.png').convert('RGB'), dtype=np.uint8)
        # frozen top - cat ears
        subject = subject[int(0/sf):int(224/sf), int(100/sf):int((1024-100)/sf), : ] #
        if imgc:
            print('subject (input to the top custodian):')
            imgcat(PIL.Image.fromarray(subject).resize((512,512), PIL.Image.BICUBIC)) 
        subject = np.array(PIL.Image.fromarray(subject).resize((slit,slit), PIL.Image.BICUBIC), dtype=np.uint8)
        subject = torch.tensor(subject.transpose([2, 0, 1]), device=device).unsqueeze(0).to(device).to(torch.float32)
        print(Image,'tensor shape for top subject',subject.shape)
        custodian_top = vgg16(subject, resize_images=False, return_lpips=True)

    if frozenbottom:
        subject = np.array(PIL.Image.open('step-in.png').convert('RGB'), dtype=np.uint8)
        # frozen bottom
        subject = subject[int((512+48)/sf):int(1024/sf), int(100/sf):int((1024-100)/sf), : ] # bottom half og 512 not 256
        if imgc:
            print('subject (input to the bottom custodian):')
            imgcat(PIL.Image.fromarray(subject).resize((512,512), PIL.Image.LANCZOS)) 
        subject = np.array(PIL.Image.fromarray(subject).resize((slit,slit), PIL.Image.LANCZOS), dtype=np.uint8)
        subject = torch.tensor(subject.transpose([2, 0, 1]), device=device).unsqueeze(0).to(device).to(torch.float32)
        print(Image,'tensor shape for bottom subject',subject.shape)
        custodian_bottom = vgg16(subject, resize_images=False, return_lpips=True)
        
    # reset values for cropping (actually we don't need to do it since we already sampled the very same network but in case we've cropped 10%)
    x = None
    y = None
    r = None
    b = None

    if noise_opt:
        # this is the inpits for noise maps. it's from OG projector
        noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }


    #if not noise_opt:
    #    pmul = 1

    if not stylemix:
        # option A: w is uniform, i.e. the same [1,1,512] is broadcast to all layers [1,18,512]
        w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) #
        w_out = torch.zeros( [num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
        print('w_opt shape:', w_opt.shape)
        print('w_out shape:', w_out.shape)
    else:
        # option B: stylemix. optimizable shape is [1,18,512]
        w_optfull = torch.tensor(w_og, dtype=torch.float32, device=device, requires_grad=True) 
        w_outfull = torch.zeros([num_steps] + list(w_optfull.shape[1:]), dtype=torch.float32, device=device)
        print('w_optfull shape:', w_optfull.shape)
        print('w_outfull shape:', w_outfull.shape)

    # we get rid of misleading shit
    # we keep noise inputs for time being
    if not stylemix:
        if noise_opt:
            optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
        else:
            optimizer = torch.optim.Adam( [w_opt], betas=(0.9, 0.999), lr=initial_learning_rate)
        print('optimizer: w_opt. shape of w_opt:',w_opt.shape)
    else:
        if noise_opt:
            optimizer = torch.optim.Adam([w_optfull] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
        else:
            optimizer = torch.optim.Adam( [w_optfull], betas=(0.9, 0.999), lr=initial_learning_rate)
        print('optimizer: w_optfull. shape:',w_optfull.shape)

    if noise_opt:
        # initialize per-layer noise maps
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

    print('Cam:', cam)
    # Descend.

    for step in range(num_steps):
        # noise schedule. kept from OG
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2

        # floor
        #if w_noise_scale < noise_floor:
        #    w_noise_scale = noise_floor

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
        else:
            w_noise = torch.randn_like(w_opt) * w_noise_scale

        # add noise to gradable w_opt / w_optfull
        if stylemix: 
            #ws = w_optfull + w_noise # we feed gradable (1,18,512) - need to add noise there
            ws = w_optfull + w_noise # we feed gradable (1,18,512) - need to add noise there
        else:
            # ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1]) # we convert (1,1,512) to (1,18,512)
            ws = (w_opt + w_noise).repeat([w_batch_size, G.mapping.num_ws, 1]) # we convert (1,1,512) to (1,18,512)


        # and do the synthesis.
        # the OG mode uses noise_mode=const since optimizing/regularizing noise buffers are a part of design
        # three modes are allowed 'const', 'random', 'none'. noise is added during synth and is controlled by trainable/learnable noise_strength (it affects all modes
        '''
        class SynthesisLayer(torch.nn.Module):
        def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        '''
        #print('w fed to generator shape', ws.shape)
        synth_images = G.synthesis(ws, noise_mode='random') # noise mode const or random
        apper_range = 8 # when increasing batch size of generated images w/random noise, decrease it in order to keep CLIP batch size reasonable

        #nom = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))        
        into = synth_images
        #into = nom(into) 

        if portrait:
            if not autoportrait:
                into = into[:, :, 0:int(512/sf), int(256/sf):int((256+512)/sf)]
                #print(into.shape)
            else:
                # mk.2 detect face using landmarks
                # convert tensor to RGB numpy array
                img = (into + 1) * (255/2) 
                img = img.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()

                if not x or not y or not r or not b: # one time population is DELIBERATE
                    xyrb = ffhq_facecrop(img,detector,shape_predictor)
                    x = xyrb[0]
                    y = xyrb[1]
                    r = xyrb[2]
                    b = xyrb[3]

                    # hack - source images contain no neck but ffhq crop does, so we take 10% off
                    #b = xyrb[3] - int((xyrb[3] - xyrb[1]) * (10 / 100))
            into = into[:, :, y:b, x:r]

        # display stuff
        if imgc and (step == 0 or step % 20 == 0): 
            #print("output as fed to classifier aka into:")
            disp = synth_images
            #disp = into
            synth_images_save = (disp + 1) * (255/2) 
            synth_images_save = synth_images_save.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            imgcat(PIL.Image.fromarray(synth_images_save))



        # EXPERIMENT
        into = into[:, :, 192:1023, 0:1023]

        # counteradversarial "apper" measures
        # HIGHLY BENEFICIAL
        # copied from lucidrains bigsleep. (sorta) helps to prevent the "adversarial generation" thing
        if cam:
            out = into
            if autoportrait:
                width = b - y # note that we don't know these if not automodel
            else:
                # assume the network output is 1024^2
                width = 1024
            pieces = []
            for ch in range(apper_range): # need to use value of 32..64
                size = int(width * torch.zeros(1,).normal_(mean=.8, std=.3).clip(.5, .95))
                offsetx = torch.randint(0, width - size, ())
                offsety = torch.randint(0, width - size, ())
                apper = out[:, :, offsetx:offsetx + size, offsety:offsety + size]
                apper = F.interpolate(apper, (clipres,clipres),  mode='bilinear', align_corners=True)
                pieces.append(apper)
            kunst = torch.cat(pieces)
        else:
            if not vgg:
                kunst = torch.nn.functional.interpolate(into, (clipres,clipres), mode='bilinear', align_corners=True)
            else:
                kunst = torch.nn.functional.interpolate(into, (slit,slit), mode='bilinear', align_corners=True)

        #print('kunst shape', kunst.shape)


        if not vgg: # CLIP
            glimmers = perceptor.encode_image(kunst)

            # if multimode, apply text to the whole image. non-augmented
            if multimode:
                shiners = perceptor.encode_image(torch.nn.functional.interpolate(synth_images, (clipres,clipres), mode='bilinear', align_corners=True))

            if not multimode:
                if Image:
                    #img = preprocess(PIL.Image.fromarray(img)).unsqueeze(0).to(device)
                    nkoff = torch.cosine_similarity(vibes, glimmers, dim = -1).mean() - 0.1 # -0.1 added on 11/5/2021
                    proximity =  -1 * pmul * nkoff
                else:
                    nkoff = torch.cosine_similarity(whispers, glimmers, dim = -1).mean()
                    proximity =  -1 * pmul * nkoff
            else:
                # if multimode, apply text to the whole image
                nkoff1 = torch.cosine_similarity(whispers, shiners, dim = -1).mean()
                nkoff2 = torch.cosine_similarity(vibes, glimmers, dim = -1).mean()
                proximity =  -1 * pmul * (nkoff1 * 0.5 + nkoff2)


        else: # VGG, assuming multimode is false and text is false (can compare only against images)
            kunst = torch.nn.functional.interpolate(into, (slit,slit), mode='area') #OG PB
            #into = (into + 1) * (255/2)
            #if synth_images.shape[2] > 256:
            #synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')
            kunst = (torch.clamp(kunst, -1, 1) + 1) * (255/2)
            #print('kunst shape', kunst.shape)
            glimmers = vgg16(kunst, resize_images=False, return_lpips=True) #.mean()
            #print('glimmers shape', glimmers.shape)
            #vibes = torch.mean(vibes,0,keepdim=True)
            nkoff =  (vibes - glimmers).square().sum()
            nkoff *= 6
            nkoff = F.relu(nkoff*nkoff - 0.1) # 0.1 min thresh
            '''
            # very experimental - add 2nd vgg
            into2 = synth_images[:, :, 512:1023, 512:1023]  # borrom (right-libertarian) quadrant
            kunst2 = torch.nn.functional.interpolate(into2, (clipres,clipres), mode='area') #OG PB
            kunst2 = (torch.clamp(kunst2, -1, 1) + 1) * (255/2)
            glimmers2 = vgg16(kunst2, resize_images=False, return_lpips=True) #.mean()
            nkoff2 =  (vibes2 - glimmers2).square().sum()
            nkoff2 *= 6
            nkoff2 = F.relu(nkoff2*nkoff2 - 0.1) # 0.1 min thresh
            '''
            proximity = nkoff # + nkoff2  

        # even moar experimental
        if frozentop:
            into = synth_images[:, :, int(0/sf):int(224/sf), int(100/sf):int((1024-100)/sf)] # frozen top lol
            into = torch.nn.functional.interpolate(into, (slit,slit), mode='area') #OG PB
            into = (torch.clamp(into, -1, 1) + 1) * (255/2)
            custodee = vgg16(into, resize_images=False, return_lpips=True) #.mean()
            custody = (custodian_top - custodee).square().sum()
            custody *= 6
            custody = F.relu(custody*custody - 0.1) # 0.1 min thresh
            proximity += custody * 1  # was 1.5 before but with pmul = 30

        if frozenbottom:
            into = synth_images[:, :, int((512+48)/sf):int(1024/sf), int(100/sf):int((1024-100)/sf)] # og [:, :, 512:1023, 100:1024-100]
            into = torch.nn.functional.interpolate(into, (slit,slit), mode='area') #OG PB
            into = (torch.clamp(into, -1, 1) + 1) * (255/2)
            custodee = vgg16(into, resize_images=False, return_lpips=True) #.mean()
            custody = (custodian_bottom - custodee).square().sum()
            custody *= 6
            custody = F.relu(custody*custody - 0.1) # 0.1 min thresh
            proximity += custody * 0.65  # was 1.5 before but with pmul = 30

        # don't do this! things become weird
        #proximity += 30




        reg_loss = 0.0
        if noise_opt:
            # noise reg, from og projector
            for v in noise_bufs.values():
                noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)

        if noise_opt:
            loss = proximity + reg_loss * regularize_noise_weight
        else:
            loss = proximity

        #loss = proximity

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if not multimode:
            if not vgg:
                print(f'step {step+1:>4d}/{num_steps}:  loss {float(loss):<5.2f}, reg_loss {float(reg_loss):<5.5f} ','lr', lr, f'noise scale: {float(w_noise_scale):<5.6f}',f'sim: {float(nkoff):<5.6f}')
            else:
                print(f'step {step+1:>4d}/{num_steps}:  loss {float(loss):<5.4f}, reg_loss {float(reg_loss):<5.5f} ','lr', lr, f'noise scale: {float(w_noise_scale):<5.6f}',f'distance: {float(proximity):<5.6f}')

        else:
            print(f'step {step+1:>4d}/{num_steps}:  loss {float(loss):<5.4f} ','lr', lr, f'noise scale: {float(w_noise_scale):<5.6f}',f'sim whispers: {float(nkoff1):<5.6f}',f'sim vibes: {float(nkoff2):<5.6f}')

        if stylemix:
            w_outfull[step] = w_optfull.detach()
        else: 
            w_out[step] = w_opt.detach()[0]

        if noise_opt:
            # Normalize noise maps
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

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
@click.option('--psi',                    help='Truncation psi for initial image', type=float, required=False, default=0.81)
@click.option('--inf',                    help='Initial noise factor', type=float, required=False, default=0.05) # og was 0.02
@click.option('--nf',                     help='Noise floor', type=float, required=False, default=0.00)
@click.option('--noise-opt',              help='optimize noise as in OG', type=bool, required=False, default=False)
@click.option('--stylemix',              help='Optimize all layers of w instead of just one', type=bool, required=False, default=False)
@click.option('--anime',              help='Do not attempt to auto crop (by face detection) source images, use manual hack instead', type=bool, required=False, default=False)
@click.option('--text',                   help='Text prompt', type=str, default='an image', required=False)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--save-ws',                help='Save intermediate ws', type=bool, default=False, show_default=True)
@click.option('--portrait',               help='Limit CLIP image sample to face', type=bool, default=False, show_default=True)
@click.option('--autoportrait',           help='When used with portrait, look for a face using FFHQ face detector insterad of just the center of the upper half', type=bool, default=False, show_default=True)
@click.option('--multimode',              help='Use two CLIP ratings to guide! Needs both --image and --text', type=bool, default=False, show_default=True)
@click.option('--frozenbottom',            help='Loss function will penalize changes to the lower half of the image. To be used only in portrait mode', type=bool, default=None, show_default=True)
@click.option('--frozentop',            help='Loss function will penalize changes to the nekomimi of the image. To be used only in portrait mode', type=bool, default=None, show_default=True)
@click.option('--cam',                    help='Counteradversarial measures. Feed N randomly cropped samples to CLIP. Helps to benefit from longer training schedules', type=bool, default=True, show_default=True)
@click.option('--kmeans',                 help='Use kmeans Z sampling for autoseed', type=bool, default=False, show_default=True)
@click.option('--image',                  help='(experimental) use image instead of text. may specify several images. for working with portrait mode, it is assumed that images are 1024px', required=False, default=None,  multiple=True)
@click.option('--imgc',              help='Use imgcat to display images in terminal (does not work with screen/tmux)', type=bool, required=False, default=False)
@click.option('--vgg',              help='Use VGG instead of CLIP. For images only, no multimode', type=bool, required=False, default=False)
@click.option('--sf',              help='Slash factor, affects various hardcoded coords. Use 2 for 512px ', type=int, required=False, default=1)


def run_approach(
    network_pkl: str,
    outdir: str,
    save_video: bool,
    save_ws: bool,
    seed: int,
    num_steps: int,
    text: str,
    autoseed: bool,
    anime: bool,
    image: str,
    lr: float,
    frozenbottom: bool,
    frozentop: bool,
    inf: float,
    nf: float,
    sf: int,
    w: str,
    psi: float,
    cam: bool,
    vgg: bool,
    imgc: bool,
    kmeans: bool,
    multimode: bool,
    stylemix: bool,
    noise_opt: bool,
    autoportrait: bool,
    portrait: bool

):
    """Descend on StyleGAN2 w vector value using CLIP, tuning an image with given text prompt. 

    Example:

    \b
    python3 approach.py --network network-snapshot-ffhq.pkl --outdir project --num-steps 100  \\ 
    --text 'an image of a girl with a face resembling Paul Krugman' --psi 0.8 --seed 12345
    
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
        Image = image,
        autoseed = autoseed,
        anime = anime,
        sf = sf,
        portrait = portrait,
        kmeans = kmeans,
        frozenbottom = frozenbottom,
        frozentop = frozentop,
        cam = cam,
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
        imc = PIL.Image.fromarray(synth_image) 
        imgcat(imc)

if __name__ == "__main__":
    run_approach() 
