import torch
import numpy


#apply random agumentation
def aug_random_apply(x, p, aug_func):
    mask    = (torch.rand(x.shape[0]) < p)
    mask    = mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    mask    = mask.float().to(x.device)
    y       = (1.0 - mask)*x + mask*aug_func(x)

    return y 

#uniform aditional noise
def aug_noise(x, k = 0.2): 
    pointwise_noise   = k*(2.0*torch.rand(x.shape, device=x.device) - 1.0)
    return x + pointwise_noise

#random tiled dropout
def aug_mask_tiles(x, p = 0.1):

    if x.shape[2] == 96:
        tile_sizes  = [1, 2, 4, 8, 12, 16]
    else:
        tile_sizes  = [1, 2, 4, 8, 16]

    tile_size   = tile_sizes[numpy.random.randint(len(tile_sizes))]

    size_h  = x.shape[2]//tile_size
    size_w  = x.shape[3]//tile_size

    mask    = (torch.rand((x.shape[0], 1, size_h, size_w)) < (1.0 - p))

    mask    = torch.kron(mask, torch.ones(tile_size, tile_size))

    return x*mask.float().to(x.device)  


#resize, downsample, and upsample back
def aug_resize(x, scale = 2):
    ds      = torch.nn.AvgPool2d(scale, scale).to(x.device)
    us      = torch.nn.Upsample(scale_factor=scale).to(x.device)

    scaled  = us(ds(x))  
    return scaled

def aug_resize2(x):
    return aug_resize(x, 2)

def aug_resize4(x):
    return aug_resize(x, 4)

#random pixel-wise dropout
def aug_mask(x, p = 0.1):
    mask = 1.0*(torch.rand_like(x) < (1.0 - p))
    return x*mask  


#apply random convolution filter
def aug_conv(x, alpha = 0.1, kernel_size = 3, groups = None):
    ch  = x.shape[1]
    w   = torch.zeros((ch, ch, kernel_size, kernel_size), device = x.device)

    r0  = range(ch) 
 
    if groups is None:
        if ch%3 == 0:
            groups = ch//3
        else:
            groups = ch
    
    ch_tmp  = ch//groups
    r1      = ch_tmp*numpy.repeat(numpy.arange(groups), ch_tmp) + numpy.tile(numpy.random.permutation(ch_tmp), groups)
    
    w[r0, r1, kernel_size//2, kernel_size//2] = 1.0

    w   = (1.0 - alpha)*w + alpha*torch.randn_like(w)

    y   = torch.nn.functional.conv2d(x, w, padding=kernel_size//2)

    return y



'''
def aug( x):
    
    #this works perfect

    x = _aug_random_apply(x, 0.5, _aug_resize2)
    x = _aug_random_apply(x, 0.25, _aug_resize4)
    x = _aug_random_apply(x, 0.125, _aug_mask)
    x = _aug_noise(x, k = 0.2)

    return x
'''