import torch
import numpy
  
#apply random agumentation
def aug_random_apply(x, p, aug_func):
    mask    = (torch.rand(x.shape[0]) < p)
    mask    = mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    mask    = mask.float().to(x.device)
    y       = (1.0 - mask)*x + mask*aug_func(x)

    return y   

#random invert colors
def aug_inverse(x): 
    r = torch.randint(0, 2, (x.shape[0], x.shape[1])).unsqueeze(2).unsqueeze(3).to(x.device)
    return r*(1.0 - x) + (1.0 - r)*x

#random channels permutation
def aug_permutation(x):
    indices = torch.stack([torch.randperm(x.shape[1]) for _ in range(x.shape[0])])
    return x[torch.arange(x.shape[0]).unsqueeze(1), indices]
     
 
#pixelate, downsample, and upsample back
def aug_pixelate(x, p = 0.5): 

    #downsample 2x or 4x
    scale   = int(2**numpy.random.randint(1, 3))
    ds      = torch.nn.AvgPool2d(scale, scale).to(x.device)
    us      = torch.nn.Upsample(scale_factor=scale).to(x.device)

    #tiles mask
    mask    = 1.0 - torch.rand((x.shape[0], 1, x.shape[2]//scale, x.shape[3]//scale))
    mask    = (mask < p).float().to(x.device)
    mask    = us(mask)

    scaled  = us(ds(x))  
    return mask*scaled + (1.0 - mask)*x

#random zeroing mask with p
def aug_pixel_dropout(x, p = 0.1):
    mask = 1.0 - (torch.rand_like(x) < p).float()
    return x*mask  


#apply random convolution filter
def aug_conv(x):
    #random kernel size : 1, 3, 5, 7
    kernel_size =  2*numpy.random.randint(0, 4) + 1

    ch = x.shape[1] 

    #random weights
    w   = torch.randn((ch, 1, kernel_size, kernel_size), dtype=torch.float32, device = x.device)
    
    #apply filter
    y   = torch.nn.functional.conv2d(x, w, padding=kernel_size//2, groups=ch)

    return y



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


#random spot noise 
def aug_random_tiles(x, max_loops = 4, p_base=0.1): 
    loops   = numpy.random.randint(0, max_loops+1)

    p       = p_base/((2*loops + 1)**2)

    mask    = (torch.rand((x.shape[0], 1, x.shape[2], x.shape[3])) < p).float()

    pool    = torch.nn.MaxPool2d(3, stride=1, padding=1)

    for i in range(loops):
        mask = pool(mask)

    mask = (1.0 - mask.to(x.device))
    return x*mask




def aug_edges(x): 
    y  = x #torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1)
    ch = x.shape[1]

    w  = torch.zeros((3, 3))
    w[1][1] =  1.0

    w[0][1] = -0.25
    w[2][1] = -0.25
    w[1][0] = -0.25
    w[1][2] = -0.25

    w = w.unsqueeze(0).unsqueeze(1)
    w = torch.repeat_interleave(w, ch, 0)

    y = torch.nn.functional.conv2d(y, w, bias=None, stride=1, padding=1, groups=ch)
    y = torch.abs(y)

    return y

#uniform aditional noise
def aug_noise(x, k = 0.2): 
    pointwise_noise   = k*(2.0*torch.rand(x.shape, device=x.device) - 1.0)
    return x + pointwise_noise

#random choice from xa or xb, 50:50 prob default
def choice_augmentation(xa, xb, pa_prob = 0.5):
    s = (torch.rand(xa.shape[0], 1, 1, 1).to(xa.device) < pa_prob).float()
    return s*xa + (1.0 - s)*xb


'''
#resize, downsample, and upsample back
def aug_resize(x, scale = 2):
    ds      = torch.nn.AvgPool2d(scale, scale).to(x.device)
    us      = torch.nn.Upsample(scale_factor=scale).to(x.device)

    scaled  = us(ds(x))  
    return scaled

def aug_resize2(x):
    return aug_resize(x, 2)

def aug_resize4(x)  :
    return aug_resize(x, 4)

#random pixel-wise dropout
def aug_mask(x, p = 0.1):
    mask = 1.0*(torch.rand_like(x) < (1.0 - p))
    return x*mask  
'''
 



'''
def aug( x):
    
    #this works perfect

    x = _aug_random_apply(x, 0.5, _aug_resize2)
    x = _aug_random_apply(x, 0.25, _aug_resize4)
    x = _aug_random_apply(x, 0.125, _aug_mask)
    x = _aug_noise(x, k = 0.2)

    return x
'''