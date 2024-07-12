import torch
import numpy
  
#apply random agumentation
def aug_random_apply(x, p, aug_func):
    mask        = (torch.rand(x.shape[0]) < p).float().to(x.device)
    mask_tmp    = mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    y           = (1.0 - mask_tmp)*x + mask_tmp*aug_func(x)
 
    return y, mask  

#uniform aditional noise
def aug_noise(x, k = 0.2): 
    pointwise_noise = k*(2.0*torch.rand(x.shape, device=x.device) - 1.0)
    return x + pointwise_noise  

# random black mask
def aug_mask(x, p = 0.75, gw = 16, gh = 16):
    up_h = x.shape[2]//gh
    up_w = x.shape[3]//gw 

    mask = torch.rand((x.shape[0], x.shape[1], gh, gw), device = x.device)
    
    mask = torch.nn.functional.interpolate(mask, scale_factor = (up_h, up_w), mode="bicubic")
    mask = (mask > (1.0 - p)).float().detach()

    return mask*x   

# random conv kernel augmentation
def aug_conv(x, kernel_size = 5):
    size    = x.shape[0]*x.shape[1]
    x_tmp   = x.reshape((1, x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
    weights = torch.randn((size, 1, kernel_size, kernel_size), device=x.device)

    y  = torch.nn.functional.conv2d(x_tmp, weights, stride=1, padding=kernel_size//2, groups=size)
    y  = y.reshape((x.shape[0], x.shape[1], x.shape[2], x.shape[3]))

    return y




#random select xa or xb, xa with prob (1 - p), xb with prob p
def aug_random_select(xa, xb, p):
    mask     = (torch.rand(xa.shape[0]) < p).float().to(xa.device)
    mask_tmp = mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    y        = (1 - mask_tmp)*xa + mask_tmp*xb
    return y, mask


#random invert colors
def aug_inverse(x): 
    r = torch.randint(0, 2, (x.shape[0], x.shape[1]), device=x.device).unsqueeze(2).unsqueeze(3)
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



#random tiled dropout
def aug_mask_tiles(x, p = 0.1):

    if x.shape[2] == 96:
        tile_sizes  = [1, 2, 4, 8, 12, 16]
    else:
        tile_sizes  = [1, 2, 4, 8, 16]

    tile_size   = tile_sizes[torch.randint(len(tile_sizes))]

    size_h  = x.shape[2]//tile_size
    size_w  = x.shape[3]//tile_size

    mask    = (torch.rand((x.shape[0], 1, size_h, size_w)) < (1.0 - p))

    mask    = torch.kron(mask, torch.ones(tile_size, tile_size))

    return x*mask.float().to(x.device)  


#random spot noise 
def aug_random_tiles(x, max_loops = 4, p_base=0.1): 
    loops   = numpy.random.randint(0, max_loops+1)

    p       = p_base/((2*loops + 1)**2)

    mask    = (torch.rand((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device) < p).float()

    pool    = torch.nn.MaxPool2d(3, stride=1, padding=1)

    for i in range(loops):
        mask = pool(mask)

    mask = (1.0 - mask)
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

#random choice from xa or xb, 50:50 prob default
def choice_augmentation(xa, xb, pa_prob = 0.5):
    s = (torch.rand(xa.shape[0], 1, 1, 1).to(xa.device) < pa_prob).float()
    return s*xa + (1.0 - s)*xb


'''
random sized tiles, with random color
'''
def aug_noisy_tiles(x, sizes = [1, 2, 4, 8, 16], p = 0.1): 

    size = numpy.random.choice(sizes)

    mask = torch.rand((x.shape[0], x.shape[1], x.shape[2]//size, x.shape[3]//size), device=x.device)
    mask = (mask < p).float()

    noise = torch.rand((x.shape[0], x.shape[1], x.shape[2]//size, x.shape[3]//size), device=x.device)

    mask  = torch.nn.functional.interpolate(mask,  scale_factor=size)
    noise = torch.nn.functional.interpolate(noise, scale_factor=size)

    result = (1.0 - mask)*x + mask*noise 

    return result

'''
random brightness and offset, channel idependent 
'''
def aug_intensity(x, k_min = 0.5, k_max = 1.5, q_min = -1.0, q_max = 1.0):

    k = torch.rand((x.shape[0], x.shape[1], 1, 1), device=x.device)
    k = (1.0 - k)*k_min + k*k_max

    q = torch.rand((x.shape[0], x.shape[1], 1, 1), device=x.device)
    q = (1.0 - q)*q_min + q*q_max 

    result = k*x + q 

    return result

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
   


def aug_channel_mask(x):
    indices = torch.randint(0, x.shape[1], (x.shape[0], ))

    result = x.clone()
    result[range(indices.shape[0]), indices] = 0
    
    return result



def aug_mask_advanced(x):
    gh = 16
    gw = 16 

    up_h = x.shape[2]//gh
    up_w = x.shape[3]//gw 

    mask = torch.rand((x.shape[0], x.shape[1], gh, gw), device = x.device)

    p    = torch.rand((x.shape[0], 1, 1, 1), device = x.device)
    
    mask = torch.nn.functional.interpolate(mask, scale_factor = (up_h, up_w), mode="bicubic")
    mask = (mask > p).float().detach()

    return mask*x       


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = torch.ones((20, 3, 96, 96))    
    
    #y = aug_noise(x) 
    y = aug_mask(x) 
    y = aug_conv(y) 
    #y = aug_mask_advanced(x) 

    print(">>> ", x.shape, y.shape, y.mean())

    plt.matshow(y.cpu().detach().numpy()[0][0])
    plt.matshow(y.cpu().detach().numpy()[1][0])
    plt.matshow(y.cpu().detach().numpy()[2][0])
    plt.show()