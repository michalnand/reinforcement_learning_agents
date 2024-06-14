import torch
import numpy

def _off_diagonal(x):
    mask = 1.0 - torch.eye(x.shape[0], device=x.device)
    return x*mask 

# invariance loss
def _loss_mse(xa, xb):
    return ((xa - xb)**2).mean()
    
# variance loss
def _loss_std(x):
    eps = 0.0001 
    std_x = torch.sqrt(x.var(dim=0) + eps)

    loss = torch.mean(torch.relu(1.0 - std_x)) 
    return loss

# covariance loss 
def _loss_cov(x):
    x_norm = x - x.mean(dim=0)
    cov_x = (x_norm.T @ x_norm) / (x.shape[0] - 1.0)
    
    loss = _off_diagonal(cov_x).pow_(2).sum()/x.shape[1] 
    return loss

# force x term into values -1 or +1 (corners of hypercube)
def _loss_hypercube_corner(x):
    loss = (1.0 - (x**2))**2
    loss = loss.mean()

    return loss

# cross correlation loss 
def _loss_cross(xa, xb):

    xa_norm = xa - xa.mean(dim=0)
    xb_norm = xb - xb.mean(dim=0)
    cov = (xa_norm @ xb_norm.T) / (xa.shape[1] - 1.0)
    
    loss = (cov**2).mean()

    return loss




def loss_vicreg_direct(za, zb):
    # invariance loss
    sim_loss = _loss_mse(za, zb)

    # variance loss
    std_loss = _loss_std(za)
    std_loss+= _loss_std(zb)
   
    # covariance loss 
    cov_loss = _loss_cov(za)
    cov_loss+= _loss_cov(zb)
   
    # total vicreg loss
    loss = 1.0*sim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss

    #info for log
    z_mag     = round(((za**2).mean()).detach().cpu().numpy().item(), 6)
    z_mag_std = round(((za**2).std()).detach().cpu().numpy().item(), 6)
    
    info = [z_mag, z_mag_std]

    return loss, info




def loss_vicreg(model_forward_func, augmentations, xa, xb):
    if augmentations is not None:
        xa_aug, _ = augmentations(xa) 
        xb_aug, _ = augmentations(xb)
    else:
        xa_aug = xa 
        xb_aug = xb

    za = model_forward_func(xa_aug)
    zb = model_forward_func(xb_aug)

    # invariance loss
    sim_loss = _loss_mse(za, zb)

    # variance loss
    std_loss = _loss_std(za)
    std_loss+= _loss_std(zb)
   
    # covariance loss 
    cov_loss = _loss_cov(za)
    cov_loss+= _loss_cov(zb)
   
    # total vicreg loss
    loss = 1.0*sim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss

    #info for log
    z_mag     = round(((za**2).mean()).detach().cpu().numpy().item(), 6)
    z_mag_std = round(((za**2).std()).detach().cpu().numpy().item(), 6)
    sim_loss_  = round(sim_loss.detach().cpu().numpy().item(), 6)
    std_loss_  = round(std_loss.detach().cpu().numpy().item(), 6)
    cov_loss_  = round(cov_loss.detach().cpu().numpy().item(), 6)
    
    info = [z_mag, z_mag_std, sim_loss_, std_loss_, cov_loss_]

    return loss, info




def loss_vicreg_hypercube(model_forward_func, augmentations, xa, xb):
    if augmentations is not None:
        xa_aug, _ = augmentations(xa) 
        xb_aug, _ = augmentations(xb)
    else:
        xa_aug = xa 
        xb_aug = xb

    za = model_forward_func(xa_aug)
    zb = model_forward_func(xb_aug)

    # invariance loss
    sim_loss = _loss_mse(za, zb)

    # variance loss
    std_loss = _loss_std(za)
    std_loss+= _loss_std(zb)
   
    # covariance loss 
    cov_loss = _loss_cov(za)
    cov_loss+= _loss_cov(zb)

    hc_loss = _loss_hypercube_corner(za)
    hc_loss+= _loss_hypercube_corner(zb)
   
    # total vicreg loss
    loss = 1.0*sim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss

    #info for log
    z_mag     = round(((za**2).mean()).detach().cpu().numpy().item(), 6)
    z_mag_std = round(((za**2).std()).detach().cpu().numpy().item(), 6)
    sim_loss_  = round(sim_loss.detach().cpu().numpy().item(), 6)
    std_loss_  = round(std_loss.detach().cpu().numpy().item(), 6)
    cov_loss_  = round(cov_loss.detach().cpu().numpy().item(), 6)
    hc_loss_   = round(hc_loss.detach().cpu().numpy().item(), 6)
    
    info = [z_mag, z_mag_std, sim_loss_, std_loss_, cov_loss_, hc_loss_]

    return loss, info


def loss_vicreg_seq(model_forward_func, xa, xb, ha, hb):
    xa_aug = xa
    xb_aug = xb

    za = model_forward_func(xa_aug, ha)
    zb = model_forward_func(xb_aug, hb)

    # invariance loss
    sim_loss = _loss_mse(za, zb)

    # variance loss
    std_loss = _loss_std(za)
    std_loss+= _loss_std(zb)
   
    # covariance loss 
    cov_loss = _loss_cov(za)
    cov_loss+= _loss_cov(zb)
   
    # total vicreg loss
    loss = 1.0*sim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss

    #info for log
    z_mag     = round(((za**2).mean()).detach().cpu().numpy().item(), 6)
    z_mag_std = round(((za**2).std()).detach().cpu().numpy().item(), 6)

    sim_loss_np = round((sim_loss.mean()).detach().cpu().numpy().item(), 6)
    std_loss_np = round((std_loss.mean()).detach().cpu().numpy().item(), 6)
    cov_loss_np = round((cov_loss.mean()).detach().cpu().numpy().item(), 6)
    
    info = [z_mag, z_mag_std, sim_loss_np, std_loss_np, cov_loss_np]

    return loss, info


def loss_vicreg_contrastive(model_forward_func, augmentations, xa, xb):
    xa_aug, _ = augmentations(xa) 
    xb_aug, _ = augmentations(xb)

    za = model_forward_func(xa_aug)
    zb = model_forward_func(xb_aug)

    # distance loss 
    target_distances = 1.0 - torch.eye(za.shape[0]).to(device=za.device)
    distances        = torch.cdist(za, zb)/za.shape[1]   
    distance_loss    = ((target_distances - distances)**2).mean()

    # variance loss
    std_loss = _loss_std(za)
    std_loss+= _loss_std(zb)
   
    # covariance loss 
    cov_loss = _loss_cov(za)
    cov_loss+= _loss_cov(zb)
   
    # total vicreg loss
    loss = 1.0*distance_loss + 1.0*std_loss + (1.0/25.0)*cov_loss

    #info for log
    z_mag     = round(((za**2).mean()).detach().cpu().numpy().item(), 6)
    z_mag_std = round(((za**2).std()).detach().cpu().numpy().item(), 6)
    
    info = [z_mag, z_mag_std]
    
    info.append(round(distance_loss.detach().cpu().numpy().item(), 6))
    info.append(round(std_loss.detach().cpu().numpy().item(), 6))
    info.append(round(cov_loss.detach().cpu().numpy().item(), 6))

    return loss, info




'''
def loss_vicreg_mask(model_forward_func, augmentations, xa, xb):
    xa_aug, mask_a = augmentations(xa) 
    xb_aug, mask_b = augmentations(xb)

    #obtain model output features
    za = model_forward_func(xa_aug)
    zb = model_forward_func(xb_aug)


    #merge masks into one
    mask = torch.max(mask_a, mask_b)

    
    #add mask flag where inputs are time-shifted (temporal augmentation)
    diff = ((xa - xb)**2).mean(dim=(1, 2, 3))
    mask[:, 0] = torch.max(mask[:, 0], (diff > 10**-6))


    repeats = za.shape[1]//mask.shape[1]
    mask_rep= torch.repeat_interleave(mask, repeats, dim=1)


    za_masked = za*mask_rep
    zb_masked = zb*mask_rep

    # invariance loss
    sim_loss = _loss_mse(za_masked, zb_masked)

    # variance loss
    std_loss = _loss_std(za)
    std_loss+= _loss_std(zb)
   
    # covariance loss 
    cov_loss = _loss_cov(za)
    cov_loss+= _loss_cov(zb)
   
    # total vicreg loss
    loss = 1.0*sim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss

    #info for log
    z_mag     = round(((za**2).mean()).detach().cpu().numpy().item(), 6)
    z_mag_std = round(((za**2).std()).detach().cpu().numpy().item(), 6)

    info = [z_mag, z_mag_std]

    #partial subsbaces magnitudes logs
    for i in range(mask.shape[1]):
        mask_tmp                             = torch.zeros(za.shape, device=za.device)
        mask_tmp[:, i*repeats:(i+1)*repeats] = 1.0

        z_tmp = za*mask_tmp

        mag = round(((z_tmp**2).mean()).detach().cpu().numpy().item(), 6)
        std = round(((z_tmp**2).std()).detach().cpu().numpy().item(), 6)
    
        info.append(mag)
        info.append(std)
    

    return loss, info
'''



def loss_vicreg_augs(model_forward_func, model_forward_func_aug, augmentations, xa, xb):
    xa_aug, aug_mask_a = augmentations(xa) 
    xb_aug, aug_mask_b = augmentations(xb)

    #obtain model output features
    za = model_forward_func(xa_aug)
    zb = model_forward_func(xb_aug)

    # invariance loss
    sim_loss = _loss_mse(za, zb)

    # variance loss
    std_loss = _loss_std(za)
    std_loss+= _loss_std(zb)
   
    # covariance loss 
    cov_loss = _loss_cov(za)
    cov_loss+= _loss_cov(zb)



    #augmentation prediction loss
    aug_pred = model_forward_func_aug(za, zb)
    aug_pred = torch.sigmoid(aug_pred)

    #merge aug_mask into one
    aug_target = torch.max(aug_mask_a, aug_mask_b)

    #add mask flag where inputs are time-shifted (temporal augmentation)
    diff = ((xa - xb)**2).mean(dim=(1, 2, 3))
    aug_target[:, 0] = (diff > 10**-6).float()


    loss_func = torch.nn.BCELoss()
    loss_aug  = loss_func(aug_pred, aug_target.float())

   
    

    # total vicreg loss
    loss = 1.0*sim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss + loss_aug

    #info for log
    z_mag     = round(((za**2).mean()).detach().cpu().numpy().item(), 6)
    z_mag_std = round(((za**2).std()).detach().cpu().numpy().item(), 6)

    info = [z_mag, z_mag_std]


    #class accuracy for log
    class_acc = ((aug_target > 0.5) == (aug_pred > 0.5)).float()
    class_acc = class_acc.mean(dim=0)
    class_acc = class_acc.detach().cpu().numpy()

    for i in range(class_acc.shape[0]):
        info.append(round(class_acc[i].item(), 6))


    return loss, info


def loss_vicreg_proj(model_forward_func, augmentations, xa, xb):
    xa_aug, _ = augmentations(xa) 
    xb_aug, _ = augmentations(xb)

    za, proj_a = model_forward_func(xa_aug)
    zb, proj_b = model_forward_func(xb_aug)

    # invariance loss
    sim_loss = _loss_mse(za, zb)

    # variance loss
    std_loss = _loss_std(proj_a)
    std_loss+= _loss_std(proj_b)
   
    # covariance loss 
    cov_loss = _loss_cov(proj_a)
    cov_loss+= _loss_cov(proj_b)
    
    # total vicreg loss
    loss = 1.0*sim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss
 
    #info for log
    z_mag     = round(((za**2).mean()).detach().cpu().numpy().item(), 6)
    z_mag_std = round(((za**2).std()).detach().cpu().numpy().item(), 6)
    
    info = [z_mag, z_mag_std]

    return loss, info





def loss_vicreg_complement(model_forward_func, augmentations, x, x_):
    xa_aug, _ = augmentations(x) 
    xb_aug    = x - xa_aug
    
    z  = model_forward_func(x)
    za = model_forward_func(xa_aug)
    zb = model_forward_func(xb_aug)

    # invariance loss
    sim_loss = _loss_mse(z, za + zb)
 
    # variance loss
    std_loss = _loss_std(z)
    std_loss+= _loss_std(za)
    std_loss+= _loss_std(zb)
   
    # covariance loss 
    cov_loss = _loss_cov(z)
    cov_loss+= _loss_cov(za)
    cov_loss+= _loss_cov(zb)
   
    # total vicreg loss
    loss = 1.0*sim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss

    #info for log
    z_mag     = round(((z**2).mean()).detach().cpu().numpy().item(), 6)
    z_mag_std = round(((z**2).std()).detach().cpu().numpy().item(), 6)

    sim_loss_ = round(sim_loss.detach().cpu().numpy().item(), 6)
    std_loss_ = round(std_loss.detach().cpu().numpy().item(), 6)
    cov_loss_ = round(cov_loss.detach().cpu().numpy().item(), 6)
    
    info = [z_mag, z_mag_std, sim_loss_, std_loss_, cov_loss_]

    return loss, info





'''
1, input images : xa, xb

2, first the features are computed :
        za = f(xa)
        zb = f(xb)

3,  for za, zb are variance (std_loss) and covariance (cov_loss) terms, regularizing z-space

4,  hidden information is computed :
        ha = h(za)
        hb = h(zb)
    which contains transformerd information about za, zb
    in final, goal is to minimize this ha, hb, to provide minimum hidden information

5, predictor is computed as : 
        pa = p(za, hb)
        pb = p(zb, ha)

        and invariance (mse loss) is called :
        MSE(za - pb), MSE(zb - pa)

        essence of JEPA is : p-model can easily predict za from zb, 
        because pa contains not only augmented context from za,
        but also "cheating" information about zb (here called hidden information)
        
        however, h-features are forced by loss to minimize information, 
        which leads to good za, zb features, and h contains only the necessary information 
        from za which can't be extracted from zb (and vice versa)

        - this necessary information can be camemera position, or noise ...
'''
def loss_vicreg_jepa(model_forward_func, augmentations, xa, xb, hidden_coeff = 0.01):
    xa_aug, _ = augmentations(xa)
    xb_aug, _ = augmentations(xb)

    za, zb, pa, pb, ha, hb = model_forward_func(xa_aug, xb_aug)  

    # invariance loss
    sim_loss = _loss_mse(za, pb)
    sim_loss+= _loss_mse(zb, pa) 

    # variance loss
    std_loss = _loss_std(za)
    std_loss+= _loss_std(zb)

    # covariance loss 
    cov_loss = _loss_cov(za)
    cov_loss+= _loss_cov(zb)

    #hidden information loss, enforce sparsity, and minimize batch-wise variance
    h_mag = torch.abs(ha).mean() + torch.abs(hb).mean() 
    h_std = (ha.std(dim=0)).mean() + (hb.std(dim=0)).mean()
    hidden_loss = h_mag + h_std

    # total loss, vicreg + info-min
    loss = 0.5*sim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss + hidden_coeff*hidden_loss

    cross_loss = _loss_cross(za, zb)
    
    #info for log
    z_mag     = round(((za**2).mean()).detach().cpu().numpy().item(), 6)
    z_mag_std = round(((za**2).std()).detach().cpu().numpy().item(), 6)
    h_mag     = round(((ha**2).mean()).detach().cpu().numpy().item(), 6)
    h_mag_std = round(((ha**2).std()).detach().cpu().numpy().item(), 6)

    cross = round(cross_loss.detach().cpu().numpy().item(), 6)

    info = [z_mag, z_mag_std, h_mag, h_mag_std, cross]

    return loss, info


def loss_vicreg_jepa_single(model_forward_func, augmentations, xa, xb, hidden_coeff = 0.01):
    xa_aug, _ = augmentations(xa)
    xb_aug, _ = augmentations(xb)

    za, zb, pa, hb = model_forward_func(xa_aug, xb_aug)  

    # invariance loss
    # predict zb from pa (pa is result from za and hb)
    sim_loss = _loss_mse(zb, pa)

    # variance loss
    std_loss = _loss_std(za)
    std_loss+= _loss_std(zb)

    # covariance loss 
    cov_loss = _loss_cov(za)
    cov_loss+= _loss_cov(zb)

    #hidden information loss, enforce sparsity, and minimize batch-wise variance
    h_mag = torch.abs(hb).mean()
    h_std = (hb.std(dim=0)).mean()
    hidden_loss = h_mag + h_std

    # total loss, vicreg + info-min
    loss = 1.0*sim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss + hidden_coeff*hidden_loss

    cross_loss = _loss_cross(za, zb)
    
    #info for log
    z_mag     = round(((za**2).mean()).detach().cpu().numpy().item(), 6)
    z_mag_std = round(((za**2).std()).detach().cpu().numpy().item(), 6)
    h_mag     = round(((hb**2).mean()).detach().cpu().numpy().item(), 6)
    h_mag_std = round(((hb**2).std()).detach().cpu().numpy().item(), 6)

    cross = round(cross_loss.detach().cpu().numpy().item(), 6)

    info = [z_mag, z_mag_std, h_mag, h_mag_std, cross]

    return loss, info


