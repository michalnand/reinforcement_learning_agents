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

    '''
    if augmentations is not None:
        xa_aug, _ = augmentations(xa) 
        xb_aug, _ = augmentations(xb)
    else:
        xa_aug = xa 
        xb_aug = xb
    '''
    if augmentations is not None:
        xb_aug, _ = augmentations(xb)
    else:
        xb_aug    = xb

    za, zb = model_forward_func(xa, xb_aug)
    
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
    z_mag      = round(((za**2).mean()).detach().cpu().numpy().item(), 6)
    z_mag_std  = round(((za**2).std()).detach().cpu().numpy().item(), 6)
    sim_loss_  = round(sim_loss.detach().cpu().numpy().item(), 6)
    std_loss_  = round(std_loss.detach().cpu().numpy().item(), 6)
    cov_loss_  = round(cov_loss.detach().cpu().numpy().item(), 6)
    
    info = [z_mag, z_mag_std, sim_loss_, std_loss_, cov_loss_]

    return loss, info








'''
1, input images : xa, xb

2, first the features are computed :
        za = f(xa)
        zb = f(xb)

3,  for za, zb are variance (std_loss) and covariance (cov_loss) terms, regularizing z-space (same as vicreg)

4,  hidden information is computed :
        h = h(za)
    which contains transformerd information about zb
    in final, goal is to minimize h, to provide minimum hidden information

5, predictor is computed as : 
        p = predictor(za, h)

        and invariance (mse loss) is called :
        MSE(za - p)

        essence of JEPA is : p-model can easily predict za from zb, 
        because p contains not only augmented context from zb,
        but also "cheating" information from za (here contained hidden information)
        
        however, h-features are forced by loss to minimize information, 
        which leads to good za, zb features, and h contains only the necessary information 
        from za which can't be extracted from zb (and vice versa)

        - this necessary information can be camemera position, or noise ...
'''
def loss_jepa(model_forward_func, augmentations, xa, xb, hidden_coeff = 0.01):
    xa_aug, _ = augmentations(xa)
    xb_aug, _ = augmentations(xb)

    za, zb, h, p = model_forward_func(xa_aug, xb_aug)  

    # invariance loss - predict za from p
    sim_loss = _loss_mse(za, p)

    # variance loss
    std_loss = _loss_std(za)
    std_loss+= _loss_std(zb)

    # covariance loss 
    cov_loss = _loss_cov(za)
    cov_loss+= _loss_cov(zb)

    #hidden information loss, enforce sparsity, and minimize batch-wise variance
    h_mag = torch.abs(h).mean()
    h_std = (h.std(dim=0)).mean()
    hidden_loss = h_mag + h_std

    # total loss, vicreg + info-min
    loss = 1.0*sim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss + hidden_coeff*hidden_loss


    #info for log
    z_mag     = round(((za**2).mean()).detach().cpu().numpy().item(), 6)
    z_mag_std = round(((za**2).std()).detach().cpu().numpy().item(), 6)
    h_mag     = round(((h**2).mean()).detach().cpu().numpy().item(), 6)
    h_mag_std = round(((h**2).std()).detach().cpu().numpy().item(), 6)

    sim_loss_ = round(sim_loss.detach().cpu().numpy().item(), 6)
    std_loss_ = round(std_loss.detach().cpu().numpy().item(), 6)
    cov_loss_ = round(cov_loss.detach().cpu().numpy().item(), 6)
    hidden_loss_ = round(hidden_loss.detach().cpu().numpy().item(), 6)
   
    info = [z_mag, z_mag_std, h_mag, h_mag_std, sim_loss_, std_loss_, cov_loss_, hidden_loss_]

    return loss, info



def loss_jepa_sim(model_forward_func, augmentations, xa, xb, hidden_coeff = 0.01):
    xa_aug, _ = augmentations(xa)
    xb_aug, _ = augmentations(xb)

    za, zb, ha, pa, hb, pb = model_forward_func(xa_aug, xb_aug)  

    # invariance loss - predict za from pa
    sim_loss = _loss_mse(za, pa)
    # invariance loss - predict zb from pb
    sim_loss+= _loss_mse(zb, pb)

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
    loss = 1.0*sim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss + hidden_coeff*hidden_loss


    #info for log
    z_mag     = round(((za**2).mean()).detach().cpu().numpy().item(), 6)
    z_mag_std = round(((za**2).std()).detach().cpu().numpy().item(), 6)
    h_mag     = round(((ha**2).mean()).detach().cpu().numpy().item(), 6)
    h_mag_std = round(((ha**2).std()).detach().cpu().numpy().item(), 6)

    sim_loss_ = round(sim_loss.detach().cpu().numpy().item(), 6)
    std_loss_ = round(std_loss.detach().cpu().numpy().item(), 6)
    cov_loss_ = round(cov_loss.detach().cpu().numpy().item(), 6)
    hidden_loss_ = round(hidden_loss.detach().cpu().numpy().item(), 6)
   
    info = [z_mag, z_mag_std, h_mag, h_mag_std, sim_loss_, std_loss_, cov_loss_, hidden_loss_]

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

    
    #info for log
    z_mag     = round(((za**2).mean()).detach().cpu().numpy().item(), 6)
    z_mag_std = round(((za**2).std()).detach().cpu().numpy().item(), 6)
    h_mag     = round(((ha**2).mean()).detach().cpu().numpy().item(), 6)
    h_mag_std = round(((ha**2).std()).detach().cpu().numpy().item(), 6)

    sim_loss_ = round(sim_loss.detach().cpu().numpy().item(), 6)
    std_loss_ = round(std_loss.detach().cpu().numpy().item(), 6)
    cov_loss_ = round(cov_loss.detach().cpu().numpy().item(), 6)
    hidden_loss_ = round(hidden_loss.detach().cpu().numpy().item(), 6)
   

    info = [z_mag, z_mag_std, h_mag, h_mag_std, sim_loss_, std_loss_, cov_loss_, hidden_loss_]

    return loss, info

