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

# variance loss with limit of max alloved variance
def _loss_std_limited(x, std_max_limit):
    eps      = 0.0001 
    std_x    = torch.sqrt(x.var(dim=0) + eps)

    tmp = (std_x < std_max_limit).float().detach()

    loss = -tmp*std_x + (1.0 - tmp)*(std_x - 2.0*std_max_limit)

    return loss.mean()


# covariance loss 
def _loss_cov(x):
    x_norm = x - x.mean(dim=0)
    cov_x = (x_norm.T @ x_norm) / (x.shape[0] - 1.0)
    
    loss = _off_diagonal(cov_x).pow_(2).sum()/x.shape[1] 
    return loss


# variance loss 
def _loss_triangle(xa, xb, xc):
    ab = (xa - xb).abs().mean(dim=-1)
    bc = (xb - xc).abs().mean(dim=-1)
    ca = (xc - xa).abs().mean(dim=-1)

    # positive d means not valid triangle inquality
    d = ca - (ab + bc)

    loss = torch.mean(torch.relu(d))
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




def loss_vicreg_max_var(model_forward_func, augmentations, xa, xb):

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

    xa_std_max = xa.std(dim=0).mean()
    xb_std_max = xb.std(dim=0).mean() 
    za, zb = model_forward_func(xa, xb_aug)
    
    # invariance loss
    sim_loss = _loss_mse(za, zb)

    # variance loss
    std_loss = _loss_std_limited(za, xa_std_max)
    std_loss+= _loss_std_limited(zb, xb_std_max)
   
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
    xa_std_max_= round(xa_std_max.detach().cpu().numpy.item(), 6)
    xb_std_max_= round(xb_std_max.detach().cpu().numpy.item(), 6)
    
    info = [z_mag, z_mag_std, sim_loss_, std_loss_, cov_loss_, xa_std_max_, xb_std_max_]

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




def loss_metrics(model_forward_func, augmentations, x, x_steps, scaling_func = None):
    if augmentations is not None:
        x_aug, _ = augmentations(x)
    else:
        x_aug    = x

    # predict features  
    za, zb = model_forward_func(x, x_aug)

    # predicted distances, normalise
    n_features = za.shape[-1]
    d_pred = torch.cdist(za, zb)/n_features

    # target each by each distance in steps count
    d_target = x_steps.float().unsqueeze(1)
    d_target = torch.cdist(d_target, d_target)
    if scaling_func is not None:
        d_target = scaling_func(d_target)

    reg_loss = (za**2).mean() + (zb**2).mean()
    
    # MSE loss      
    dist_loss = ((d_target - d_pred)**2).mean()

    loss = dist_loss + 0.1*reg_loss  

    # log results
    z_mag         = round(((za**2).mean()).detach().cpu().numpy().item(), 6)
    z_mag_std     = round(((za**2).std()).detach().cpu().numpy().item(), 6)
    
    dist_loss_    = round(dist_loss.mean().detach().cpu().numpy().item(), 6)
    d_target_mean = round(d_target.mean().detach().cpu().numpy().item(), 6)
    d_target_std  = round(d_target.std().detach().cpu().numpy().item(), 6)
    d_pred_mean   = round(d_pred.mean().detach().cpu().numpy().item(), 6)
    d_pred_std    = round(d_pred.std().detach().cpu().numpy().item(), 6)
    

    info = [z_mag, z_mag_std, dist_loss_, d_target_mean, d_target_std, d_pred_mean, d_pred_std]

    return loss, info



def loss_metrics_cov_var(model_forward_func, augmentations, x, x_steps, scaling_func = None):

    if augmentations is not None:
        x_aug, _ = augmentations(x)
    else:
        x_aug    = x

    # predict features  
    za, zb = model_forward_func(x, x_aug)

    # predicted distances, normalise
    n_features = za.shape[-1]
    d_pred = torch.cdist(za, zb)/n_features

    # target each by each distance in steps count
    d_target = x_steps.float().unsqueeze(1)
    d_target = torch.cdist(d_target, d_target)
    if scaling_func is not None:
        d_target = scaling_func(d_target)
    
    
    # MSE loss
    dist_loss = ((d_target - d_pred)**2).mean()

    std_loss = _loss_std(za)
    std_loss+= _loss_std(zb)
   
    # covariance loss 
    cov_loss = _loss_cov(za)
    cov_loss+= _loss_cov(zb)
   
    # total vicreg loss
    loss = 1.0*dist_loss + 1.0*std_loss + (1.0/25.0)*cov_loss


    # log results
    z_mag         = round(((za**2).mean()).detach().cpu().numpy().item(), 6)
    z_mag_std     = round(((za**2).std()).detach().cpu().numpy().item(), 6)

    dist_loss_ = round(dist_loss.detach().cpu().numpy().item(), 6)
    std_loss_  = round(std_loss.detach().cpu().numpy().item(), 6)
    cov_loss_  = round(cov_loss.detach().cpu().numpy().item(), 6)
   
    d_target_mean = round(d_target.mean().detach().cpu().numpy().item(), 6)
    d_target_std  = round(d_target.std().detach().cpu().numpy().item(), 6)
    d_pred_mean   = round(d_pred.mean().detach().cpu().numpy().item(), 6)
    d_pred_std    = round(d_pred.std().detach().cpu().numpy().item(), 6)
    

    info = [z_mag, z_mag_std, dist_loss_, std_loss_, cov_loss_, d_target_mean, d_target_std, d_pred_mean, d_pred_std]

    return loss, info
