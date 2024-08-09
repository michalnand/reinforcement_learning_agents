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


def loss_cov_var(z):    
    # variance loss
    std_loss = _loss_std(z)
   
    # covariance loss 
    cov_loss = _loss_cov(z)

    loss = 1.0*std_loss + (1.0/25.0)*cov_loss

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
        xb_aug, _ = augmentations(xb)
    else:
        xb_aug    = xb

    # obtain features
    za = model_forward_func(xa)
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
    z_mag      = round(((za**2).mean()).detach().cpu().numpy().item(), 6)
    z_mag_std  = round(((za**2).std()).detach().cpu().numpy().item(), 6)
    sim_loss_  = round(sim_loss.detach().cpu().numpy().item(), 6)
    std_loss_  = round(std_loss.detach().cpu().numpy().item(), 6)
    cov_loss_  = round(cov_loss.detach().cpu().numpy().item(), 6)
    
    info = [z_mag, z_mag_std, sim_loss_, std_loss_, cov_loss_]

    return loss, info

def _target_distances(idx_a, idx_b, scaling_func):
    # target each by each distance in steps count
    idx_a = idx_a.unsqueeze(1).float()
    idx_b = idx_b.unsqueeze(1).float()

    d_target = torch.cdist(idx_a.float(), idx_b.float())

    # target distances scaling if any (e.g. logarithmic)
    if scaling_func is not None:
        d_target_scaled = scaling_func(d_target)
    else:
        d_target_scaled = d_target  

    return d_target_scaled


def loss_vicreg_distance(model_forward_func, augmentations, x, steps, dist_scaling_func = None):

    if augmentations is not None:
        x_aug, _ = augmentations(x)
    else:
        x_aug    = x

    # obtain features
    z, d_pred = model_forward_func(x_aug)
    
    # predict distances, each by each
    d_target    = _target_distances(steps, steps, dist_scaling_func)
    
    # flatten predicted distances   
    d_pred      = d_pred.reshape((d_pred.shape[0]*d_pred.shape[1], 1))
    d_target    = d_target.reshape((d_target.shape[0]*d_target.shape[1], 1))

    # MSE loss
    dist_loss = ((d_target - d_pred)**2).mean()

    # variance loss
    std_loss = _loss_std(z)
   
    # covariance loss 
    cov_loss = _loss_cov(z)
   
    # total vicreg loss
    loss = 1.0*dist_loss + 1.0*std_loss + (1.0/25.0)*cov_loss

    

    #info for log
    z_mag         = round(((z**2).mean()).detach().cpu().numpy().item(), 6)
    z_mag_std     = round(((z**2).std()).detach().cpu().numpy().item(), 6)
    dist_loss_    = round(dist_loss.detach().cpu().numpy().item(), 6)
    std_loss_     = round(std_loss.detach().cpu().numpy().item(), 6)
    cov_loss_     = round(cov_loss.detach().cpu().numpy().item(), 6)
    
    d_target_mean = round(d_target.mean().detach().cpu().numpy().item(), 6)
    d_target_std  = round(d_target.std().detach().cpu().numpy().item(), 6)
    
    d_pred_mean   = round(d_pred.mean().detach().cpu().numpy().item(), 6)
    d_pred_std    = round(d_pred.std().detach().cpu().numpy().item(), 6)


    info = [z_mag, z_mag_std, dist_loss_, std_loss_, cov_loss_, d_target_mean, d_target_std, d_pred_mean, d_pred_std]

    return loss, info


def loss_vicreg_distance_categorical(model_forward_func, augmentations, x, steps, dist_scaling_func = None):

    if augmentations is not None:
        x_aug, _ = augmentations(x)
    else:
        x_aug    = x

    # obtain features
    z, d_pred = model_forward_func(x_aug)
    
    # predict distances, each by each
    d_target    = _target_distances(steps, steps, dist_scaling_func)
    
    # flatten predicted distances
    d_pred      = d_pred.reshape((d_pred.shape[0]*d_pred.shape[1], d_pred.shape[2]))
    d_target    = d_target.reshape((d_target.shape[0]*d_target.shape[1]))
    d_target    = d_target.long()

    # classification loss
    loss_func = torch.nn.CrossEntropyLoss()
    dist_loss = loss_func(d_pred, d_target)   

    # variance loss
    std_loss = _loss_std(z)
   
    # covariance loss 
    cov_loss = _loss_cov(z)
   
    # total vicreg loss
    loss = 1.0*dist_loss + 1.0*std_loss + (1.0/25.0)*cov_loss

    # accuracy
    acc  = (torch.argmax(d_pred, dim=1) == d_target).float()

    #info for log
    z_mag         = round(((z**2).mean()).detach().cpu().numpy().item(), 6)
    z_mag_std     = round(((z**2).std()).detach().cpu().numpy().item(), 6)
    dist_loss_    = round(dist_loss.detach().cpu().numpy().item(), 6)
    std_loss_     = round(std_loss.detach().cpu().numpy().item(), 6)
    cov_loss_     = round(cov_loss.detach().cpu().numpy().item(), 6)
    
    d_target_mean = round(d_target.float().mean().detach().cpu().numpy().item(), 6)
    d_target_std  = round(d_target.float().std().detach().cpu().numpy().item(), 6)
    
    acc_mean      = round(acc.mean().detach().cpu().numpy().item(), 6)
    acc_std       = round(acc.std().detach().cpu().numpy().item(), 6)


    info = [z_mag, z_mag_std, dist_loss_, std_loss_, cov_loss_, d_target_mean, d_target_std, acc_mean, acc_std]

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




def loss_metric(model_forward_func, x, x_steps, scaling_func):
    
    # predicted distances, each by each
    d_pred = model_forward_func(x)

    # target each by each distance in steps count
    d_target = x_steps.float().unsqueeze(1)
    d_target = torch.cdist(d_target, d_target)

    # target distances scaling if any (e.g. logarithmic)
    if scaling_func is not None:
        d_target_scaled = scaling_func(d_target)
    else:
        d_target_scaled = d_target

    # MSE loss
    loss = ((d_target_scaled - d_pred)**2).mean()

    # log results
    d_target_mean        = round(d_target.mean().detach().cpu().numpy().item(), 6)
    d_target_std         = round(d_target.std().detach().cpu().numpy().item(), 6)
    d_target_scaled_mean = round(d_target_scaled.mean().detach().cpu().numpy().item(), 6)
    d_target_scaled_std  = round(d_target_scaled.std().detach().cpu().numpy().item(), 6)
    d_pred_mean          = round(d_pred.mean().detach().cpu().numpy().item(), 6)
    d_pred_std           = round(d_pred.std().detach().cpu().numpy().item(), 6)
   
    info = [d_target_mean, d_target_std, d_target_scaled_mean, d_target_scaled_std, d_pred_mean, d_pred_std]

    return loss, info




def loss_metric_categorical(model_forward_func, x, x_steps, scaling_func):
    
    # predicted distances, each by each
    d_pred = model_forward_func(x)

    # target each by each distance in steps count
    d_target = x_steps.float().unsqueeze(1)
    d_target = torch.cdist(d_target, d_target)

    # target distances scaling if any (e.g. logarithmic)
    if scaling_func is not None:
        d_target_scaled = scaling_func(d_target)
    else:
        d_target_scaled = d_target  

    d_pred          = d_pred.reshape((d_pred.shape[0]*d_pred.shape[1], d_pred.shape[2]))
    d_target_scaled = d_target_scaled.reshape((d_target_scaled.shape[0]*d_target_scaled.shape[1]))
    d_target_scaled = d_target_scaled.long()

    # classification loss
    loss_func = torch.nn.CrossEntropyLoss()

    loss = loss_func(d_pred, d_target_scaled)   

    # accuracy
    acc  = (torch.argmax(d_pred, dim=1) == d_target_scaled).float()

    # log results
    d_target_mean        = round(d_target.mean().detach().cpu().numpy().item(), 6)
    d_target_std         = round(d_target.std().detach().cpu().numpy().item(), 6)
    d_target_scaled_mean = round(d_target_scaled.float().mean().detach().cpu().numpy().item(), 6)
    d_target_scaled_std  = round(d_target_scaled.float().std().detach().cpu().numpy().item(), 6)
    acc_mean             = round(acc.mean().detach().cpu().numpy().item(), 6)
    acc_std              = round(acc.std().detach().cpu().numpy().item(), 6)
   
    info = [d_target_mean, d_target_std, d_target_scaled_mean, d_target_scaled_std, acc_mean, acc_std]

    return loss, info


def loss_metric_distributional(model_forward_func, x, x_steps, scaling_func):
    
    # predicted distances, each by each
    d_pred_mean, d_pred_var = model_forward_func(x)

    # target each by each distance in steps count
    d_target = x_steps.float().unsqueeze(1)
    d_target = torch.cdist(d_target, d_target)

    # target distances scaling if any (e.g. logarithmic)
    if scaling_func is not None:
        d_target_scaled = scaling_func(d_target)
    else:
        d_target_scaled = d_target

    # NLL loss : negative log-likelihood of the Gaussian distribution
    loss = 0.5*torch.log(2.0*torch.pi*d_pred_var + 0.0001) 
    loss+= ((d_target_scaled - d_pred_mean)**2)/(2.0*d_pred_var + 0.0001)
    loss = loss.mean()   

    # log results
    d_target_mean        = round(d_target.mean().detach().cpu().numpy().item(), 6)
    d_target_std         = round(d_target.std().detach().cpu().numpy().item(), 6)
    d_target_scaled_mean = round(d_target_scaled.mean().detach().cpu().numpy().item(), 6)
    d_target_scaled_std  = round(d_target_scaled.std().detach().cpu().numpy().item(), 6)
    d_pred_mean_         = round(d_pred_mean.mean().detach().cpu().numpy().item(), 6)
    d_pred_std_          = round(d_pred_mean.std().detach().cpu().numpy().item(), 6)
   
    info = [d_target_mean, d_target_std, d_target_scaled_mean, d_target_scaled_std, d_pred_mean_, d_pred_std_]

    return loss, info

'''
    z.shape = (seq_length, batch_size, features_count)
    k       = exponential decay for similarity term, k > 0
'''
def loss_vicreg_temporal(model_forward_func, x, k = 0.1):
    seq_length = x.shape[0]

    # obtain features
    z = []
    for n in range(seq_length):
        z.append(model_forward_func(x[n]))

    z = torch.stack(z, dim=0)


    r = torch.arange(seq_length)
    w = torch.exp(-k*r)
    w = w.to(x.device)

   
    # distance weighted target
    z_target = (w.unsqueeze(1).unsqueeze(2)*z).sum(dim=0)/w.sum()   

    # current time step sample 
    z_now = z[0]

    # temporal invariance loss
    sim_loss = _loss_mse(z_target, z_now)

    # variance loss
    std_loss = _loss_std(z_now)
   
    # covariance loss 
    cov_loss = _loss_cov(z_now)
   
    # total vicreg loss
    loss = 1.0*sim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss


    #info for log   
    z_mag      = round(((z_now**2).mean()).detach().cpu().numpy().item(), 6)
    z_mag_std  = round(((z_now**2).std()).detach().cpu().numpy().item(), 6)
    sim_loss_  = round(sim_loss.detach().cpu().numpy().item(), 6)
    std_loss_  = round(std_loss.detach().cpu().numpy().item(), 6)
    cov_loss_  = round(cov_loss.detach().cpu().numpy().item(), 6)
    
    info = [z_mag, z_mag_std, sim_loss_, std_loss_, cov_loss_]

    return loss, info




