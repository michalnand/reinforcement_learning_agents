import torch
import numpy

def _off_diagonal(x):
    mask = 1.0 - torch.eye(x.shape[0], device=x.device)
    return x*mask 

def loss_vicreg_direct(za, zb):
    eps = 0.0001 
 
    # invariance loss
    sim_loss = ((za - zb)**2).mean()

    # variance loss
    std_za = torch.sqrt(za.var(dim=0) + eps)
    std_zb = torch.sqrt(zb.var(dim=0) + eps) 
    
    std_loss = torch.mean(torch.relu(1.0 - std_za)) 
    std_loss+= torch.mean(torch.relu(1.0 - std_zb))
   
    # covariance loss 
    za_norm = za - za.mean(dim=0)
    zb_norm = zb - zb.mean(dim=0)
    cov_za = (za_norm.T @ za_norm) / (za.shape[0] - 1.0)
    cov_zb = (zb_norm.T @ zb_norm) / (zb.shape[0] - 1.0)
    
    cov_loss = _off_diagonal(cov_za).pow_(2).sum()/za.shape[1] 
    cov_loss+= _off_diagonal(cov_zb).pow_(2).sum()/zb.shape[1]

    # total vicreg loss
    loss = 1.0*sim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss

 
    #info for log
    z_mag     = round(((za**2).mean()).detach().cpu().numpy().item(), 6)
    z_mag_std = round(((za**2).std()).detach().cpu().numpy().item(), 6)
    
    info = [z_mag, z_mag_std]

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
def loss_vicreg_jepa_direct(za, zb, pa, pb, ha, hb, hidden_coeff = 0.01):
    eps = 0.0001 

 
    # invariance loss
    sim_loss = ((za - pb)**2).mean() + ((zb - pa)**2).mean()

    # variance loss
    std_za = torch.sqrt(za.var(dim=0) + eps)
    std_zb = torch.sqrt(zb.var(dim=0) + eps) 
    
    std_loss = torch.mean(torch.relu(1.0 - std_za)) 
    std_loss+= torch.mean(torch.relu(1.0 - std_zb))
   
    # covariance loss 
    za_norm = za - za.mean(dim=0)
    zb_norm = zb - zb.mean(dim=0)
    cov_za = (za_norm.T @ za_norm) / (za.shape[0] - 1.0)
    cov_zb = (zb_norm.T @ zb_norm) / (zb.shape[0] - 1.0)
     
    cov_loss = _off_diagonal(cov_za).pow_(2).sum()/za.shape[1] 
    cov_loss+= _off_diagonal(cov_zb).pow_(2).sum()/zb.shape[1]

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

    info = [z_mag, z_mag_std, h_mag, h_mag_std]

    return loss, info




def loss_vicreg_jepa_proj_direct(za, zb, pa, pb, ha, hb, proj_za, proj_zb, hidden_coeff = 0.01):
    eps = 0.0001 

    print(za.shape, pa.shape, ha.shape, proj_za.shape, proj_zb.shape)

 
    # invariance loss
    sim_loss = ((za - pb)**2).mean() + ((zb - pa)**2).mean()

    # variance loss
    std_za = torch.sqrt(proj_za.var(dim=0) + eps)
    std_zb = torch.sqrt(proj_zb.var(dim=0) + eps) 
    
    std_loss = torch.mean(torch.relu(1.0 - std_za)) 
    std_loss+= torch.mean(torch.relu(1.0 - std_zb))
   
    # covariance loss 
    za_norm = proj_za - proj_za.mean(dim=0)
    zb_norm = proj_zb - proj_zb.mean(dim=0)
    cov_za = (za_norm.T @ za_norm) / (proj_za.shape[0] - 1.0)
    cov_zb = (zb_norm.T @ zb_norm) / (proj_zb.shape[0] - 1.0)
     
    cov_loss = _off_diagonal(cov_za).pow_(2).sum()/proj_za.shape[1] 
    cov_loss+= _off_diagonal(cov_zb).pow_(2).sum()/proj_zb.shape[1]

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

    info = [z_mag, z_mag_std, h_mag, h_mag_std]

    return loss, info
 


def loss_vicreg(model_forward_func, augmentations, states_a, states_b):
    xa_aug, _ = augmentations(states_a)
    xb_aug, _ = augmentations(states_b)

    za = model_forward_func(xa_aug)
    zb = model_forward_func(xb_aug)

    return loss_vicreg_direct(za, zb)


def loss_vicreg_jepa(model_forward_func, augmentations, states_a, states_b, hidden_coeff = 0.01):
    xa_aug, _ = augmentations(states_a)
    xb_aug, _ = augmentations(states_b)

    res = model_forward_func(xa_aug, xb_aug)  

    if len(res) == 6:
        za, zb, pa, pb, ha, hb = res
        loss = loss_vicreg_jepa_direct(za, zb, pa, pb, ha, hb, hidden_coeff)
    elif len(res) == 8:
        za, zb, pa, pb, ha, hb, proj_za, proj_zb = res
        loss = loss_vicreg_jepa_proj_direct(za, zb, pa, pb, ha, hb, proj_za, proj_zb, hidden_coeff)

    return loss