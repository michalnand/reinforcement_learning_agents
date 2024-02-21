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



# cross correlation loss 
def _loss_cross(xa, xb):

    xa_norm = xa - xa.mean(dim=0)
    xb_norm = xb - xb.mean(dim=0)
    cov = (xa_norm @ xb_norm.T) / (xa.shape[1] - 1.0)
    
    loss = (cov**2).mean()

    return loss





def loss_vicreg(model_forward_func, augmentations, xa, xb):
    xa_aug, _ = augmentations(xa) 
    xb_aug, _ = augmentations(xb)

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
    
    info = [z_mag, z_mag_std]

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



def loss_vicreg_jepa_proj(model_forward_func, augmentations, xa, xb, hidden_coeff = 0.01):
    xa_aug, _ = augmentations(xa)
    xb_aug, _ = augmentations(xb)

    za, zb, pa, pb, ha, hb, proj_a, proj_b = model_forward_func(xa_aug, xb_aug)  

    # invariance loss
    sim_loss = _loss_mse(za, pb)
    sim_loss+= _loss_mse(zb, pa) 

    # variance loss
    std_loss = _loss_std(proj_a)
    std_loss+= _loss_std(proj_b) 

    # covariance loss 
    cov_loss = _loss_cov(proj_a)
    cov_loss+= _loss_cov(proj_b)

    #hidden information loss, enforce sparsity, and minimize batch-wise variance
    h_mag = torch.abs(ha).mean() + torch.abs(hb).mean() 
    h_std = (ha.std(dim=0)).mean() + (hb.std(dim=0)).mean()
    hidden_loss = h_mag + h_std

    cross_loss = _loss_cross(za, zb)

    # total loss, vicreg + info-min
    loss = 0.5*sim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss + hidden_coeff*hidden_loss

    #info for log
    z_mag     = round(((za**2).mean()).detach().cpu().numpy().item(), 6)
    z_mag_std = round(((za**2).std()).detach().cpu().numpy().item(), 6)
    h_mag     = round(((ha**2).mean()).detach().cpu().numpy().item(), 6)
    h_mag_std = round(((ha**2).std()).detach().cpu().numpy().item(), 6)

    cross = round(cross_loss.detach().cpu().numpy().item(), 6)

    info = [z_mag, z_mag_std, h_mag, h_mag_std, cross]

    return loss, info






def loss_vicreg_jepa_cross(model_forward_func, augmentations, xa, xb, hidden_coeff = 0.01):
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

    cross_loss = _loss_cross(za, zb)

    #hidden information loss, enforce sparsity, and minimize batch-wise variance
    h_mag = torch.abs(ha).mean() + torch.abs(hb).mean() 
    h_std = (ha.std(dim=0)).mean() + (hb.std(dim=0)).mean()
    hidden_loss = h_mag + h_std

    # total loss, vicreg + info-min
    loss = 0.5*sim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss + 1.0*cross_loss + hidden_coeff*hidden_loss

    #info for log
    z_mag     = round(((za**2).mean()).detach().cpu().numpy().item(), 6)
    z_mag_std = round(((za**2).std()).detach().cpu().numpy().item(), 6)
    h_mag     = round(((ha**2).mean()).detach().cpu().numpy().item(), 6)
    h_mag_std = round(((ha**2).std()).detach().cpu().numpy().item(), 6)

    cross = round(cross_loss.detach().cpu().numpy().item(), 6)

    info = [z_mag, z_mag_std, h_mag, h_mag_std, cross]

    return loss, info





def loss_vicreg_jepa_ema(model_forward_func, augmentations, xa, xb, hidden_coeff = 0.01):
    xa_aug, _ = augmentations(xa)
    xb_aug, _ = augmentations(xb)

    z_pred, z_target, p, h = model_forward_func(xa_aug, xb_aug)  

    # invariance loss
    sim_loss = _loss_mse(p, z_target)

    # variance loss
    std_loss = _loss_std(z_pred)

    # covariance loss 
    cov_loss = _loss_cov(z_pred)

    #hidden information loss, enforce sparsity, and minimize batch-wise variance
    h_mag = torch.abs(h).mean()
    h_std = h.std(dim=0).mean()
    hidden_loss = h_mag + h_std

    # total loss, vicreg + info-min
    loss = 1.0*sim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss + hidden_coeff*hidden_loss

    cross_loss = _loss_cross(z_pred, z_target)
    
    #info for log
    z_mag     = round(((z_pred**2).mean()).detach().cpu().numpy().item(), 6)
    z_mag_std = round(((z_pred**2).std()).detach().cpu().numpy().item(), 6)
    h_mag     = round(((h**2).mean()).detach().cpu().numpy().item(), 6)
    h_mag_std = round(((h**2).std()).detach().cpu().numpy().item(), 6)

    cross = round(cross_loss.detach().cpu().numpy().item(), 6)

    info = [z_mag, z_mag_std, h_mag, h_mag_std, cross]



    return loss, info


if __name__ == "__main__":

    batch_size = 5
    features   = 8
    

    x_initial = torch.randn((batch_size, features))

    xa = torch.nn.parameter.Parameter(x_initial.detach(), requires_grad=True) 
    xb = torch.nn.parameter.Parameter(x_initial.detach(), requires_grad=True) 

    optim = torch.optim.Adam([xa, xb], lr=0.1)

    print(xa)
    print(xb)
    print( ((xa@xb.T)**2).mean())
    print("\n\n\n\n")
    for i in range(10000):

        loss = _loss_cross(xa, xb)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if i%100 == 0:
            print(loss)


    print(xa)
    print(xb)
    print( ((xa@xb.T)**2).mean())
    print("\n\n\n\n")