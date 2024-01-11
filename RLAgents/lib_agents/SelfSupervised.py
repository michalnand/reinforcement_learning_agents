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

    return loss


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
def loss_vicreg_jepa_direct(za, zb, pa, pb, ha, hb):
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

    #hidden information loss, minimize
    hidden_loss = torch.abs(ha).mean() + torch.abs(hb).mean() 

    print(">>> ", (za**2).mean(), (za**2).std(), (ha**2).mean(), (ha**2).std())

    # total loss, vicreg + info-min
    loss = 0.5*sim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss + 10.0*hidden_loss

    return loss



def loss_vicreg(model_forward_func, augmentations, states_a, states_b):
    xa_aug, _ = augmentations(states_a)
    xb_aug, _ = augmentations(states_b)

    za = model_forward_func(xa_aug)  
    zb = model_forward_func(xb_aug) 

    return loss_vicreg_direct(za, zb)


def loss_vicreg_jepa(model_forward_func, augmentations, states_a, states_b):
    xa_aug, _ = augmentations(states_a)
    xb_aug, _ = augmentations(states_b)

    za, zb, pa, pb, ha, hb = model_forward_func(xa_aug, xb_aug)  

    return loss_vicreg_jepa_direct(za, zb, pa, pb, ha, hb)


def loss_vicreg_temporal(model_forward_func, augmentations, states_a, states_b, hidden_a, hidden_b, max_seq_length):
    xa_aug, _ = augmentations(states_a)
    xb_aug, _ = augmentations(states_b)  

    za, _ = model_forward_func(xa_aug, hidden_a, max_seq_length)  
    zb, _ = model_forward_func(xb_aug, hidden_b, max_seq_length) 

    #reshape for vicreg loss (batch, seq*features)
    za  = za.reshape((za.shape[0], za.shape[1]*za.shape[2]))
    zb  = zb.reshape((zb.shape[0], zb.shape[1]*zb.shape[2]))

    #print(">>> z = ", za.shape, zb.shape)

    return loss_vicreg_direct(za, zb)



def loss_vicreg_jepa_temporal(model_forward_func, augmentations, states_a, states_b, hidden_a, hidden_b, max_seq_length):
    xa_aug, _ = augmentations(states_a)
    xb_aug, _ = augmentations(states_b)  

    za, zb, pa, pb, ha, hb = model_forward_func(xa_aug, xb_aug, hidden_a, hidden_b, max_seq_length)  

    #reshape for vicreg loss (batch, seq*features)
    za  = za.reshape((za.shape[0], za.shape[1]*za.shape[2]))
    zb  = zb.reshape((zb.shape[0], zb.shape[1]*zb.shape[2]))

    pa  = pa.reshape((pa.shape[0], pa.shape[1]*pa.shape[2]))
    pb  = pb.reshape((pb.shape[0], pb.shape[1]*pb.shape[2]))

    ha  = ha.reshape((ha.shape[0], ha.shape[1]*ha.shape[2]))
    hb  = hb.reshape((hb.shape[0], hb.shape[1]*hb.shape[2]))

    #print(">>> ", za.shape, pa.shape, ha.shape)
    #print(">>> ", zb.shape, pb.shape, hb.shape)
    #print("\n\n")

    return loss_vicreg_jepa_direct(za, zb, pa, pb, ha, hb)

