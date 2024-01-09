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
    in final, goal is to minimize this ha, hb, to provide minimum information

5, predictor is computed as : 
        pa = p(za, hb)
        pb = p(zb, ha)

        and invariance (mse loss) is called :
        MSE(za - pb), MSE(zb - pa)

        essence of JEPA is : p-model can easily predict za, 
        because pa contains not only augmented context from za,
        but also "cheating" information about zb (called hidden information)
        
        however, h-features are forced to minimize information, which leads to good
        za, zb features, and h contains only the necessary information 
        from za which can't be extracted from zb (and vice versa)
'''
def loss_jepa_direct(za, zb, pa, pb, ha, hb):
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

    # total vicreg loss
    loss = 1.0*sim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss + 10.0*hidden_loss

    return loss



def loss_vicreg_contrastive_direct(za, zb, steps_a, steps_b):
    eps = 0.0001 
 
    # invariance loss, similarity
    sim_loss = ((za - zb)**2).mean()
    
    # disimilarity loss (contrastive term), compare random pairs
    idx_a     = torch.randperm(za.shape[0])
    idx_b     = torch.randperm(zb.shape[0])   
    
    # compute distance
    distance_req  = torch.abs(steps_a[idx_a] - steps_b[idx_b])
    distance_req  = torch.log(1.0 + distance_req)
 
    distance       = ((za[idx_a] - zb[idx_b])**2).mean(dim=1)

    '''
    print((steps_a[idx_a])[0:5], (steps_b[idx_b])[0:5])
    print(distance_req[0:5])
    print(distance[0:5])
    print("\n\n")  
    '''

    #dsim_loss = torch.mean(torch.relu(distance_req - distance))
    dsim_loss = ((distance_req - distance)**2).mean()

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
    loss = 1.0*sim_loss + 1.0*dsim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss

    return loss


def loss_vicreg(model_forward_func, augmentations, states_a, states_b):
    xa_aug, _ = augmentations(states_a)
    xb_aug, _ = augmentations(states_b)

    za = model_forward_func(xa_aug)  
    zb = model_forward_func(xb_aug) 

    return loss_vicreg_direct(za, zb)


def loss_jepa(model_forward_func, augmentations, states_a, states_b):
    xa_aug, _ = augmentations(states_a)
    xb_aug, _ = augmentations(states_b)

    za, zb, pa, pb, ha, hb = model_forward_func(xa_aug, xb_aug)  

    return loss_jepa_direct(za, zb, pa, pb, ha, hb)


def loss_vicreg_contrastive(model_forward_func, augmentations, states_a, states_b, episode_steps_a, episode_steps_b):
    xa_aug, _ = augmentations(states_a)
    xb_aug, _ = augmentations(states_b)

    za = model_forward_func(xa_aug)  
    zb = model_forward_func(xb_aug) 

    return loss_vicreg_contrastive_direct(za, zb, episode_steps_a, episode_steps_b)


def loss_vicreg_temporal(model_forward_func, augmentations, states_a, states_b, hidden_a, hidden_b, max_seq_length):
    xa_aug, _ = augmentations(states_a)
    xb_aug, _ = augmentations(states_b)  

    za, _ = model_forward_func(xa_aug, hidden_a, max_seq_length)  
    zb, _ = model_forward_func(xb_aug, hidden_b, max_seq_length) 

    #reshape for vicreg loss (batch*seq, features)
    za  = za.reshape((za.shape[0]*za.shape[1], za.shape[2]))
    zb  = zb.reshape((zb.shape[0]*zb.shape[1], zb.shape[2]))

    return loss_vicreg_direct(za, zb)


def loss_vicreg_mast(model_forward_func, augmentations_func, states_a, states_b):
    
    xa_aug, used_aug_a = augmentations_func(states_a) 
    xb_aug, used_aug_b = augmentations_func(states_b)

    used_aug = torch.clip(used_aug_a + used_aug_b, 0.0, 1.0)
    
    # obtain features from model
    za, mask_w  = model_forward_func(xa_aug)  
    zb, _       = model_forward_func(xb_aug) 

    #used_aug = (augs_count, batch_size, 1)
    used_aug = used_aug.unsqueeze(2)

    #mask_w = (augs_count, features_count, 1) 
    mask_w = mask_w.unsqueeze(1) 

    #zx_tmp = (1, batch_size, features_count)
    za_tmp = za.unsqueeze(0)
    zb_tmp = zb.unsqueeze(0)

    #masked features
    za_tmp = za_tmp*mask_w*used_aug
    zb_tmp = zb_tmp*mask_w*used_aug

    #loss term to prevent mask_w collapse to zero
    mask_tmp  = torch.relu(1.0 - mask_w.sum(dim=0))
    loss_mask = (mask_tmp**2).mean()

    # masked invariance loss
    sim_loss = ((za_tmp - zb_tmp)**2).mean() 

    eps = 0.0001 

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

    # final vicreg loss
    loss = 1.0*sim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss + 10.0*loss_mask

    return loss

'''
def loss_vicreg_mast(model_forward_func, augmentations, states_a, states_b):
    
    xa_aug, used_aug_a = augmentations(states_a)
    xb_aug, used_aug_b = augmentations(states_b)

    used_aug = torch.clip(used_aug_a + used_aug_b, 0.0, 1.0)
    
    # obtain features from model
    za, mask_w  = model_forward_func(xa_aug)  
    zb, _       = model_forward_func(xb_aug) 

    #add extra "augmentation" to prevent loss collapse
    #used_aug = (augs_count + 1, batch_size, 1)
    ones     = torch.ones((1, used_aug.shape[1])).to(used_aug.device)
    used_aug = torch.cat([used_aug, ones], dim=0)
    used_aug = used_aug.unsqueeze(2)

    #mask_w = (augs_count + 1, features_count, 1) 
    mask_add = torch.relu(1.0 - mask_w.sum(dim=0).unsqueeze(0).detach())
    mask_w = torch.cat([mask_w, mask_add], dim=0)
    mask_w = mask_w.unsqueeze(1) 

    #zx_tmp = (1, batch_size, features_count)
    za_tmp = za.unsqueeze(0)
    zb_tmp = zb.unsqueeze(0)

    #masked features
    za_tmp = za_tmp*mask_w*used_aug
    zb_tmp = zb_tmp*mask_w*used_aug

    # masked invariance loss
    sim_loss = ((za_tmp - zb_tmp)**2).mean() 

    eps = 0.0001 

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

    # final vicreg loss
    loss = 1.0*sim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss

    return loss
'''

'''
#constructor theory loss
#for phylosophical stuff read https://www.constructortheory.org
#train model for causality prediction, a.k.a. if state transition is posible
def loss_constructor(model_forward_func, augmentations, states_now, states_next, states_similar, states_random, actions, relations):
    batch_size  = states_now.shape[0]

    #classes target IDs
    #0 : state_now,  state_random, two different states
    #1 : state_now,  state_next, two consecutive states
    #2 : state_next, state_now, two inverted consecutive states
    labels  = torch.randint(0, 3, (batch_size, ), dtype=torch.long).to(states_now.device)
 
    #mix states coresponding to three cases
    select  = labels.unsqueeze(1).unsqueeze(2).unsqueeze(3)

    xa      = (select == 0)*states_now    + (select == 1)*states_now  + (select == 2)*states_next
    xb      = (select == 0)*states_random + (select == 1)*states_next + (select == 2)*states_now

    #states augmentation
    if augmentations is not None:
        xa = augmentations(xa) 
        xb = augmentations(xb)

    transition_pred = model_forward_func(xa, xb)

    #classification loss
    loss_func       = torch.nn.CrossEntropyLoss()
    loss            = loss_func(transition_pred, labels)
    
    
    #compute accuracy, per class
    labels_pred = torch.argmax(transition_pred.detach(), dim=1)

    acc = numpy.zeros((3, ))

    for class_id in range(acc.shape[0]):
        total_count = (labels == class_id).sum()
        hits_count  = ((labels == class_id)*(labels_pred == class_id)).sum()

        acc[class_id] = 100.0*hits_count/(total_count + 10**-6)

    return loss, acc
'''