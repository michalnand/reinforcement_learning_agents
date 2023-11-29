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


def loss_vicreg(model_forward_func, augmentations, states_a, states_b, episode_steps_a, episode_steps_b):
    xa_aug, _ = augmentations(states_a)
    xb_aug, _ = augmentations(states_b)

    za = model_forward_func(xa_aug)  
    zb = model_forward_func(xb_aug) 

    return loss_vicreg_direct(za, zb)

def loss_vicreg_contrastive(model_forward_func, augmentations, states_a, states_b, episode_steps_a, episode_steps_b):
    xa_aug, _ = augmentations(states_a)
    xb_aug, _ = augmentations(states_b)

    za = model_forward_func(xa_aug)  
    zb = model_forward_func(xb_aug) 

    return loss_vicreg_contrastive_direct(za, zb, episode_steps_a, episode_steps_b)



'''
def loss_vicreg_mast(model_forward_func, augmentations, states_now, states_similar):
    xa_aug, xb_aug, mask_a_aug, mask_b_aug = augmentations(states_now, states_similar)
    
    #used augmentations mask
    #mask_aug.shape = (augs_count, batch_size, 1)
    mask_aug = torch.clip(mask_a_aug + mask_b_aug, 0.0, 1.0)
    mask_aug = mask_aug.unsqueeze(2)
    
    # obtain features from model
    #za.shape   = (batch_size, features_count)
    #mask_w.shape = (augs_count, 1, features_count)
    za, mask_w  = model_forward_func(xa_aug)  
    zb,    _    = model_forward_func(xb_aug) 

    #masked invariance term loss
    za_tmp = za.unsqueeze(0)
    zb_tmp = zb.unsqueeze(0)

    mask_w_tmp = torch.nn.functional.softmax(mask_w, dim=0)

    za_tmp = za_tmp*mask_w_tmp*mask_aug
    zb_tmp = zb_tmp*mask_w_tmp*mask_aug

    sim_loss = ((za_tmp - zb_tmp)**2).mean()
    

    #vicreg loss    
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
    
    # total vicreg loss
    loss = 1.0*sim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss
    

    return loss
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