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


def loss_vicreg(model_forward_func, augmentations, states_now, states_similar):
    xa_aug, xb_aug, mask_a_aug, mask_b_aug = augmentations(states_now, states_similar)

    za = model_forward_func(xa_aug)  
    zb = model_forward_func(xb_aug) 

   
    return loss_vicreg_direct(za, zb)




def loss_vicreg_mast(model_forward_func, augmentations, states_now, states_similar):
    xa_aug, xb_aug, mask_a_aug, mask_b_aug = augmentations(states_now, states_similar)

    #used augmentations mask
    #mask_aug.shape = (augs_count, batch_size, 1)
    mask_aug = torch.clip(mask_a_aug + mask_b_aug, 0.0, 1.0)

    print("mask_a_aug = ", mask_a_aug.shape)
    print("mask_b_aug = ", mask_b_aug.shape)
    print("mask_aug = ", mask_aug.shape)
    mask_aug = mask_aug.unsqueeze(2)
    
    # obtain features from model
    #za.shape   = (batch_size, features_count)
    #mask.shape = (augs_count, 1, features_count)
    za, mask = model_forward_func(xa_aug)  
    zb,    _ = model_forward_func(xb_aug) 


    #masked invariance term loss
    za_tmp = za.unsqueeze(0)
    zb_tmp = zb.unsqueeze(0)

    za_tmp = za_tmp*mask*mask_aug
    zb_tmp = za_tmp*mask*mask_aug

    print(">>> ", za_tmp.shape, mask.shape, mask_aug.shape)

    sim_loss = ((za_tmp - zb_tmp)**2).mean()

    sim_loss = sim_loss/mask.shape[0]



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
    

    #mask sparisity term
    sparsity_loss = torch.abs(mask)*0.1/mask.shape[0]

    # total vicreg loss
    loss = 1.0*sim_loss + 1.0*std_loss + (1.0/25.0)*cov_loss + sparsity_loss

    return loss


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