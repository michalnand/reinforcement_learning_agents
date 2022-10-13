import torch

def accuraccy(za, zb):
    #info NCE loss, CE with target classes on diagonal
    similarity  = torch.cdist(za, zb)
    target      = torch.arange(za.shape[0]).to(za.device)
     
    #compute accuraccy in [%]
    hits = torch.argmin(similarity, dim=1) == target
    hits = torch.sum(hits.float()) 

    acc  = 100.0*hits/similarity.shape[0]

    return acc.detach().to("cpu").numpy()

def contrastive_loss_mse(model, states_a, states_b, target, regularisation_coeff = 0.0, normalise = None, augmentation = None):
    xa = states_a.clone()
    xb = states_b.clone()

    #normalise states
    if normalise is not None:
        xa = normalise(xa) 
        xb = normalise(xb)

    #states augmentation
    if augmentation is not None:
        xa = augmentation(xa) 
        xb = augmentation(xb)

    #obtain features from model
    if hasattr(model, "forward_features"):
        za = model.forward_features(xa)  
        zb = model.forward_features(xb) 
    else:
        za = model(xa)  
        zb = model(xb) 

    #predict close distance for similar, far distance for different states 
    predicted = ((za - zb)**2).mean(dim=1)

    #MSE loss
    loss_mse = ((target - predicted)**2).mean()

    #magnitude regularisation, keep magnitude in small range (optional)

    #L2 magnitude regularisation
    magnitude       = (za**2).mean() + (zb**2).mean() 
    loss_magnitude  = regularisation_coeff*magnitude

    loss = loss_mse + loss_magnitude

    #compute accuraccy in [%]
    hits = torch.logical_and(target > 0.5, predicted > 0.5)
    hits = torch.sum(hits.float()) 

    acc  = 100.0*hits/predicted.shape[0]


    return loss, magnitude.detach().to("cpu").numpy(), acc.detach().to("cpu").numpy()


def contrastive_loss_nce( model, states_a, states_b, target, regularisation_coeff = 0.0, normalise = None, augmentation = None):
    xa = states_a.clone()
    xb = states_a.clone()

    #normalise states
    if normalise is not None:
        xa = normalise(xa) 
        xb = normalise(xb)

    #states augmentation
    if augmentation is not None:
        xa = augmentation(xa) 
        xb = augmentation(xb)

    #obtain features from model
    if hasattr(model, "forward_features"):
        za = model.forward_features(xa)  
        zb = model.forward_features(xb) 
    else:
        za = model(xa)  
        zb = model(xb) 

    #normalise
    eps = 10**-12

    za_norm = (za - za.mean(dim = 0))/(za.std(dim = 0) + eps)
    zb_norm = (zb - zb.mean(dim = 0))/(zb.std(dim = 0) + eps)
    
    #info NCE loss, CE with target classes on diagonal
    similarity      = torch.matmul(za_norm, zb_norm.T)/za_norm.shape[1]
    lf              = torch.nn.CrossEntropyLoss()
    target          = torch.arange(za_norm.shape[0]).to(za_norm.device)
    loss_info_max   = lf(similarity, target)
    
    #magnitude regularisation, keep magnitude in small range (optional)

    #L2 magnitude regularisation
    magnitude       = (za**2).mean() + (zb**2).mean()
    loss_magnitude  = regularisation_coeff*magnitude

    loss = loss_info_max + loss_magnitude

    #compute accuraccy in [%]
    hits = torch.argmax(similarity, dim=1) == target
    hits = torch.sum(hits.float()) 

    acc  = 100.0*hits/similarity.shape[0]

    return loss, magnitude.detach().to("cpu").numpy(), acc.detach().to("cpu").numpy()

#barlow twins self supervised
def contrastive_loss_barlow( model, states_a, states_b, target, regularisation_coeff = 0.0, normalise = None, augmentation = None):
    xa = states_a.clone()
    xb = states_a.clone()

    #normalise states
    if normalise is not None:
        xa = normalise(xa) 
        xb = normalise(xb)

    #states augmentation
    if augmentation is not None:
        xa = augmentation(xa) 
        xb = augmentation(xb)
 
    #obtain features from model
    if hasattr(model, "forward_features"):
        za = model.forward_features(xa)  
        zb = model.forward_features(xb) 
    else:
        za = model(xa)  
        zb = model(xb) 


    eps = 10**-12

    za_norm = (za - za.mean(dim = 0, keepdim=True))/(za.std(dim = 0, keepdim=True) + eps)
    zb_norm = (zb - zb.mean(dim = 0, keepdim=True))/(zb.std(dim = 0, keepdim=True) + eps) 
 
    c = torch.mm(za_norm.T, zb_norm)/za.shape[0]
 
    diag        = torch.eye(c.shape[0]).to(c.device)
    off_diag    = 1.0 - diag  

    loss_invariance = (diag*(1.0 - c)**2).mean()
    loss_redundance = (off_diag*(c**2)).mean()/(c.shape[0] - 1)
 

    #magnitude regularisation, keep magnitude in small range (optional)

    #L2 magnitude regularisation
    magnitude       = (za**2).mean() + (zb**2).mean()
    loss_magnitude  = regularisation_coeff*magnitude

    loss = loss_invariance + loss_redundance + loss_magnitude

    acc = accuraccy(za, zb)

    return loss, magnitude.detach().to("cpu").numpy(), acc




def contrastive_loss_vicreg(model, states_a, states_b, target, regularisation_coeff = 0.0, normalise = None, augmentation = None):
    xa = states_a.clone()
    xb = states_a.clone()

    #normalise states
    if normalise is not None:
        xa = normalise(xa) 
        xb = normalise(xb)

    #states augmentation
    if augmentation is not None:
        xa = augmentation(xa) 
        xb = augmentation(xb)
 
    #obtain features from model
    if hasattr(model, "forward_features"):
        za = model.forward_features(xa)  
        zb = model.forward_features(xb) 
    else:
        za = model(xa)  
        zb = model(xb) 

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
    
    cov_loss = off_diagonal(cov_za).pow_(2).sum()/za.shape[1] 
    cov_loss+= off_diagonal(cov_zb).pow_(2).sum()/zb.shape[1]
    
    #magnitude regularisation, keep magnitude in small range (optional)
    
    #L2 magnitude regularisation
    magnitude       = (za**2).mean() + (zb**2).mean()

    loss = 10.0*sim_loss + 10.0*std_loss + 1.0*cov_loss + regularisation_coeff*magnitude

    return loss, magnitude.detach().to("cpu").numpy(), 0


def symmetry_loss_rules(model, states_now, states_next, actions, regularisation_coeff = 0.0, normalise = None, augmentation = None):
    xa = states_now.clone()
    xb = states_next.clone()

    #normalise states
    if normalise is not None:
        xa = normalise(xa) 
        xb = normalise(xb)

    #states augmentation
    if augmentation is not None:
        xa = augmentation(xa) 
        xb = augmentation(xb)

    #obtain features from model and predict action (inverse model)
    z, actions_pred = model.forward_features(states_now, states_next)

    #correlation, same actions should have similar features
    #this is symmetry in actions, same action behaves same across whole state space
    similarity  = torch.matmul(z, z.T)/z.shape[1]

    target = (actions.unsqueeze(0) == actions.unsqueeze(1)).float()

    loss_symmetry = ((target - similarity)**2).mean()

    #predict action from two consectuctive states
    lf = torch.nn.CrossEntropyLoss()
    loss_inverse = lf(actions_pred, actions)

    #L2 magnitude regularisation
    magnitude = (z**2).mean()

    loss = loss_symmetry + loss_inverse + regularisation_coeff*magnitude

    #compute accuraccy in [%], use only true positive (since true negative are trivial to hit)
    hits = torch.argmax(similarity, dim=1) == target
    hits = torch.sum(hits.float()) 

    acc  = 100.0*hits/similarity.shape[0]

    return loss, magnitude.detach().to("cpu").numpy(), acc.detach().to("cpu").numpy()


