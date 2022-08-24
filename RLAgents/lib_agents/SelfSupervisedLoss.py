import torch

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


def contrastive_loss_mse_equivariance( model, states_a, states_b, target, regularisation_coeff = 0.0, normalise = None, augmentation = None):
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
        za, pa = model.forward_features(xa)  
        zb, pb = model.forward_features(xb) 
    else:
        za, pa = model(xa)  
        zb, pb = model(xb) 

    #predict close distance for similar, far distance for different states 
    predicted_a = ((za - pb)**2).mean(dim=1)
    predicted_b = ((zb - pa)**2).mean(dim=1)

    #MSE loss
    loss_mse = 0.5*((target - predicted_a)**2).mean()
    loss_mse+= 0.5*((target - predicted_b)**2).mean()

    #magnitude regularisation, keep magnitude in small range (optional)

    #L2 magnitude regularisation
    magnitude       = (za**2).mean() + (zb**2).mean() 
    loss_magnitude  = regularisation_coeff*magnitude

    loss = loss_mse + loss_magnitude

    #compute accuraccy in [%]
    hits = torch.logical_and(target > 0.5, predicted_a > 0.5)
    hits = torch.sum(hits.float()) 

    acc  = 100.0*hits/predicted_a.shape[0]


    return loss, magnitude.detach().to("cpu").numpy(), acc.detach().to("cpu").numpy()


def contrastive_loss_mse_all( model, states_a, states_b, target, regularisation_coeff = 0.0, normalise = None, augmentation = None):
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

    #predict distances, each by each
    distances = (torch.cdist(za, zb)**2)/za.shape[1] 

    #zeros on diagonal -> close distances, ones elsewhere
    target_   = (1.0 - torch.eye(za.shape[0])).to(za.device)
    
    #MSE loss
    loss_mse = ((target_ - distances)**2).mean()

    #magnitude regularisation, keep magnitude in small range (optional)

    #L2 magnitude regularisation
    magnitude       = (za**2).mean() + (zb**2).mean() 
    loss_magnitude  = regularisation_coeff*magnitude

    loss = loss_mse + loss_magnitude

    #compute accuraccy in [%], smallest distance should be on diagonal 
    hits = torch.argmin(distances, dim=1) == torch.arange(za.shape[0]).to(za.device)
    hits = torch.sum(hits.float()) 

    acc  = 100.0*hits/distances.shape[0]

    return loss, magnitude.detach().to("cpu").numpy(), acc.detach().to("cpu").numpy()




def contrastive_loss_mse_equivariance_all( model, states_a, states_b, target, regularisation_coeff = 0.0, normalise = None, augmentation = None):
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
        za, pa = model.forward_features(xa)  
        zb, pb = model.forward_features(xb) 
    else:
        za, pa = model(xa)  
        zb, pb = model(xb) 

    #predict distances, each by each
    distances_a = (torch.cdist(za, pb)**2)/za.shape[1] 
    distances_b = (torch.cdist(zb, pa)**2)/za.shape[1] 

    #zeros on diagonal -> close distances, ones elsewhere
    target_   = (1.0 - torch.eye(za.shape[0])).to(za.device)
    
    #MSE loss
    loss_mse = 0.5*((target_ - distances_a)**2).mean()
    loss_mse+= 0.5*((target_ - distances_b)**2).mean()

    #magnitude regularisation, keep magnitude in small range (optional)

    #L2 magnitude regularisation
    magnitude       = (za**2).mean() + (zb**2).mean() 
    loss_magnitude  = regularisation_coeff*magnitude

    loss = loss_mse + loss_magnitude

    #compute accuraccy in [%], smallest distance should be on diagonal 
    hits = torch.argmin(distances_a, dim=1) == torch.arange(za.shape[0]).to(za.device)
    hits = torch.sum(hits.float()) 

    acc  = 100.0*hits/distances_a.shape[0]

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

#barlow twins supervised
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

    #barlow loss
    eps = 10**-12

    za_norm = (za - za.mean(dim = 0))/(za.std(dim = 0) + eps)
    zb_norm = (zb - zb.mean(dim = 0))/(zb.std(dim = 0) + eps)
    
    c = torch.mm(za_norm.T, zb_norm)/za.shape[0]

    diag        = torch.eye(c.shape[0]).to(c.device)
    off_diag    = 1.0 - diag

    loss_invariance = (diag*(1.0 - c)**2).sum()
    loss_redundance = (off_diag*(c**2)).sum()/(c.shape[0] - 1)

    #magnitude regularisation, keep magnitude in small range (optional)

    #L2 magnitude regularisation
    magnitude       = (za**2).mean() + (zb**2).mean()
    loss_magnitude  = regularisation_coeff*magnitude

    loss = loss_invariance + loss_redundance + loss_magnitude

    acc = 0.0

    return loss, magnitude.detach().to("cpu").numpy(), acc



