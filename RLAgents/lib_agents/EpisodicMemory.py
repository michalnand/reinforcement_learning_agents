import torch

class EpisodicMemory:
    def __init__(self, size, downsample = -1):
        self.size       = size
        self.downsample = downsample

        self.idx        = 0
        self.buffer     = None
         
        
    def reset(self, state_t): 
        if self.downsample != -1:
            self.layer_downsample = torch.nn.AvgPool2d((self.downsample, self.downsample), (self.downsample, self.downsample))
            self.layer_downsample.to(state_t.device)
        else:
            self.layer_downsample = None

        tmp_t = self._preprocess(state_t)

        self.buffer = torch.zeros((self.size, ) + tmp_t.shape).float().to(tmp_t.device)
        for i in range(self.size): 
            self.buffer[i] = tmp_t.float()

        self.idx    = 0
        self.h_new  = torch.zeros(tmp_t.shape).float().to(tmp_t.device)

        
    def add(self, state_t): 
        if self.buffer is None:
            self.reset(state_t) 

        tmp_t = self._preprocess(state_t)

        #add to buffer
        self.buffer[self.idx] = tmp_t.float()
        self.idx = (self.idx+1)%self.size

        #compute variance
        self.h_old  = self.h_new.clone()
        self.h_new  = self.buffer.var(axis=0) 
        
        h   = self.h_new - self.h_old
        h   = h.mean().detach().to("cpu").numpy()

        #return
        return h

    #downsample and flatten
    def _preprocess(self, x):
        y = x.unsqueeze(0)

        if self.layer_downsample is not None:
            y = self.layer_downsample(y)
        
        y = torch.flatten(y).squeeze(0)
        return y 
