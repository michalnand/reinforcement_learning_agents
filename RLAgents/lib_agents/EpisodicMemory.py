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

        self.buffer = torch.zeros((self.size, ) + tmp_t.shape).to(tmp_t.device)
        for i in range(self.size): 
            self.buffer[i] = tmp_t.clone()

        self.idx = 0

        
    def add(self, state_t): 
        if self.buffer is None:
            self.reset(state_t)

        tmp_t = self._preprocess(state_t)

        #compute current variance
        h0  = self.buffer.var(axis=0) 
        h0  = h0.mean().detach().to("cpu").numpy()

        #add to buffer
        self.buffer[self.idx] = tmp_t.clone()
        self.idx = (self.idx+1)%self.size

        #compute new variance
        h1  = self.buffer.var(axis=0) 
        h1  = h1.mean().detach().to("cpu").numpy()

        #return
        return h1, h1 - h0

    #downsample and flatten
    def _preprocess(self, x):
        y = x.unsqueeze(0)

        if self.layer_downsample is not None:
            y = self.layer_downsample(y)
        
        y = torch.flatten(y).squeeze(0)
        return y 
