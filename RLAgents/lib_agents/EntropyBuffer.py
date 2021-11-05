import torch

class EntropyBuffer:

    def __init__(self, buffer_size, envs_count, state_shape, device, downsample = 4):

        self.buffer_size    = buffer_size
        self.envs_count     = envs_count

        count               = (state_shape[1]//downsample)*(state_shape[2]//downsample)

        self.buffer         = torch.zeros((self.buffer_size, self.envs_count, count)).to(device)
        self.zeros          = torch.zeros((count)).to(device)

        #downsampling model
        self.layers = [
            torch.nn.AvgPool2d(downsample, downsample),
            torch.nn.Flatten()
        ]

        self.model = torch.nn.Sequential(*self.layers)
        self.model.to(device) 
        self.model.eval() 

        self.ptr = 0

    def add(self, state):
        #add new state
        x = state[:,0,:,:].unsqueeze(1)
        self.buffer[self.ptr] = self.model(x)

        self.ptr = (self.ptr+1)%self.buffer_size

    def compute(self):
        #compute variance
        variance = torch.var(self.buffer, dim=0).mean(dim=1)

        print(variance.shape)
        return variance

    def clear(self, env_idx):
        #clear buffer for given env
        for i in range(self.buffer_size):
            self.buffer[i][env_idx] = self.zeros.clone()
