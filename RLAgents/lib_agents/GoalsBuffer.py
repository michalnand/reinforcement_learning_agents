import torch

class GoalsBuffer:  

    def __init__(self, buffer_size, shape, reach_threshold, mastering_threshold):

        downsample           = 2

        shape_down           = (shape[0], shape[1]//downsample, shape[2]//downsample)
        self.features_count  = shape_down[0]*shape_down[1]*shape_down[2]

        self.reach_threshold     = reach_threshold
        self.mastering_threshold  = mastering_threshold


        self.downsample     = torch.nn.AvgPool2d(downsample, downsample)

        self.states_b       = torch.zeros((buffer_size, self.features_count))
        self.score_b        = torch.zeros((buffer_size, )) 
        self.steps_b        = torch.zeros((buffer_size, )) 
        self.mastered_b     = torch.zeros((buffer_size, ))

        self.current_target = torch.zeros((buffer_size, ))

        self.mastered_b[0]  = 1.0

        self.current_idx    = 0


    

    def update(self, states, steps, score_sum):
        batch_size  = states.shape[0]
        states_t    = torch.from_numpy(states).float()
        steps_t     = torch.from_numpy(steps).float()
        score_sum_t = torch.from_numpy(score_sum).float()

        #flatten
        states_down     = self.downsample(states_t)
        states_fltn     = states_down.reshape((states_down.shape[0], self.features_count))

        if self.current_idx == 0:
            self._add_new_state(states_fltn[0], 0, -10**6)
            self.mastered_b[0]  = 1.0

        used_states = self.states_b[0:self.current_idx]

        #mean distances
        distances = torch.cdist(states_fltn, used_states)/self.features_count

        #find closest distances and indices
        closest_val, closest_ids = torch.min(distances, dim=1)


        #add new states if threshold reached and worth to add
        for i in range(batch_size):
            if score_sum_t[i] > torch.max(self.score_b):
                self._add_new_state(states_fltn[i], steps_t[i], score_sum_t[i])
                break

        reached_any = closest_val <= self.reach_threshold
        
        current_target_id   = self._get_non_mastered_target()
        reached             = torch.logical_and(torch.logical_and(reached_any, current_target_id == closest_ids), score_sum_t == self.score_b[current_target_id])

        #reward +1 for reaching target
        reached_reward = 1.0*reached

        #reward +1 for less steps
        steps_tmp           = self.steps_b[closest_ids]
        less_steps_reward   = torch.round(steps_t) < torch.round(steps_tmp)

        #update with better count
        self.steps_b[closest_ids] = less_steps_reward*steps_t + torch.logical_not(less_steps_reward)*steps_tmp

        less_steps_reward = 1.0*less_steps_reward


        #if target reached, but NOT mastered yet, reset episode to train better skill
        dones = torch.logical_and(reached, self.mastered_b[closest_ids] < self.mastering_threshold)

        #update mastered counter, for any reached target
        k = 0.1
        for i in range(batch_size):
            if reached_any[i]:
                target_id = closest_ids[i]
                self.mastered_b[target_id] = (1.0 - k)*self.mastered_b[target_id] + k*1.0
        
        
        #mastered target decay  
        self.mastered_b*= 0.9999
        self.mastered_b[0] = 1.0 

        '''
        #print debug
        print("mastered          : ", self.mastered_b)
        print("rewards           : ", reached_reward, less_steps_reward)
        print("current_target_id : ", current_target_id)
        print("\n\n")
        '''

        return reached_reward.detach().numpy(), less_steps_reward.detach().numpy(), dones.detach().numpy()

    def mastered(self):

        result = (self.mastered_b[0:self.current_idx]).mean()
        
        return result.numpy()


    def _add_new_state(self, state, steps, score_sum):
        if self.current_idx < self.states_b.shape[0]:

            self.states_b[self.current_idx]     = state.clone()
            self.steps_b[self.current_idx]      = steps
            self.score_b[self.current_idx]      = score_sum
            self.mastered_b[self.current_idx]   = 0.0

            self.current_idx+= 1
            
    def _get_non_mastered_target(self):
        target_id = 0
        for i in range(self.current_idx):
            if self.mastered_b[i] < self.mastering_threshold:
                target_id = i
                break

        return target_id
 
