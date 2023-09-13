
class DecayConst():
    def __init__(self, training_value = 0.1, testing_value = 0.1):
        self.training_value = training_value
        self.testing_value  = testing_value

    def process(self):
        return self.training_value

    def get(self):
        return self.training_value

    def get_start(self):
        return self.training_value

    def get_end(self):
        return self.training_value

    def get_testing(self):
        return self.testing_value


class DecayStep():
    def __init__(self, iterations = 1000000, start_value = 1.0, end_value = 0.1, testing_value = 0.02):
        self.iterations     = iterations
        self.start_value    = start_value
        self.end_value      = end_value
        self.testing_value  = testing_value

        self.steps = 0 

        self.epsilon = self.start_value

    def process(self):
        if self.steps > self.iterations:
            self.epsilon = self.end_value
        else:
            self.epsilon = self.start_value
        
        self.steps+= 1

        return self.epsilon

    def get(self):
        return self.epsilon

    def get_start(self):
        return self.start_value

    def get_end(self):
        return self.end_value

    def get_testing(self):
        return self.testing_value


class DecayLinear():
    def __init__(self, iterations = 1000000, start_value = 1.0, end_value = 0.1, testing_value = 0.02):
        self.start_value = start_value
        self.end_value = end_value
        self.testing_value = testing_value

        self.decay = (self.start_value - self.end_value)*1.0/iterations   

        self.epsilon = self.start_value

    def process(self):
        if self.epsilon > self.end_value:
            self.epsilon-= self.decay
        return self.epsilon

    def get(self):
        return self.epsilon

    def get_start(self):
        return self.start_value

    def get_end(self):
        return self.end_value

    def get_testing(self):
        return self.testing_value


class DecayLinearDelayed():
    def __init__(self, start_iterations = 200000, iterations = 1000000, start_value = 1.0, end_value = 0.1, testing_value = 0.02):
        self.start_iterations = start_iterations
        self.start_value = start_value
        self.end_value = end_value
        self.testing_value = testing_value

        self.decay = (self.start_value - self.end_value)*1.0/iterations   

        self.epsilon = self.start_value

        self.start = False
        self.iterations = 0

    def process(self):
        if self.iterations > self.start_iterations:
            if self.epsilon > self.end_value:
                self.epsilon-= self.decay
        
        self.iterations+= 1
        
        return self.epsilon

    def get(self):
        return self.epsilon

    def get_start(self):
        return self.start_value

    def get_end(self):
        return self.end_value

    def get_testing(self):
        return self.testing_value


class DecayExponential():
    def __init__(self, q = 0.999999, start_value = 1.0, end_value = 0.1, testing_value = 0.02):
        self.q = q
        self.start_value = start_value
        self.end_value = end_value
        self.testing_value = testing_value

        self.epsilon = self.start_value

    def process(self):
        if self.epsilon > self.end_value:
            self.epsilon = self.epsilon*self.q
        return self.epsilon

    def get(self):
        return self.epsilon

    def get_start(self):
        return self.start_value

    def get_end(self):
        return self.end_value

    def get_testing(self):
        return self.testing_value




