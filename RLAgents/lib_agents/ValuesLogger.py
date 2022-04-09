
class ValuesLogger:

    def __init__(self):
        self.values = {}

    def add(self, name, value, smoothing = 0.02):
        if name int self.values
            self.values[name] = (1.0 - smoothing)*self.values[name] + smoothing*value
        else: 
            self.values[name] = value
            
    def get_str(self, decimals = 7): 
        result = "" 

        for index, (key, value) in enumerate(self.values.items()):
            s = str(round(value, decimals)) + " "
            result+= s

        return result 
