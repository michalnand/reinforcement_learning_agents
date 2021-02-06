import numpy
from scipy import stats

class RLStatsCompute:
    def __init__(self, files_list, confidence = 0.95):
        self.data = self.load_files(files_list)
        self.mean, self.std, self.lower, self.upper = self.compute_stats(self.data, confidence)

    def load_files(self, files_list):
        data      = []
      
        for f in files_list:
            print("loading ", f)
            data_ = numpy.loadtxt(f, unpack = True)
            data.append(data_)

        data      = numpy.array(data)
      
        return data
        

    def compute_stats(self, data, confidence = 0.95):
        n       = data.shape[2]

        mean    = numpy.mean(data, axis = 0)
        std     = numpy.std(data, axis = 0)
        se      = stats.sem(data, axis=0)
        h       = se * stats.t.ppf((1 + confidence) / 2., n-1)

        lower = mean - h
        upper = mean + h

        return mean, std, lower, upper

