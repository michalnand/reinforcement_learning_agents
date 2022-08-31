from importlib.metadata import files
import re
import numpy
from scipy import stats
import json

class RLStatsCompute:
    def __init__(self, files_list, confidence = 0.68, extended_names = []):

        self.data, self.extended = self.load_files(files_list)
        
        self.mean, self.std, self.lower, self.upper, self.hist = self.compute_stats(self.data, confidence)


        if len(extended_names) > 0:
            extended_stats =[]
            for ex_idx in range(len(extended_names)):
                extended = []
                for f_idx in range(len(self.extended)):
                    es = [d[extended_names[ex_idx]] for d in self.extended[f_idx]]
                    extended.append(es)

                extended_stats.append(extended)
                
            extended_stats = numpy.array(extended_stats)
            extended_stats = numpy.moveaxis(extended_stats, 0, 1)
            
            self.extended_mean, self.extended_std, self.extended_lower, self.extended_upper, self.extended_hist = self.compute_stats(extended_stats, confidence)
            



    def load_files(self, files_list):
        data      = []
        extended  = []
       
        for f in files_list:
            print("loading ", f)
            data_ = numpy.loadtxt(f, unpack = True, comments='{')
            data.append(data_)

            extended_f = []

            with open(f) as file:
                lines = file.readlines()

                for line in lines:
                    tmp = "{" + line.split('{')[1]
                    if len(tmp) > 0:
                        try:
                            tmp = tmp.replace("'", "\"")
                            tmp = json.loads(tmp)
                            extended_f.append(tmp)
                        except Exception:
                            pass
            
            extended.append(extended_f)

        data      = numpy.array(data)
        
        return data, extended
        


    def compute_stats(self, data, confidence):
        n       = data.shape[2]

        mean    = numpy.mean(data, axis = 0)
        std     = numpy.std(data, axis = 0)
        se      = stats.sem(data, axis=0)
        h       = se * stats.t.ppf((1 + confidence) / 2., n-1)

        lower = mean - h
        upper = mean + h

        hist = []

        for col in range(data.shape[1]):
            h, e = numpy.histogram(data[0][col], bins=64)

            e    = e[0:-1]
            h    = h/numpy.sum(h)

            hist.append([e, h])

        hist = numpy.array(hist)

        return mean, std, lower, upper, hist

