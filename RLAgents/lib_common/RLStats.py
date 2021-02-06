import numpy

class RLStats:
    def __init__(self):
        self.iterations_result      = []
        self.episodes_result        = []
        self.score_result           = []
        self.score_per_episode_result  = []

    def add(self, iterations, episodes, score_result, score_per_episode_result):
        self.iterations_result.append(iterations)
        self.episodes_result.append(episodes)
        self.score_result.append(score_result)
        self.score_per_episode_result.append(score_per_episode_result)


    def save(self, file_name = "result.log"):

        iterations_result       = numpy.transpose(self.iterations_result)
        episodes_result         = numpy.transpose(self.episodes_result)
        score_result            = numpy.transpose(self.score_result)
        score_per_episode_result   = numpy.transpose(self.score_per_episode_result)


        iterations_mean   = numpy.mean(iterations_result, axis = 1)
        episodes_mean     = numpy.mean(episodes_result, axis = 1)
        
        score_mean     = numpy.mean(score_result, axis = 1)
        score_std      = numpy.std(score_result, axis = 1)

        score_min, score_max = self._confidence_interval(score_mean, score_std, len(score_mean))

        score_per_episode_mean = numpy.mean(score_per_episode_result, axis = 1)
        score_per_episode_std = numpy.std(score_per_episode_result, axis = 1)

        score_per_episode_min, score_per_episode_max = self._confidence_interval(score_per_episode_mean, score_per_episode_std, len(score_per_episode_mean))


        f = open(file_name,"w") 
        for j in range(len(iterations_mean)):

            r = 3
            s = ""
            s+= str(int(iterations_mean[j])) + " "
            s+= str(int(episodes_mean[j])) + " "

            s+= str(round(score_mean[j], r)) + " "
            s+= str(round(score_std[j], r)) + " "
            s+= str(round(score_min[j], r)) + " "
            s+= str(round(score_max[j], r)) + " "

            s+= str(round(score_per_episode_mean[j], r)) + " "
            s+= str(round(score_per_episode_std[j], r)) + " "
            s+= str(round(score_per_episode_min[j], r)) + " "
            s+= str(round(score_per_episode_max[j], r)) + " "

            f.write(s+"\n")
            print(s) 

        f.close()
           
    def _confidence_interval(self, mean, std, count):
        z = 1.960
        v = z*std/(count**0.5)
        return mean - v, mean + v