import numpy
import cv2
import time

class StateBuffer: 

    def __init__(self, shape, size = 64, threshold = 0.01, alpha = 0.01):
        self.mean       = numpy.zeros((size, ) + shape)
        self.distance   = numpy.zeros((size, ))
        self.count      = numpy.zeros((size, ), dtype=int)

        self.threshold      = threshold
        self.alpha          = alpha
        self.max_idx        = size
        self.current_idx    = 0

        self.loops = 0 

 
    def add(self, x):
        distance, idx = self._distance(x)

        if self.current_idx < self.max_idx and distance > self.threshold:
            self.mean[self.current_idx] = x.copy()
            self.current_idx+= 1
        else:
            self.mean[idx]  = (1.0 - self.alpha)*self.mean[idx] + self.alpha*x
        
        self.distance[idx]  = (1.0 - self.alpha)*self.distance[idx] + self.alpha*distance
        result              = distance - self.distance[idx]

        self.count[idx]+= 1

        self.visiting_probs = self.count/(numpy.sum(self.count) + 0.0000001)

        weight = 1.0 - self.visiting_probs[idx]

        result = (distance - self.distance[idx])*weight

        self.render() 
        #print(">>> ", idx, result)
        return result   

    def _distance(self, x):
        distances   = ((self.mean - x)**2)

        shape       = (distances.shape[0], numpy.prod(distances.shape[1:]))
        distances   = distances.reshape(shape)
        distances   = distances.mean(axis=1)

        return numpy.min(distances), numpy.argmin(distances)


    def render(self):

        percent     = 100.0*self.count/(numpy.sum(self.count) + 0.0000001)
        percent     = numpy.round(percent, 1)

        space       = 2

        count       = self.mean.shape[0]
        height      = self.mean.shape[1]
        width       = self.mean.shape[2]

        grid_width = int(count**0.5)
        while count%grid_width != 0:
            grid_width+= 1
        grid_height = count//grid_width

        im_height   = grid_height*(height + space)
        im_width    = grid_width*(width + space)

        image = numpy.zeros((im_height, im_width))

        idx = 0

        for ry in range(grid_height):
            for rx in range(grid_width):

                ys = ry*(height + space) + space//2
                xs = rx*(width + space) + space//2

                ye = ys + height
                xe = xs + width

                image[ys:ye, xs:xe] = self.mean[idx]

                font = cv2.FONT_HERSHEY_SIMPLEX
                text = str(percent[idx]) + "%"
                cv2.putText(image, text, (xs , ys + height//4), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                idx+= 1
                         
        cv2.imshow("StateBuffer", image) 
        cv2.waitKey(1)