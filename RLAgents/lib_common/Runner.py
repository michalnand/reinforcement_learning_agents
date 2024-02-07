import multiprocessing
import time
import torch

class Runner:

    # devices = [cuda:0, cuda:1]
    def __init__(self, experiments, devices, delay_s = 10.0):
        multiprocessing.set_start_method('spawn')
        #multiprocessing.set_start_method('fork')

        process = []
        for i in range(len(experiments)):

            device = devices[i%len(devices)]

            #with multiprocessing.Pool(1) as p:
            #    p.map(module.run, [])

            params = [experiments[i], i, device]
            p = multiprocessing.Process(target=self._run, args=params)
            process.append(p)

        for p in process:
            p.start()
            time.sleep(delay_s)

        for p in process:
            p.join()

        print("Runner : experiments done")

    def _run(self, experiment, i, device):
        print("Runner : starting ", i)

        try:
            if "cuda" in device:
                torch.cuda.set_device(device)
                print("Runner : device   ", device)
        except:
            pass

        module = __import__(experiment)
        module.run()
        
