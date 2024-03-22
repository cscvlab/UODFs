import time

class Timer():
    def __init__(self):
        pass
    def start(self):
        self.start = time.time()

    def stop(self):
        self.end = time.time()

    def print(self):
        seconds = self.end - self.start
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        print("time: %d:%02d:%02d" % (h, m, s))