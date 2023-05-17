import sys


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Logger(object):
    def __init__(self, path_logfile):
        self.terminal = sys.stdout
        self.log = open(path_logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

