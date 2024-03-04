import torch


torch.manual_seed(0)

class Node:
    def __init__(self, name, nominalVoltage=None, busType = None, loadsConnected=None):
        self.name = name
        self.nominalVoltage = nominalVoltage
        self.busType = busType          # 0 = slack, 1 = PV, 2 = PQ
        self.loadsConnected = loadsConnected

class Edge:
    def __init__(self, name, lineType=None, lineLength = None, lineAge=None):
        self.name = name
        self.lineType = lineType        # 0 = Underground, 1 = Overhead
        self.lineLength = lineLength
        self.lineAge = lineAge
