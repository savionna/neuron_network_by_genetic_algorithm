import pickle
import numpy as np
import Person


def loadData():
    # for reading also binary mode is important
    dbfile = open('wnet0.txt', 'rb')     
    neural_net = pickle.load(dbfile)
    dbfile.close()

    x = []
    with open("testnet0.txt", 'r') as file:
        for line in file:
            pattern = line.strip()
            x.append([int(bit) for bit in pattern])
            
    x_test = np.array(x)
    neural_net.test(x_test)


if __name__ == "__main__":
    loadData()