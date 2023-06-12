import numpy as np
import SimpleNeuralNetwork

def read_data(file_path):
    X = []
    y = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        num_train = int(len(lines) * 3 / 4)  # 3/4 of rows for training
        train_lines = lines[:num_train]
        test_lines = lines[num_train:]

        for line in train_lines:
            pattern, label = line.strip().split()
            X.append([int(bit) for bit in pattern])
            y.append(int(label))

    X_train = np.array(X)
    y_train = np.array(y)

    X_test = []
    Y_test = []
    for line in test_lines:
        pattern2, label2 = line.strip().split()
        X_test.append([int(bit) for bit in pattern2])
        Y_test.append(int(label2))

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)


    return X_train, y_train, X_test, Y_test

def min_max_norm(X):
        min_value = np.min(X)
        max_value = np.max(X)

        range_value = max_value - min_value   
        normalized_X = (X - min_value) / range_value
            
        return normalized_X

def main():
    # Read data
    file = 'nn0.txt'
    X_train, y_train, X_test, test_real_lables = read_data(file)

    n_x_train = min_max_norm(X_train)
    n_x_test = min_max_norm(X_test)


    print(X_train.shape, y_train.shape)

    #print(str(test_real_lables) + "\n")
    # Define network parameters
    feature_number = len(X_train[1])
    hidden_size = 3
    output_size = 1
    num_of_epochs = 5

  
    # Create and train the neural network
    neural_net = SimpleNeuralNetwork.SimpleNeuralNetwork(feature_number, hidden_size,output_size)
    neural_net.train(X_train, y_train, num_of_epochs)
    accuracy = neural_net.test(X_test, test_real_lables)
    print(accuracy)

if __name__ == '__main__':
    main()
