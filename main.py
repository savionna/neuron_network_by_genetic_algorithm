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
            y.append([int(label)])

    X_train = np.array(X)
    y_train = np.array(y)

    X_test = []
    Y_test = []
    for line in test_lines:
        pattern2, label2 = line.strip().split()
        X_test.append([int(bit) for bit in pattern2])
        Y_test.append([int(label2)])

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)


    return X_train, y_train, X_test, Y_test


def main():
    # Read data
    file = 'nn0.txt'
    X_train, y_train, X_test, test_real_lables = read_data(file)

    #print(str(test_real_lables) + "\n")
    #print(y_train[0:20])

    # Define network parameters
    input_size = len(X_train[0])
    hidden_size = 4  # Adjust the number of hidden units as desired
    output_size = 1

    # Create and train the neural network
    neural_net = SimpleNeuralNetwork.NeuralNetwork(input_size, hidden_size, output_size)
    epochs = 3  # Adjust the number of training epochs as desired
    learning_rate = 0.1  # Adjust the learning rate as desired
    for _ in range(epochs):
        neural_net.forward(X_train)
        neural_net.backward(X_train, y_train, learning_rate)

    # Classify test patterns and save the labels
    test_labels = []
    for i in range(len(X_test)):
        y_pred = neural_net.forward(X_test[i])
        test_labels.append(int(np.round(y_pred)))
    
    #verify the results:
    for label in range(len(test_real_lables)):
        if test_labels[label] != test_real_lables[label]:
            print(f'Test pattern {label}: real label is: {test_real_lables[label]}  the result was: {test_labels[label]}')

if __name__ == '__main__':
    main()
