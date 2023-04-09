import model
from config import *
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score

mnist=fetch_openml("mnist_784",cache=True)
X,y=mnist["data"],mnist["target"]
X=X.to_numpy()
y=y.to_numpy().astype("int64")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale pixel values to range [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

import matplotlib.pyplot as plt
def train(model, X_train, y_train,X_test,y_test, num_epochs, batch_size, learning_rate):
    num_samples = X_train.shape[0]
    train_loss_history = []
    test_loss_history=[]
    train_accuracy_history=[]
    test_accuracy_history=[]
    for epoch in range(num_epochs):
        # Shuffle the data
        shuffle_indices = np.random.permutation(num_samples)
        shuffled_X = X_train[shuffle_indices]
        shuffled_y = y_train[shuffle_indices]

        # Split the data into batches
        num_batches = num_samples // batch_size
        for i in range(num_batches):
            # Get the current batch
            start = i * batch_size
            end = (i + 1) * batch_size
            batch_X = shuffled_X[start:end]
            batch_y = shuffled_y[start:end]

            # Forward pass
            caches, y_hat = model.forward(batch_X)


            # Backward pass
            model.backward(batch_X, batch_y, learning_rate,batch_size)

        # Compute loss on full dataset
        _, y_train_hat = model.forward(X_train)
        train_loss = model.compute_loss(y_train_hat, y_train)
        y_train_predict=model.predict(X_train)

        train_accuracy = accuracy_score(y_train,y_train_predict)
        train_loss_history.append(train_loss)
        train_accuracy_history.append(train_accuracy)

        _,y_test_hat=model.forward(X_test)
        test_loss = model.compute_loss(y_test_hat, y_test)
        y_test_predict = model.predict(X_test)

        test_accuracy = accuracy_score(y_test, y_test_predict)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        print(f"Epoch {epoch + 1}/{num_epochs}, train_loss = {train_loss:.5f}, test_loss={test_loss:.5f},train_accuracy = {train_accuracy:.5f},test_accurcy={test_accuracy:.5f}")


    plt.plot(train_loss_history,label="Train")
    plt.plot(test_loss_history, label="Test")

    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.plot(train_accuracy_history,label="Train")
    plt.plot(test_accuracy_history, label="Test")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    #return loss_history
def evaluate(model, X, y):
    _, y_pred = model.forward(X)
    accuracy = accuracy_score(y, np.argmax(y_pred, axis=1))
    print(f"Accuracy on test set: {accuracy:.5f}")
    return accuracy
model=model.TwoLayerNet(input_size=INPUT_SIZE,output_size=OUTPUT_SIZE,hidden_size=HIDDEN_SIZE,reg_lambda=LAMBDA)##初始化模型
train(model,X_train,y_train,X_test,y_test,NUM_EPOCHES,BATCH_SIZE,LEARNING_RATE)


import pickle

# 保存模型
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# 加载模型
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
