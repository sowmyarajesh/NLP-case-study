import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import  Dropout, Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

def trainNNetwork(X,y, epochs=1):
    Xtrain,Xtest, ytrain, ytest = train_test_split(X,y, test_size=0.2, random_state=48)
    
    xs = np.array(Xtrain)
    ys = np.array(ytrain)
    
    testX = np.array(Xtest)
    testy=np.array(ytest)

    model = Sequential([
        tf.keras.layers.InputLayer(input_shape=xs.shape[1]),
        Dense(7000,activation='tanh'),
        Dropout(0.5),
        Dense(300,activation='tanh'),
        Dense(ys.shape[1], activation="sigmoid")
    ])
    callback = tf.keras.callbacks.EarlyStopping(monitor='auc', patience=3)
    loss = tf.keras.losses.CategoricalCrossentropy()
    optim = tf.keras.optimizers.Adam()
    metrics = [tf.keras.metrics.AUC()]

    model.compile(loss=loss, optimizer=optim, metrics=metrics)
    history = model.fit(xs, ys, batch_size=4, epochs=epochs,validation_data=(testX, testy),callbacks=[callback],verbose=1)
    return {"model":model, "history":history}