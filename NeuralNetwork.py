import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.models import Model
import keras_tuner as kt 


def linear_seq_nnmodel(X_train,X_test,y_train,y_test):
 # Create a Keras Sequential model and add more than one Dense hidden layer
    nn_model = tf.keras.models.Sequential()

    # Set the input nodes to the number of features
    input_nodes = len(X_train.columns)

    nn_model.add(tf.keras.layers.Dense(units=128, activation="relu", input_dim=input_nodes))

    nn_model.add(tf.keras.layers.Dense(units=128, activation="relu"))
    nn_model.add(tf.keras.layers.Dense(units=128, activation="relu"))
    nn_model.add(tf.keras.layers.Dense(units=128, activation="relu"))


    nn_model.add(tf.keras.layers.Dense(units=1, activation="linear"))

    # Check the structure of the Sequential model
    nn_model.summary()
    nn_model.compile(loss="mse", optimizer="adam", metrics=[tf.keras.metrics.RootMeanSquaredError()])

    fit_model = nn_model.fit(X_train, y_train, epochs=500)

    model_loss, model_accuracy = nn_model.evaluate(X_test,y_test,verbose=2)
    print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

    return nn_model

def linear_funct_model(X_train,X_test,y_train,y_test):
    # defining layers
    input_layer = Input(shape=(len(X_train.columns),))
    dense_layer_1 = Dense(units = 128, activation = "relu")(input_layer) 
    dense_layer_2 = Dense(units = 128, activation = "relu")(dense_layer_1)
    # dense_layer_3 = Dense(units = 64, activation = "relu")(dense_layer_2)

    #Y1 output
    y1_output = Dense(units = 1, activation = "linear", name = "y1_output")(dense_layer_2)

    #Y2 output
    # y2_output = Dense(units = 1, activation = "linear", name = "y2_output")(dense_layer_3)

    #Define the model with the input layer and a list of outputs
    model = Model(inputs = input_layer, outputs = y1_output)

    #specify the optimizer and compile with the loss function for both outputs
    

    model.compile(optimizer = 'adam',
                loss = {'y1_output':'mse'},
                metrics = {
                    'y1_output':tf.keras.metrics.RootMeanSquaredError()
                }
                )
    #training process
    history = model.fit(X_train, y_train, epochs = 500, batch_size = 10,
                    validation_data = (X_test, y_test), verbose = 0)
    return model

def hypertuning(X_train,X_test,y_train,y_test):
    def create_model(hp):
        nn_model = tf.keras.models.Sequential()

        # Allow kerastuner to decide which activation function to use in hidden layers
        activation = hp.Choice('activation',['relu','tanh'])

        # Allow kerastuner to decide number of neurons in first layer
        nn_model.add(tf.keras.layers.Dense(units=hp.Int('first_units',
            min_value=1,
            max_value=30,
            step=5), activation=activation, input_dim=len(X_train.columns)))

        # Allow kerastuner to decide number of hidden layers and neurons in hidden layers
        for i in range(hp.Int('num_layers', 1, 8)):
            nn_model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
                min_value=1,
                max_value=30,
                step=5),
                activation=activation))

        nn_model.add(tf.keras.layers.Dense(units=1, activation="linear"))

        # Compile the model
        nn_model.compile(loss="mse", optimizer="adam", metrics=[tf.keras.metrics.RootMeanSquaredError()])

        return nn_model

    tuner = kt.Hyperband(
        create_model,
        objective='val_loss',
        max_epochs=50,
        overwrite=True,
        hyperband_iterations=2)
        
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    tuner.search(X_train,y_train,epochs=50,validation_data=(X_test,y_test), callbacks=[stop_early])

    top_hyper = tuner.get_best_hyperparameters(1)
    
    for param in top_hyper:
        print(param.values)

    top_model = tuner.get_best_models(3)
    for model in top_model:
        model_loss, model_accuracy = model.evaluate(X_test,y_test,verbose=2)
        print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

def multi_class_seq_model(X_train,X_test,y_train,y_test,X):
    model=tf.keras.models.Sequential()
    input_nodes=len(X.columns)
    model.add(tf.keras.layers.Dense(units=50,activation='tanh',input_dim=input_nodes))

    model.add(tf.keras.layers.Dense(units=35,activation='tanh'))
    model.add(tf.keras.layers.Dense(units=50,activation='tanh'))
    model.add(tf.keras.layers.Dense(units=45,activation='tanh'))
    model.add(tf.keras.layers.Dense(units=45,activation='relu'))

    model.add(tf.keras.layers.Dense(units=10,activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')

    history=model.fit(X_train,y_train, epochs=500,batch_size=32,validation_data=(X_test,y_test))

    modelloss,modelaccuracy=model.evaluate(X_test,y_test,verbose=2)
    print(f'model loss: {modelloss},  model accuracy: {modelaccuracy}')
    return model

def hypertuning_multiclass():

    def create_model(hp):
        nn_model = tf.keras.models.Sequential()

        # Allow kerastuner to decide which activation function to use in hidden layers
        activation = hp.Choice('activation',['relu','tanh'])

        # Allow kerastuner to decide number of neurons in first layer
        nn_model.add(tf.keras.layers.Dense(units=hp.Int('first_units',
            min_value=20,
            max_value=60,
            step=5), activation=activation, input_dim=len(X.columns)))

        # Allow kerastuner to decide number of hidden layers and neurons in hidden layers
        for i in range(hp.Int('num_layers', 1, 5)):
            nn_model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
                min_value=20,
                max_value=60,
                step=5),
                activation=activation))

        nn_model.add(tf.keras.layers.Dense(units=10, activation="softmax"))

        # Compile the model
        nn_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

        return nn_model


    tuner = kt.Hyperband(
            create_model,
            objective='loss',
            max_epochs=20,
            overwrite=True,
            hyperband_iterations=2)
            
    stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)

    tuner.search(X_train,y_train,epochs=20,batch_size=1000,validation_data=(X_test,y_test), callbacks=[stop_early])

    top_hyper = tuner.get_best_hyperparameters(3)

    for param in top_hyper:
        print(param.values)

    top_model = tuner.get_best_models(3)
    for model in top_model:
        model_loss, model_accuracy = model.evaluate(X_test,y_test,verbose=2)
        print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")