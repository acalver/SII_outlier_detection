from keras.layers import LSTM
import MLfunctions as ML
import dataProcessing as dP
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

trainingData, trainingIndex, cols = dP.load_and_format_data('./data/S.02_training.csv')
testData, _, _ = dP.load_and_format_data('./data/S.02_test.csv')

scalar = MinMaxScaler()
scaledData = scalar.fit_transform(trainingData)
scaledData = pd.DataFrame(scaledData, index=trainingIndex, columns=cols)

#dummy Y variable for train test split function
Y = np.repeat(1, len(scaledData))

#create test and validation sets for autoencoder
AE_train, AE_valid, _, _ = train_test_split(scaledData, Y, test_size=0.2, random_state=7)
#%% Autoencoder
import tensorflow as tf

tf.random.set_seed(7)

autoencoder, encoder = ML.autoencoder_model(cs=40, DO=0.1, activeFct='tanh')
#no benefit in performance past 40 epochs, avoid overfitting
history = autoencoder.fit(AE_train, AE_train, epochs=40, 
                          validation_data=(AE_valid, AE_valid))


encodedTraining = encoder.predict(scaledData)
encodedTraining = pd.DataFrame(encodedTraining, index=trainingIndex)

trainingJoined = scaledData.join(encodedTraining)

#%% Feature selection

dropCols = dP.correlated_columns(trainingJoined, cols)

#%% Grid search
from keras.models import Sequential
from keras.layers import Dense


def gridSearch_model(features, activeFct, recAct, nodes,do):

    lstm = Sequential()
    lstm.add(LSTM(n, input_shape=(5, features), activation = activeFct,
              recurrent_activation = recAct, dropout=do))
    lstm.add(Dense(features))
    lstm.compile(loss='mean_squared_error', optimizer='adam')
    return lstm


d=dict()
for er in ['']:#redacted due to confidentiality

    singleFirm, singleScalar = dP.single_firm(er, data=trainingData)
    
    trainX, trainY, testX, testY, singleEncoded = ML.LSTM_train_test_data(firmData=singleFirm, EntRef=er, dc=dropCols, 
                                                                   encode=encoder, window=5,
                                                                   testData=testData)
    f = np.shape(trainY)[1]
    best_loss=1
    for n in [90, 100, 110, 120]:
        
        for ra in ['relu', 'sigmoid', 'tanh', None]:
            
            for af in ['relu', 'sigmoid', 'tanh', None]:
                
                for do in [0, 0.1, 0.2]:
                    
                    tf.random.set_seed(7)
                    grid = gridSearch_model(f, af, ra, n, do)
                    grid_result = grid.fit(trainX, trainY, epochs=200, verbose=0)
                    loss = grid_result.history['loss']
                
                    if min(loss) < best_loss:
                        best_loss=min(loss)
                        d[er+str(ra or '_')+str(n)+str(af or '_')+str(do)] = loss


#%% refined grid search
import matplotlib.pyplot as plt

def refined_gridSearch_model(features, nodes, do=0, activeFct='tanh', recAct='tanh'):

    lstm = Sequential()
    lstm.add(LSTM(n, input_shape=(5, features), activation = activeFct,
              recurrent_activation = recAct, dropout=do))
    lstm.add(Dense(features))
    lstm.compile(loss='mean_squared_error', optimizer='adam')
    return lstm

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,9), sharex=True)
fig.suptitle("Comparison of number of nodes on 4 sample firms", fontsize=20)
firms= ['']#redacted due to confidentiality
for i, ax in enumerate(axes.flatten()):
    
    er=firms[i]
    
    d=dict()
    singleFirm, singleScalar = dP.single_firm(er, data=trainingData)
    
    trainX, trainY, testX, testY, singleEncoded = ML.LSTM_train_test_data(firmData=singleFirm, EntRef=er, dc=dropCols, 
                                                                   encode=encoder, window=5,
                                                                   testData=testData)
    f = np.shape(trainY)[1]
    for n in [90, 100, 110, 120]:
            
            tf.random.set_seed(7)
            rgrid = refined_gridSearch_model(f, n)
            grid_result = rgrid.fit(trainX, trainY, epochs=200, verbose=0)
            loss = grid_result.history['loss']

            d[er+'_'+str(n)] = loss
    
    
        
    for k in d.keys():
        ax.plot(d[k])

plt.show()
