#%% Load and process data
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import MLfunctions as ML
from sklearn.preprocessing import MinMaxScaler
import dataProcessing as dP

trainingData, trainingIndex, cols = dP.load_and_format_data('./data/S.02_training.csv')
testData, _, _ = dP.load_and_format_data('./data/S.02_test.csv')

scaler = MinMaxScaler()
scaledData = scaler.fit_transform(trainingData)
scaledData = pd.DataFrame(scaledData, index=trainingIndex, columns=cols)

#dummy Y variable for train test split function
Y = np.repeat(1, len(scaledData))

#create test and validation sets for autoencoder
AE_train, AE_valid, _, _ = train_test_split(scaledData, Y, test_size=0.2, random_state=7)
#%% Autoencoder
import tensorflow as tf


tf.random.set_seed(7)

autoencoder, encoder = ML.autoencoder_model()
#no benefit in performance past 50 epochs, avoid overfitting
history = autoencoder.fit(AE_train, AE_train, epochs=50, 
                          validation_data=(AE_valid, AE_valid))


#save and load autoencoder for repeated runs
'''
from tensorflow.keras.models import Model
from tensorflow import keras
autoencoder.save('data/autoencoder.h5')
autoencoder = keras.models.load_model('data/autoencoder.h5')
encoder = Model(autoencoder.input, autoencoder.layers[-2].output)
'''
#%% Feature selection

encodedTraining = encoder.predict(scaledData)
encodedTraining = pd.DataFrame(encodedTraining, index=trainingIndex)

trainingJoined = scaledData.join(encodedTraining)

#find generated columns showing over 0.7 correlation
dropCols = dP.correlated_columns(trainingJoined, cols)

#%% LSTM 

#data with summed totals on S.02 included
trainingFull, _, _ = dP.load_and_format_data('./data/S.02_training.csv', removeSums=False)
testFull, _,_ = dP.load_and_format_data('./data/S.02_test.csv', removeSums=False)

#only run model for firms that reported Q1, have YE in December and have over 5 time periods
reportedQ1 = dP.firm_population(testData, trainingData)

#window set at 5 as 5 periods in a year
window=5
#outlier results
SII_outliers=[]
AE_outliers=[]
#training MSE results
SII_trainMSE=[]
AE_trainMSE=[]
#test MSE results
SII_testMSE=[]
AE_testMSE=[]
for er in reportedQ1:
        
    tf.random.set_seed(7)
    
    singleFirm, singleScaler = dP.single_firm(er, data=trainingData)
    
    #training and test data for LSTM model
    trainX, trainY, testX, testY, singleEncoded = ML.LSTM_train_test_data(firmData=singleFirm, EntRef=er, dc=dropCols, 
                                                                           encode=encoder, window=window,
                                                                           testData=testData)
    features = np.shape(trainY)[1]
    
    lstm = ML.LSTM_model(window, features, nodes=100)
    lstm.fit(trainX, trainY, epochs=150, verbose=0)
    
    #split predictions between SII and AE for assessment
    testPredict_SII, testPredict_AE = ML.seperate_predictions(lstm, testX, singleScaler)
            
    SII_outliersDF = pd.DataFrame(testPredict_SII, columns=cols, index=pd.MultiIndex.from_arrays([['Predicted'],['']]))
    SII_outliersDF = dP.calculated_SII_fields(SII_outliersDF)        
    SII_outliersDF, SIITestMse = ML.SII_outliers(er, predictedDF=SII_outliersDF, trainingF=trainingFull,
                                testF=testFull)
    
    AE_outliersDF, AETestMmse = ML.AE_outliers(testData, er=er, predictedAE=testPredict_AE, scalar=singleScaler,
                            encode=encoder, encodedTraining=singleEncoded, dc=dropCols)
    

    print('Firm: %s' % er)
    
    SIITrainMse, AETrainMSE = ML.trainingMSE(trainX, trainY, singleScaler, lstm)
    
    SII_trainMSE.append(SIITrainMse)
    AE_trainMSE.append(AETrainMSE)
    
    SII_outliers.append(len(SII_outliersDF[SII_outliersDF['Anomaly test']]))
    AE_outliers.append(len(AE_outliersDF[AE_outliersDF['Anomaly test']]))
    
    SII_testMSE.append(SIITestMse)
    AE_testMSE.append(AETestMmse)
 
#%% Plot results
import seaborn as sns
import matplotlib.pyplot as plt

flaggedFirms=pd.read_csv('./data/Plausibility Response Analysis.csv', usecols=['FRN'])['FRN'].values
flaggedFirms=np.unique(flaggedFirms)

#outlier chart
outliers=pd.DataFrame({'SII outliers' :SII_outliers, 'AE outliers':AE_outliers, 
                            'outlier':['Outlier firm' if x in flaggedFirms else 'Non-outlier firm' for x in reportedQ1]})
ax=sns.lmplot(x='AE outliers', y='SII outliers', data=outliers, x_jitter=0.3, y_jitter=0.3, hue='outlier',
                ci = None)
plt.title('Relationship between detected SII and autoencoder outliers\n(whole population)')
plt.show()

#MSE training chart
MSEtrain=pd.DataFrame({'SII mse (log)' :np.log(SII_trainMSE), 'AE mse':AE_trainMSE, 
                       'outlier':['Outlier firm' if x in flaggedFirms else 'Non-outlier firm' for x in reportedQ1]})
ax=sns.scatterplot(x='AE mse', y='SII mse (log)', data=MSEtrain, x_jitter=0.3, y_jitter=0.3, hue='outlier')
ax.set_title('Relationship between training MSE of SII and autoencoder features\n(whole population)')
plt.show()

#MSE test chart
MSEtest=pd.DataFrame({'SII mse (log)' :np.log(SII_testMSE), 'AE mse':AE_testMSE, 
                       'outlier':['Outlier firm' if x in flaggedFirms else 'Non-outlier firm' for x in reportedQ1]})
ax=sns.scatterplot(x='AE mse', y='SII mse (log)', data=MSEtest, x_jitter=0.3, y_jitter=0.3, hue='outlier')
ax.set_title('Relationship between test MSE of SII and autoencoder features\n(whole population)')
plt.show()

