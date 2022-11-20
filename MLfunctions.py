import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM
import dataProcessing as dP

def autoencoder_model(cs=40, DO=0.1, activeFct='tanh'):
    """
    Parameters
    ----------
    cs : int
        size of bottle neck layer.
    DO : int
        level of dropout.
    activeFct : str
        activation function to be used.

    Returns
    -------
    autoencoder : keras autoencoder model.
    encoder : compressor of autoencoder

    """
        
    input_df = Input(shape=(62,))
    
    core = Dense(cs, activation = activeFct)(input_df)
    core = Dropout(rate = DO)(core)
    
    output_df = Dense(62, activation= activeFct)(core)
    
    autoencoder = Model(input_df, output_df)
    encoder = Model(input_df, core)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, encoder

def LSTM_model(window, features, nodes, activeFct='tanh', recAct='tanh'):
    
    lstm = Sequential()
    lstm.add(LSTM(nodes, input_shape=(window, features), activation = activeFct, recurrent_activation = recAct))
    lstm.add(Dense(features))
    lstm.compile(loss='mean_squared_error', optimizer='adam')
    
    return lstm


def LSTM_train_test_data(firmData, EntRef, dc, encode, window, testData):
    """
    Prepare data for LSTM model
    
    Parameters
    ----------
    firmData : DataFrame
        time series dataframe of a single firm.
    EntRef : str
        Entity Reference of the selected firm.
    dc : array
        columns to be dropped.
    encode : keras autoencoder
        compressor of autoencoder.
    window : int
        size of time periods to split training data into.
    testData : dataframe
        dataframe of 2021Q1 reporting periods.

    Returns
    -------
    trainX : 3D array
        3D array of LSTM training data (does not include the last window).
    trainY : 2D array
        training outputs for LSTM.
    testX : 3D array
        latest training period window for testing.
    testY : dataframe
        2021Q1 data filtered on EntRef firm.
    singleEncoded : dataframe
        dataframe of autoencodered features for firm.

    """
      
    index = firmData.index
    
    #generate autoencoder features
    singleEncoded = encode.predict(firmData)
    singleEncoded = pd.DataFrame(singleEncoded, index=index)
    singleEncoded = singleEncoded.drop(dc, axis='columns')
    
    singleExtended = firmData.join(singleEncoded)
     
    #create 3D training array for LSTM, corresponding Y values
    trainX, trainY = dP.time_series_windows(singleExtended, window=window)
    features = trainX.shape[2]
    
    #create dummy row so time_series_windows function can select last 
    #training window as testX
    dummyRows = np.vstack((singleExtended, np.repeat(0, features)))
    rows = dummyRows.shape[0]
    testX, _ = dP.time_series_windows(dummyRows[rows-(window+1):rows], window=window)
    
    testY = testData.filter(like=str(EntRef), axis='index')
    
    return trainX, trainY, testX, testY, singleEncoded


def seperate_predictions(lstmModel, testData, scalar):
    """
    Make a prediction on the test data, then
    split the data into SII features and autoencoder features

    Parameters
    ----------
    lstmModel : keras LSTM model.
    testData : array
        3D array for LSTM to make predicition on.
    scalar : sklearn MinMaxScalar
        the firm-specific scalar for inverse transformation.

    Returns
    -------
    testPredict_SII : array
        array with 62 columns.
    testPredict_AE : array
        array with remaining columns from testData input.

    """
    testPredict = lstmModel.predict(testData)
    
    testPredict_SII = testPredict[:,:62]
    testPredict_SII = scalar.inverse_transform(testPredict_SII)
    testPredict_AE = testPredict[:,62:]
    
    return testPredict_SII, testPredict_AE
    
def trainingMSE(trainX, trainY, scalar, model):
    
    #latest training period
    latest = np.reshape(trainX[-1], (1,5,trainX[-1].shape[1]))

    #SII MSE
    predSII=model.predict(latest)
    predSII=predSII[:,:62]
    predSII=scalar.inverse_transform(predSII)
    trueSII=scalar.inverse_transform([trainY[-1,:62]])
    SII = mean_squared_error(predSII, trueSII, squared=False)
    
    
    #AE MSE
    predAE=model.predict(latest)
    predAE=predAE[:,62:]
    trueAE=[trainY[-1,62:]]
    AE = mean_squared_error(predAE, trueAE, squared=False)
    
    return SII, AE


def SII_outliers(ER, predictedDF, trainingF, testF):
    
    firmData = trainingF.filter(like=str(ER), axis='index')
    latestPeriod = firmData.iloc[-1]    
    
    comparison = pd.DataFrame(latestPeriod)
    
    #calculate importance relative to total assets
    assets = comparison.loc['S.02.01R0500C0010'].values[0]
    comparison['Total Assets'] = assets
    comparison['Importance'] = comparison[str(ER)] * 100/ assets
    comparison = comparison.drop(['Total Assets'], axis='columns')
    
    #calculate st. dev. of each feature
    SD = np.std(firmData)
    comparison['SD'] = SD 
    
    #calculate st. dev. range from last training observation
    comparison['Upper bound'] = latestPeriod + SD
    comparison['Lower bound'] = latestPeriod - SD
    comparison = comparison.fillna(0)
    comparison = comparison.round(2)
    
    comparison = comparison.join(predictedDF.T)
    
    #is predicted value outside range of st dev?
    comparison['Difference'] = (comparison['Predicted'] > comparison['Upper bound']) | \
                            (comparison['Predicted'] < comparison['Lower bound'])
    
    comparison['Anomaly test'] = (comparison['Importance'] > 5) & comparison['Difference'] & (comparison['SD'] !=0)
    
    TestValue = testF.filter(like=str(ER), axis='index').T
    mse = mean_squared_error(TestValue, predictedDF.T, squared=False)
    
    return comparison, mse 

def AE_outliers(testData, er, predictedAE, scalar, encode, encodedTraining, dc):
    """
    Run tests on predicted generated data, calculate outliers, and calculate
    mean square error of results

    Parameters
    ----------
    testData : datafrane
        DESCRIPTION.
    er : str
        entiy reference of selected firm.
    predictedAE : array
        predictions of generated features for test period.
    scalar : MinMaxScalar()
        scalar of selected firm.
    encode : keras autoencoder
        compressor of keras autoencoder.
    encodedTraining : datafrane
        trainging data of autoencoder features.
    dc : array
        columns to be dropped.

    Returns
    -------
    outlierDF : dataframe
        lists outliers detected for this firm's autoencoded generated features.
    mse : float
        mean square error of predictions.

    """
    #format true test period data
    testPeriod = testData.filter(like=str(er), axis='index')
    testPeriod = scalar.transform(testPeriod)
    testPeriod = encode.predict(testPeriod)
    testPeriod = pd.DataFrame(testPeriod)
    testPeriod = testPeriod.drop(dc, axis='columns')
    
    latestPeriod = encodedTraining.iloc[-1]       
    SD = np.std(encodedTraining)
    
    outlierDF = np.vstack((latestPeriod, predictedAE))
    outlierDF = pd.DataFrame(outlierDF.T, columns = ['Last training period', 'Predicted'], 
                             index=encodedTraining.columns)
    
    outlierDF['SD'] = SD
    outlierDF['Upper bound'] = latestPeriod + outlierDF['SD']
    outlierDF['Lower bound'] = latestPeriod - outlierDF['SD']
    
    outlierDF['Anomaly test'] = (outlierDF['Predicted'] > outlierDF['Upper bound']) | \
                        (outlierDF['Predicted'] < outlierDF['Lower bound'])
                        
    mse = mean_squared_error(testPeriod.T, predictedAE[0], squared=False)

    return outlierDF, mse