import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input, Dropout
from keras.models import Model
import matplotlib.pyplot as plt
import functions as fct

tf.random.set_seed(7)

trainingData, _, _ = fct.load_and_format_data('./data/S.02_training.csv')
trainingData = MinMaxScaler().fit_transform(trainingData)

Y = np.repeat(1, len(trainingData))
X_train, X_valid, _, _ = train_test_split(trainingData, Y, test_size=0.2, random_state=7)
input_size = 62

#%% No of layers

parameters= dict()

hidden_size_1 = 40
core_size = 25
    
input_df = Input(shape=(input_size,))

encoded_1 = Dense(hidden_size_1, activation = 'relu')(input_df)
encoded_1 = Dropout(rate = 0.1)(encoded_1)

core = Dense(core_size, activation = 'relu')(encoded_1)
core = Dropout(rate = 0.1)(core)

decoded_1 =  Dense(hidden_size_1, activation = 'relu')(core)
decoded_1 = Dropout(rate = 0.1)(decoded_1)

output_df = Dense(input_size, activation='relu')(decoded_1)


autoencoder = Model(input_df, output_df)

autoencoder.compile(optimizer='adam', loss='mse')
history = autoencoder.fit(X_train, X_train, epochs=50, verbose=0,
                          validation_data=(X_valid, X_valid))

parameters['2 layers val_loss'] = history.history['val_loss']
parameters['2 layers loss'] = history.history['loss']




core_size = 40
    
input_df = Input(shape=(input_size,))

core = Dense(core_size, activation = 'relu')(input_df)
core = Dropout(rate = 0.1)(core)

output_df = Dense(input_size, activation='relu')(core)


autoencoder = Model(input_df, output_df)

autoencoder.compile(optimizer='adam', loss='mse')
history = autoencoder.fit(X_train, X_train, epochs=50, verbose=0,
                          validation_data=(X_valid, X_valid))

parameters['1 layer val_loss'] = history.history['val_loss']
parameters['1 layer loss'] = history.history['loss']


plt.plot(range(1, 51), parameters['1 layer val_loss'], label='1 layer val_loss', color='k')
plt.plot(range(1, 51), parameters['1 layer loss'], label='1 layer loss', color='k')

plt.plot(range(1, 51), parameters['2 layers loss'], label='2 layers loss', color='r')
plt.plot(range(1, 51), parameters['2 layers val_loss'], label='2 layers val_loss', color='r')

plt.legend()
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title('Loss comparison for 1 or 2 hidden layers')
plt.show()

#%% Grid Search
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

tf.random.set_seed(7)

def autoencoderModel(cs=20, DO=0.1, activeFct = 'relu'):

    core_size = cs
        
    input_df = Input(shape=(62,))
    
    core = Dense(core_size, activation = activeFct)(input_df)
    core = Dropout(rate = DO)(core)
    
    output_df = Dense(62, activation= activeFct)(core)
    
    autoencoder = Model(input_df, output_df)
    
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mse'])
    
    return autoencoder


model = KerasRegressor(build_fn=autoencoderModel, verbose=0)

parameters = {'cs': [20, 25, 30, 35, 40], 
              'DO': [0.1, 0.2, 0.3, 0.4, 0.5],
              'activeFct' : ['relu', 'sigmoid', 'tanh'],
              'epochs' : [20, 40, 60, 80]
              }
grid = GridSearchCV(estimator=model, param_grid=parameters, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train, X_train, validation_data=(X_valid, X_valid))


