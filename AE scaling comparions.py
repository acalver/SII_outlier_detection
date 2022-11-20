import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import functions as fct
from sklearn.preprocessing import MinMaxScaler


trainingData, trainingIndex, trainingCols = fct.load_and_format_data('./data/S.02_training.csv')

#normal scaler
scalar = MinMaxScaler()
normal = scalar.fit_transform(trainingData)



EntRef = pd.read_csv('./data/S.02_training.csv', usecols=['Entity Reference'],
                     dtype={'Entity Reference':'str'})
EntRef = np.unique(EntRef.values)

#scale firms individually
byFirm = pd.DataFrame()

#scale firms individually with initial base of 0
byFirm_withbase = pd.DataFrame()

for ER in EntRef:
    
    oneFirm = trainingData.filter(like=str(ER), axis='index')
    
    cols = oneFirm.columns
    index = oneFirm.index
    
    firmScalar = MinMaxScaler()
    firm = firmScalar.fit_transform(oneFirm)
    firm = pd.DataFrame(firm, columns=cols, index=index)
    byFirm = byFirm.append(firm)
    
    #add 0 row as double jump will recognised as bigger than double
    base0 = pd.DataFrame([np.repeat(0, len(cols))], columns=cols, index=[0]).append(oneFirm)   
    base0Scalar = MinMaxScaler()
    base0 = base0Scalar.fit_transform(base0)
    base0 = np.delete(base0,0, axis=0)
    base0 = pd.DataFrame(base0, columns=cols, index=index)
    byFirm_withbase = byFirm_withbase.append(base0)


#%%
from keras.layers import Dense, Input
from keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt

data_dict = {'normal' : normal, 'byFirm' : byFirm, 'byFirm_withbase' : byFirm_withbase}
parameters = dict()

eps = 25
input_size = 62
core_size = 40
for i in data_dict.keys():
    
    #dummy Y variable for train test split function
    Y = np.repeat(1, len(data_dict[i]))
    
    #create test and validation sets for autoencoder
    X_train, X_valid, _, _ = train_test_split(data_dict[i], Y, test_size=0.2, random_state=7)    
    
    
    tf.random.set_seed(7)
      
    input_df = Input(shape=(input_size,))
    
    core = Dense(core_size, activation='relu')(input_df)
    
    output_df = Dense(input_size, activation='relu')(core)
    
    
    autoencoder = Model(input_df, output_df)
        
    autoencoder.compile(optimizer='adam', loss='mse')
    history = autoencoder.fit(X_train, X_train, epochs=eps,
                              validation_data=(X_valid, X_valid))
    
    parameters[i + 'val_loss'] = history.history['val_loss']
    parameters[i + 'loss'] = history.history['loss']
    
    
plt.plot(range(1, eps+1), parameters['normalval_loss'], label='normal val_loss', color='orange')
plt.plot(range(1, eps+1), parameters['normalloss'], label='normal loss', color='orange')

plt.plot(range(1, eps+1), parameters['byFirmval_loss'], label='byFirm val_loss', color='k')
plt.plot(range(1, eps+1), parameters['byFirmloss'], label='byFirm loss', color='k')

plt.plot(range(1, eps+1), parameters['byFirm_withbaseval_loss'], label='byFirm_withbase val_loss', color='b')
plt.plot(range(1, eps+1), parameters['byFirm_withbaseloss'], label='byFirm_withbase loss', color='b')

plt.legend()
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title('Loss for different scaling methods')
plt.show()