#%% Dynamic Time Warping
import pandas as pd
import MLfunctions as ML
import dataProcessing as dP
from sklearn.preprocessing import MinMaxScaler
from tslearn.metrics import dtw
import numpy as np

trainingData, trainingIndex, cols = dP.load_and_format_data('./data/S.02_training.csv')
testData, _, _ = dP.load_and_format_data('./data/S.02_test.csv')

#scale all data at once
scalar = MinMaxScaler()
scaledData = scalar.fit_transform(trainingData)
scaledData = pd.DataFrame(scaledData, index=trainingIndex, columns=cols)

DecFirms = pd.read_csv('./data/31 Dec firms.csv')
DecFirms = DecFirms['Entity Reference'].values

#scale on a firm by firm basis, then join together
firmScaled = pd.DataFrame()
for ER in DecFirms:
    
    oneFirm = trainingData.filter(like=str(ER), axis='index')
    
    cols = oneFirm.columns
    index = oneFirm.index
        
    oneFirmScalar = MinMaxScaler()
    oneFirm = oneFirmScalar.fit_transform(oneFirm)
    
    oneFirm = pd.DataFrame(oneFirm, columns=cols, index=index)
   
    firmScaled = firmScaled.append(oneFirm)


AllScaledSimilarity = []
FirmScaledSimilarity = []
#calculate similiarty between firms
for i in range(len(DecFirms)-1):
    print(i)
    
    for j in range(i + 1, len(DecFirms)):
        
        t1 = scaledData.filter(like=str(DecFirms[i]), axis='index')
        t2 = scaledData.filter(like=str(DecFirms[j]), axis='index')
        
        dist1 = dtw(t1, t2)
        AllScaledSimilarity.append(dist1)
        
        t3 = firmScaled.filter(like=str(DecFirms[i]), axis='index')
        t4 = firmScaled.filter(like=str(DecFirms[j]), axis='index')
        
        dist2 = dtw(t3, t4)
        FirmScaledSimilarity.append(dist2)

#save similairty array to save time
'''        
with open('AllScaledSimilarity.npy', 'wb') as f:
    np.save(f, AllScaledSimilarity)
with open('FirmScaledSimilarity.npy', 'wb') as f:
    np.save(f, FirmScaledSimilarity)
    

with open('./numpyData/AllScaledSimilarity.npy', 'rb') as f:
    AllScaledSimilarity = np.load(f)
with open('./numpyData/FirmScaledSimilarity.npy', 'rb') as f:
    FirmScaledSimilarity = np.load(f)
        
'''
sim = {'AllScaled' : AllScaledSimilarity, 'FirmScaled' : FirmScaledSimilarity}

#%% Clustering
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

for s in sim.keys():

    #symmetric matrix from DTW list
    similarityMx = squareform(sim[s])
    clustering = AgglomerativeClustering(n_clusters = 2)
    cluster = clustering.fit_predict(similarityMx)
    
    dendoLink = linkage(sim[s])
    dendrogram(dendoLink)
    plt.tick_params(axis='x',labelbottom=False)
    if s == 'AllScaled': plt.title('Firm scaled as a whole') 
    else: plt.title('Firms scaled individually') 
    
    plt.show()
    

#%% LSTM
from sklearn.model_selection import train_test_split
import tensorflow as tf

similarityMx = squareform(FirmScaledSimilarity)
clustering = AgglomerativeClustering(n_clusters = 2)
cluster = clustering.fit_predict(similarityMx)

firmCluster1 = DecFirms[cluster == 0]
firmCluster2 = DecFirms[cluster == 1]

cluster1 = firmScaled.loc[firmCluster1]
cluster2 = firmScaled.loc[firmCluster2]

trainingFull, _, _ = dP.load_and_format_data('./data/S.02_training.csv', removeSums=False)
testFull, _,_ = dP.load_and_format_data('./data/S.02_test.csv', removeSums=False)

#only run model for firms that reported Q1, have YE in December and have over 5 time periods
reportedQ1 = dP.firm_population(testData, trainingData)


#window set at 5 as 5 periods in a year
window=5
#outlier results
SII_outliersCL=[]
AE_outliersCL=[]
#training MSE results
SII_trainMSECL=[]
AE_trainMSECL=[]
#test MSE results
SII_testMSECL=[]
AE_testMSECL=[]
for cluster in [cluster1, cluster2]:
    
    tf.random.set_seed(7)
        
    Y = np.repeat(1, len(cluster))
    X_train, X_valid, _, _ = train_test_split(cluster, Y, test_size=0.2, random_state=7)
    
    autoencoder, encoder = ML.autoencoder_model()
    history = autoencoder.fit(X_train, X_train, epochs=50, verbose=0,
                              validation_data=(X_valid, X_valid))
    
    encodedTraining = encoder.predict(cluster)
    encodedTraining = pd.DataFrame(encodedTraining, index=cluster.index)
    
    trainingCluster = cluster.join(encodedTraining)
    
    #in both cluster cases, no columns were dropped - more useful AE, but not for LSTM
    dropCols = dP.correlated_columns(trainingCluster, cols)
    
    clusterfirms = np.unique(cluster.index.get_level_values('Entity Reference'))
    firms = [x for x in reportedQ1 if x in clusterfirms]

    for er in firms:
            
        #reset seed for each loop for reproducibility
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
        
        SII_trainMSECL.append(SIITrainMse)
        AE_trainMSECL.append(AETrainMSE)
        
        SII_outliersCL.append(len(SII_outliersDF[SII_outliersDF['Anomaly test']]))
        AE_outliersCL.append(len(AE_outliersDF[AE_outliersDF['Anomaly test']]))
        
        SII_testMSECL.append(SIITestMse)
        AE_testMSECL.append(AETestMmse)
            
#%% Plot results
import seaborn as sns

flaggedFirms=pd.read_csv('./data/Plausibility Response Analysis.csv', usecols=['FRN'])['FRN'].values
flaggedFirms=np.unique(flaggedFirms)
           
#get cluster orderings for proper outlier mapping on graphs
cluster1firms = np.unique(cluster1.index.get_level_values('Entity Reference'))
cluster1firms = [x for x in cluster1firms if x in reportedQ1]

cluster2firms = np.unique(cluster2.index.get_level_values('Entity Reference'))
cluster2firms = [x for x in cluster2firms if x in reportedQ1]

population = np.concatenate((cluster1firms, cluster2firms))

#outliers chart
outliers_clust=pd.DataFrame({'SII outliers' :SII_outliersCL, 'AE outliers':AE_outliersCL, 
                            'cluster':((['cluster 1']*len(cluster1firms)+['cluster 2']*len(cluster2firms))),
                            'outlier':['Outlier firm' if x in flaggedFirms else 'Non-outlier firm' for x in population]})
ax=sns.lmplot(x='AE outliers', y='SII outliers', data=outliers_clust, x_jitter=0.3, y_jitter=0.3, hue='outlier',
                ci = None)
plt.title('Relationship between detected SII and autoencoder outliers\n(clusters)')
plt.show()

#MSE training chart
MSEtrain_clust=pd.DataFrame({'SII mse (log)' :np.log(SII_trainMSECL), 'AE mse':AE_trainMSECL, 
                       'colour':(['cluster 1']*len(cluster1firms)+['cluster 2']*len(cluster2firms)),
                            'outlier':['Outlier firm' if x in flaggedFirms else 'Non-outlier firm' for x in population]})
ax=sns.scatterplot(x='AE mse', y='SII mse (log)', data=MSEtrain_clust, x_jitter=0.3, y_jitter=0.3, hue='outlier')
ax.set_title('Relationship between training MSE of SII and autoencoder features\n(clusters)')
plt.show()

#MSE test chart
MSEtest_clust=pd.DataFrame({'SII mse (log)' :np.log(SII_testMSECL), 'AE mse':AE_testMSECL, 
                       'colour':(['cluster 1']*len(cluster1firms)+['cluster 2']*len(cluster2firms)),
                            'outlier':['Outlier firm' if x in flaggedFirms else 'Non-outlier firm' for x in population]})
ax=sns.scatterplot(x='AE mse', y='SII mse (log)', data=MSEtest_clust, x_jitter=0.3, y_jitter=0.3, hue='outlier')
ax.set_title('Relationship between test MSE of SII and autoencoder features\n(clusters)')
plt.show()
