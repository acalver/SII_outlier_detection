#%% whole population files
'''        
with open('./numpyData/First100SIIMSE_train.npy', 'wb') as f:
    np.save(f, SII_trainMSE)
with open('./numpyData/First100AEMSE_train.npy', 'wb') as f:
    np.save(f, AE_trainMSE)
    
with open('./numpyData/First100SIIMSE_test.npy', 'wb') as f:
    np.save(f, SII_testMSE)
with open('./numpyData/First100AEMSE_test.npy', 'wb') as f:
    np.save(f, AE_testMSE)

with open('./numpyData/First100SIIoutliers.npy', 'wb') as f:
    np.save(f, SII_outliers)
with open('./numpyData/First100AEoutliers.npy', 'wb') as f:
    np.save(f, AE_outliers)



with open('./numpyData/Last100SIIMSE_train.npy', 'wb') as f:
    np.save(f, SII_trainMSE)
with open('./numpyData/Last100AEMSE_train.npy', 'wb') as f:
    np.save(f, AE_trainMSE)
    
with open('./numpyData/Last100SIIMSE_test.npy', 'wb') as f:
    np.save(f, SII_testMSE)
with open('./numpyData/Last100AEMSE_test.npy', 'wb') as f:
    np.save(f, AE_testMSE)

with open('./numpyData/Last100SIIoutliers.npy', 'wb') as f:
    np.save(f, SII_outliers)
with open('./numpyData/Last100AEoutliers.npy', 'wb') as f:
    np.save(f, AE_outliers)
    
'''
import numpy as np

def load_outliers():
    with open('./numpyData/First100SIIoutliers.npy', 'rb') as f:
        First100outliersSII = np.load(f)
    with open('./numpyData/Last100SIIoutliers.npy', 'rb') as f:
        Last100outliersSII = np.load(f)
        
    outliersSII = np.concatenate((First100outliersSII, Last100outliersSII))
        
    
    with open('./numpyData/First100AEoutliers.npy', 'rb') as f:
        First100outliersAE = np.load(f)
    with open('./numpyData/Last100AEoutliers.npy', 'rb') as f:
        Last100outliersAE = np.load(f)
    
    outliersAE = np.concatenate((First100outliersAE, Last100outliersAE))

    
    with open('./numpyData/First100SIIMSE_train.npy', 'rb') as f:
        First100MSE_SIItrain = np.load(f)
    with open('./numpyData/Last100SIIMSE_train.npy', 'rb') as f:
        Last100MSE_SIItrain = np.load(f)
    
    MSE_SIItrain = np.concatenate((First100MSE_SIItrain, Last100MSE_SIItrain))

    
    with open('./numpyData/First100AEMSE_train.npy', 'rb') as f:
        First100MSE_AEtrain = np.load(f)
    with open('./numpyData/Last100AEMSE_train.npy', 'rb') as f:
        Last100MSE_AEtrain = np.load(f)
        
    MSE_AEtrain = np.concatenate((First100MSE_AEtrain, Last100MSE_AEtrain))
    
    
    with open('./numpyData/First100SIIMSE_test.npy', 'rb') as f:
        First100MSE_SIItest = np.load(f)
    with open('./numpyData/Last100SIIMSE_test.npy', 'rb') as f:
        Last100MSE_SIItest = np.load(f)
    
    MSE_SIItest = np.concatenate((First100MSE_SIItest, Last100MSE_SIItest))
    
    
    with open('./numpyData/First100AEMSE_test.npy', 'rb') as f:
        First100MSE_AEtest = np.load(f)
    with open('./numpyData/Last100AEMSE_test.npy', 'rb') as f:
        Last100MSE_AEtest = np.load(f)
        
    MSE_AEtest = np.concatenate((First100MSE_AEtest, Last100MSE_AEtest))
    
    return outliersSII, outliersAE, MSE_SIItrain, MSE_AEtrain, MSE_SIItest, MSE_AEtest



SII_outliers, AE_outliers, SII_trainMSE, AE_trainMSE, SII_testMSE, AE_testMSE = load_outliers()


#%% cluster files
'''        
with open('./numpyData/cluster1SIIMSE_train.npy', 'wb') as f:
    np.save(f, SII_trainMSE)
with open('./numpyData/cluster1AEMSE_train.npy', 'wb') as f:
    np.save(f, AE_trainMSE)
    
with open('./numpyData/cluster1SIIMSE_test.npy', 'wb') as f:
    np.save(f, SII_testMSE)
with open('./numpyData/cluster1AEMSE_test.npy', 'wb') as f:
    np.save(f, AE_testMSE)

with open('./numpyData/cluster1SIIoutliers.npy', 'wb') as f:
    np.save(f, SII_outliers)
with open('./numpyData/cluster1AEoutliers.npy', 'wb') as f:
    np.save(f, AE_outliers)



with open('./numpyData/cluster2SIIMSE_train.npy', 'wb') as f:
    np.save(f, SII_trainMSE)
with open('./numpyData/cluster2AEMSE_train.npy', 'wb') as f:
    np.save(f, AE_trainMSE)
    
with open('./numpyData/cluster2SIIMSE_test.npy', 'wb') as f:
    np.save(f, SII_testMSE)
with open('./numpyData/cluster2AEMSE_test.npy', 'wb') as f:
    np.save(f, AE_testMSE)

with open('./numpyData/cluster2SIIoutliers.npy', 'wb') as f:
    np.save(f, SII_outliers)
with open('./numpyData/cluster2AEoutliers.npy', 'wb') as f:
    np.save(f, AE_outliers)
    
'''

def load_cluster_outliers():
    with open('./numpyData/cluster1SIIoutliers.npy', 'rb') as f:
        cluster1outliersSII = np.load(f)
    with open('./numpyData/cluster2SIIoutliers.npy', 'rb') as f:
        cluster2outliersSII = np.load(f)
        
    outliersSII = np.concatenate((cluster1outliersSII, cluster2outliersSII))
        
    
    with open('./numpyData/cluster1AEoutliers.npy', 'rb') as f:
        cluster1outliersAE = np.load(f)
    with open('./numpyData/cluster2AEoutliers.npy', 'rb') as f:
        cluster2outliersAE = np.load(f)
    
    outliersAE = np.concatenate((cluster1outliersAE, cluster2outliersAE))

    
    with open('./numpyData/cluster1SIIMSE_train.npy', 'rb') as f:
        cluster1MSE_SIItrain = np.load(f)
    with open('./numpyData/cluster2SIIMSE_train.npy', 'rb') as f:
        cluster2MSE_SIItrain = np.load(f)
    
    MSE_SIItrain = np.concatenate((cluster1MSE_SIItrain, cluster2MSE_SIItrain))

    
    with open('./numpyData/cluster1AEMSE_train.npy', 'rb') as f:
        cluster1MSE_AEtrain = np.load(f)
    with open('./numpyData/cluster2AEMSE_train.npy', 'rb') as f:
        cluster2MSE_AEtrain = np.load(f)
        
    MSE_AEtrain = np.concatenate((cluster1MSE_AEtrain, cluster2MSE_AEtrain))
    
    
    with open('./numpyData/cluster1SIIMSE_test.npy', 'rb') as f:
        cluster1MSE_SIItest = np.load(f)
    with open('./numpyData/cluster2SIIMSE_test.npy', 'rb') as f:
        cluster2MSE_SIItest = np.load(f)
        
    MSE_SIItest = np.concatenate((cluster1MSE_SIItest, cluster2MSE_SIItest))
    
    
    with open('./numpyData/cluster1AEMSE_test.npy', 'rb') as f:
        cluster1MSE_AEtest = np.load(f)
    with open('./numpyData/cluster2AEMSE_test.npy', 'rb') as f:
        cluster2MSE_AEtest = np.load(f)
        
    MSE_AEtest = np.concatenate((cluster1MSE_AEtest, cluster2MSE_AEtest))
    
    return outliersSII, outliersAE, MSE_SIItrain, MSE_AEtrain, MSE_SIItest, MSE_AEtest


SII_outliersCL, AE_outliersCL, SII_trainMSECL, AE_trainMSECL, SII_testMSECL, AE_testMSECL = load_cluster_outliers()
