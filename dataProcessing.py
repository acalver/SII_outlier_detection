import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_format_data(file, removeSums=True):
    """
    Load and format SII S.02.01 Balance Sheet data

    Parameters
    ----------
    file : str
        file path to data
    removeSums : bool, optional
        Option to remove the sum values from S.02.01. The default is True.

    Returns
    -------
    data : DataFrame
        return data pivoted on reporting period and entity reference.
    index : index
        index of returned data.
    cols : index
        columns of returned data.

    """

    data = pd.read_csv(file, dtype={'Entity Reference':'str'})
    
    if removeSums:
        data = data[~data['Row Code'].isin(['R0070', 'R0100', 'R0130', 'R0230', 'R0270',
                           'R0280', 'R0310', 'R0500',
                           'R0510', 'R0520', 'R0560', 'R0600', 'R0610',
                           'R0650', 'R0690', 'R0850', 'R0900', 'R1000',
                           'R0010', 'R0020', 'R0730' #these are only valid for C0020 on S.02
                           ])]
    data = data[['Entity Short Name', 'Entity Reference', 'Row Code',
               'Cell Ref Code', 'Observation Value GBP',
               'Reporting Period']]
    data = data.pivot_table(index=['Entity Reference', 'Reporting Period'],
                    columns='Cell Ref Code', values='Observation Value GBP')
    index = data.index
    cols = data.columns

    
    data = data.fillna(0)
    
    return data, index, cols

def correlated_columns(data, SIIcols):
    """
    remove correlated columns

    Parameters
    ----------
    data : dataframe.
    SIIcols : list
        list of SII features.

    Returns
    -------
    corr_cols : list
        columns with high correlation to other columns.

    """
    
    corrMatrix = data.corr()
    #only test correlation of generated features
    corrMatrix = corrMatrix.drop(SIIcols, axis='index')
    
    corr_cols=[]
    #track column pairs
    corr_tuples=[]
    for r in corrMatrix.index:
        for c in corrMatrix.columns:
            
            if abs(corrMatrix.loc[r,c])>0.7 and r != c and (c,r) not in corr_tuples:
                
                corr_cols.append(r)
                #needed to avoid removing both columns
                corr_tuples.append((r, c))
            
    corr_cols = np.unique(corr_cols)
    
    return corr_cols

def firm_population(ReportingP, training):
    
    reportedQ1 = ReportingP.index.get_level_values('Entity Reference')
    DecFirms = pd.read_csv('./data/31 Dec firms.csv')
    DecFirms = DecFirms['Entity Reference'].values
    reportedQ1 = [x for x in reportedQ1 if x in DecFirms]
    
    belowWindow=training.index.to_frame(index=False)
    belowWindow=belowWindow.groupby(['Entity Reference']).size()
    belowWindow=belowWindow[belowWindow<=5].index
    
    reportedQ1 = [y for y in reportedQ1 if y not in belowWindow]
    
    return reportedQ1

def single_firm(ER, data):
    """
    Filter data to select a single firm, then scale
    
    Parameters
    ----------
    ER : str
        Entity Reference to filter on.
    data : DataFrame
        input dataframe of firms.

    Returns
    -------
    single_f : DataFrame
        filtered dataframe.
    scalar : MinMaxScaler()
        scalar specific for the ER firm selected.

    """
    
    single_f = data.filter(like=str(ER), axis='index')
    
    cols = single_f.columns
    index = single_f.index
    
    scalar = MinMaxScaler()
    single_f = scalar.fit_transform(single_f)
    
    single_f = pd.DataFrame(single_f, columns=cols, index=index)
    
    return single_f, scalar

def time_series_windows(data, window):
    """
    Creates time series blocks for LSTM 
    
    Parameters
    ----------
    data : dataframe
        time series dataframe.
    window : int
        size of time series windows to split data into.

    Returns
    -------
    3D time series array for LSTM, corresponding 2D array of outputs for each window
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
        
    X, Y = [], []
    for i in range(len(data)-window):
        #add each window period to a list so output is 3D
        period = data[i:(i + window), :]
        X.append(period)
        Y.append(data[i + window, :])
    return np.array(X), np.array(Y)


def calculated_SII_fields(data):
    """
    Calculates summed columns on S.02

    Parameters
    ----------
    data : dataframe

    Returns
    -------
    data : dataframe

    """
    
    #Assets
    data['S.02.01R0100C0010'] = data['S.02.01R0110C0010'] + data['S.02.01R0120C0010']
    data['S.02.01R0130C0010'] = data['S.02.01R0140C0010'] + data['S.02.01R0150C0010'] + \
                                    data['S.02.01R0160C0010'] + data['S.02.01R0170C0010']
    data['S.02.01R0070C0010'] = data['S.02.01R0080C0010'] + data['S.02.01R0090C0010'] + \
                                    data['S.02.01R0100C0010'] + data['S.02.01R0130C0010'] + \
                                    data['S.02.01R0180C0010'] +  data['S.02.01R0190C0010'] + \
                                    data['S.02.01R0200C0010'] + data['S.02.01R0210C0010']
    data['S.02.01R0230C0010'] = data['S.02.01R0240C0010'] + data['S.02.01R0250C0010'] + \
                                    data['S.02.01R0260C0010']
    data['S.02.01R0280C0010'] = data['S.02.01R0290C0010'] + data['S.02.01R0300C0010']
    data['S.02.01R0310C0010'] = data['S.02.01R0320C0010'] + data['S.02.01R0330C0010']
    data['S.02.01R0270C0010'] = data['S.02.01R0280C0010'] + data['S.02.01R0310C0010'] + \
                                    data['S.02.01R0340C0010']
                                    
    data['S.02.01R0500C0010'] = data['S.02.01R0030C0010'] + data['S.02.01R0040C0010'] + \
                                    data['S.02.01R0050C0010'] + data['S.02.01R0060C0010'] + \
                                    data['S.02.01R0070C0010'] + data['S.02.01R0220C0010'] + \
                                    data['S.02.01R0230C0010'] + data['S.02.01R0270C0010'] + \
                                    data['S.02.01R0350C0010'] + data['S.02.01R0360C0010'] + \
                                    data['S.02.01R0370C0010'] + data['S.02.01R0380C0010'] + \
                                    data['S.02.01R0390C0010'] + data['S.02.01R0400C0010'] + \
                                    data['S.02.01R0410C0010'] + data['S.02.01R0420C0010']
    
    #Liabilities
    data['S.02.01R0520C0010'] = data['S.02.01R0530C0010'] + data['S.02.01R0540C0010'] + \
                                    data['S.02.01R0550C0010']
    data['S.02.01R0560C0010'] = data['S.02.01R0570C0010'] + data['S.02.01R0580C0010'] + \
                                    data['S.02.01R0590C0010']
    data['S.02.01R0610C0010'] = data['S.02.01R0620C0010'] + data['S.02.01R0630C0010'] + \
                                    data['S.02.01R0640C0010']
    data['S.02.01R0650C0010'] = data['S.02.01R0660C0010'] + data['S.02.01R0670C0010'] + \
                                    data['S.02.01R0680C0010']
    data['S.02.01R0690C0010'] = data['S.02.01R0700C0010'] + data['S.02.01R0710C0010'] + \
                                    data['S.02.01R0720C0010']
    data['S.02.01R0850C0010'] = data['S.02.01R0860C0010'] + data['S.02.01R0870C0010']
    data['S.02.01R0510C0010'] = data['S.02.01R0520C0010'] + data['S.02.01R0560C0010'] 
    data['S.02.01R0600C0010'] = data['S.02.01R0610C0010'] + data['S.02.01R0650C0010'] 
    
    data['S.02.01R0900C0010'] = data['S.02.01R0510C0010'] + data['S.02.01R0600C0010'] + \
                                data['S.02.01R0690C0010'] + data['S.02.01R0740C0010'] + \
                                data['S.02.01R0750C0010'] + data['S.02.01R0760C0010'] + \
                                data['S.02.01R0770C0010'] + data['S.02.01R0780C0010'] + \
                                data['S.02.01R0790C0010'] + data['S.02.01R0800C0010'] + \
                                data['S.02.01R0810C0010'] + data['S.02.01R0820C0010'] + \
                                data['S.02.01R0830C0010'] + data['S.02.01R0840C0010'] + \
                                data['S.02.01R0850C0010'] + data['S.02.01R0880C0010']

    
    data['S.02.01R1000C0010'] = data['S.02.01R0500C0010'] - data['S.02.01R0900C0010'] 
    
    return data