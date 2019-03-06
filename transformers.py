from sklearn import base
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline


class SelectColumnsTransfomer(base.BaseEstimator, base.TransformerMixin):
    
    def __init__(self, columns=[]):
        self.columns = columns

    def transform(self, X, **transform_params):

        trans = X[self.columns].copy() 
        return trans

    def fit(self, X, y=None, **fit_params):
        return self
    
class DataFrameFeatureUnion(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, list_of_transformers):
        self.list_of_transformers = list_of_transformers
        
    def transform(self, X, **transformparamn):
        
        concatted = pd.concat([transformer.transform(X)
                            for transformer in
                            self.fitted_transformers_], axis=1).copy()
        return concatted


    def fit(self, X, y=None, **fitparams):
        
        self.fitted_transformers_ = []
        for transformer in self.list_of_transformers:
            fitted_trans = base.clone(transformer).fit(X, y=None, **fitparams)
            self.fitted_transformers_.append(fitted_trans)
        return self
    
class DataFrameFunctionTransformer(base.BaseEstimator, base.TransformerMixin):
   
    def __init__(self, func, impute = False, missing_values = None ):
        self.func = func
        self.impute = impute
        self.series = pd.Series() 
        self.missing_values = missing_values 

    def transform(self, X, **transformparams):
        
        if self.impute:
            trans = pd.DataFrame(X).fillna(self.series).copy()

        else:
            trans = pd.DataFrame(X).apply(self.func).copy()
            
        return trans

    def fit(self, X, y=None, **fitparams):
        if self.impute:
      
            self.series = pd.DataFrame(X).apply(self.func).copy()
 
            
        return self
    
class PandasOneHotEncoderTransformer( base.BaseEstimator, base.TransformerMixin):
    
    def transform(self, X, **transformparams):
        trans = pd.get_dummies(X).copy()
        return trans

    def fit(self, X, y=None, **fitparams):
        return self
    
class BinaryGenderScaler( base.BaseEstimator, base.TransformerMixin):
        
    def __init__(self):
        self.converter = {  'Male': 0,
                            'Female': 1 }
                          
    def transform(self, X, **transformparams):
        trans = X.applymap( lambda x: self.converter[x] )
        return trans

    def fit(self, X, y=None, **fitparams):
        return self
    
    
class SymptomScalerTransformer( base.BaseEstimator, base.TransformerMixin):
    """
     Yes / No / Unknown or
    'Same As Or Less Than Usual' / 'More Than Usual' / 'Unknow
    """
    def __init__(self, converter ):
        self.converter = converter
        
    def transform(self, X, **transformparams):
        trans = X.applymap( lambda x: self.converter[x] )
        return trans

    def fit(self, X, y=None, **fitparams):
        return self
    
    
class PandasStandardScaler( base.BaseEstimator, base.TransformerMixin):
    
    def __init__( self, scaler ):
        self.scaler = scaler
        
    def fit(self, X, y=None):
        
        X[X.columns] = self.scaler.fit_transform(X[X.columns])
        return X


    def transform(self, X, y='deprecated', copy=None):
 
        X[X.columns] = self.scaler.fit_transform(X[X.columns])
        return X

 
    
class PandasVectorizerTransformer( base.BaseEstimator, base.TransformerMixin):
    def __init__( self ):
        self.v = DictVectorizer(sparse=False)
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y='deprecated', copy=None):
 
        print( X )
        trans = self.v.fit_transform(X)
        return pd.DataFrame( trans, columns= self.v.get_feature_names())
 
def processDataframe( df_in, copy=True ):
    
    if copy:
        df = df_in.copy()
    else:
        df = df_in
        

    bins = [0, 18, 30, 40, 50, 60, np.inf]
    names = ['<18', '18-30', '30-40', '40-50', '50-60', '>60']
    df['Age_'] = pd.cut(df['CALCAGE'], bins, labels=names)
    df['Age_'] = df['Age_'].replace(np.nan, 'Unknown', regex=True)

    
    bins = [0, 2000, 4000, 6000,  np.inf]
    names = ['<2000', '2000-4000', '4000-6000', '>6000']
    df['Max_Height_'] = pd.cut(df['MPERHIGHPT'], bins, labels=names, include_lowest=True)
    df['Max_Height_'] = df['Max_Height_'].replace(np.nan, 'Unknown', regex=True)
    
     
#     bins = [0, 2000, 4000, 6000, 8000,  np.inf]
#     names = ['<2000', '2000-4000', '4000-6000', '6000-8000', '>8000']
#     df['P_Height_'] = pd.cut(df['HEIGHTM'], bins, labels=names, include_lowest=True)      

    bins = [-1,  0 , 5, 10, 25, 35, 50, 100, np.inf]
    names = ['0', '1-5', '5-10', '10-25', '25-35', '35-50', '50-100', '>100']
    df['Past_Exped_'] = pd.cut(df['Past Expeditions'], bins, labels=names)
      
    
    df['Difficulty'] = df['SUCCESS_ATTEMPTS']/ ( df['SUCCESS_ATTEMPTS']+df['FAILED_ATTEMPTS'] )
    df['Difficulty'] = df['Difficulty'].replace(np.nan, 0, regex=True)

    
    df["MSEASON"] = df["MSEASON"].astype('category')
    df["MO2USED"] = df["MO2USED"].astype('int')

    return df

def makeTransformerPipeline():
    
    features_for_binning = ['Age_',
                        'Max_Height_',
    #                         'P_Height_',   
                            'MSEASON',
                            'Past_Exped_'
                            ]

    binning_transformer = Pipeline([
            ( 'select', SelectColumnsTransfomer( features_for_binning )),
            ('onehot', PandasOneHotEncoderTransformer())
    ])

    bringThrough_transformer = Pipeline([
            ( 'select', SelectColumnsTransfomer( ['MO2USED', 'Difficulty', 'HEIGHTM'] )),
    ])
    
    union = DataFrameFeatureUnion( [binning_transformer, 
                                    bringThrough_transformer,
                                ]
                                )


    return union
