import dill
import pandas as pd
from data import getExpeditions, getMembers, getPeaks

## basics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## python pkgs
import re

## sklearn
from sklearn.pipeline import make_pipeline, FeatureUnion, Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer

##estimators
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_recall_fscore_support,accuracy_score,confusion_matrix
from sklearn.model_selection import cross_validate

from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
sns.set_palette("colorblind")


import dill


from bokeh.io import output_file, show
from bokeh.models import BasicTicker, ColorBar, ColumnDataSource, LinearColorMapper, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.transform import transform

from bokeh.transform import dodge
from bokeh.core.properties import value
from math import pi

from transformers import DataFrameFeatureUnion, DataFrameFunctionTransformer, SelectColumnsTransfomer, PandasOneHotEncoderTransformer, PandasStandardScaler, PandasVectorizerTransformer
from transformers import processDataframe, makeTransformerPipeline


trans_pipe = makeTransformerPipeline()

clf = dill.load(open('static/peak_clf.dill', 'rb'))


col_to_add = ['Age__18-30', 'Age__30-40', 'Age__40-50', 'Age__50-60', 'Age__<18', 'Age__>60', 'Age__Unknown', 'Difficulty', 'HEIGHTM', 'MO2USED', 'MSEASON_0', 'MSEASON_1', 'MSEASON_2', 'MSEASON_3', 'MSEASON_4', 'Max_Height__2000-4000', 'Max_Height__4000-6000', 'Max_Height__<2000', 'Max_Height__>6000', 'Past_Exped__0', 'Past_Exped__1-5', 'Past_Exped__10-25', 'Past_Exped__25-35', 'Past_Exped__35-50', 'Past_Exped__5-10', 'Past_Exped__50-100', 'Past_Exped__>100']

df_members = getMembers()
df_peaks = getPeaks()
df_exped = getExpeditions()

peaks = df_peaks[ pd.notnull(df_peaks['PSMTDATE'] ) ]


    
def predict_one_peak( peakid, age, useoxygen, exp_over_4000 ):

    ## First see if there is a model:
    est = load_peak_model( peakid )
    
    ## if we have the model lets use it
    ## if we dont, make one and store it.

    if not est:
        est = build_peak_model( peakid )

    ## predict
    case = [[ age, exp_over_4000, useoxygen]]
    return est.predict( case ), est.predict_proba( case )

def load_peak_model( peakid ):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV,ShuffleSplit


    try:
        est = dill.load(open( '{}_est.dill'.format( peakid ), 'rb'))
        return est
    except:
        return None

def build_peak_model( peakid ):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV,ShuffleSplit
    
    pipe = Pipeline([ ( 'kn', KNeighborsClassifier() )
    ])

    param_grid = {
    'kn__n_neighbors': range(1,50,2)
    }

    ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)


    est = GridSearchCV(pipe, param_grid, iid=False, cv=ss,
                        return_train_score=False, verbose=10)


    data = get_data_for_model( peakid )
    y = data['Success']
    X = data.drop( ['Success'] , axis=1) 
    y=y.astype('int')

    
    est.fit(X, y)

    dill.dump(est, open( '{}_est.dill'.format( peakid ) , 'wb'))

    return est 


def get_data_for_model( peakid ):
    df = pd.DataFrame( columns=['Age', 'PastExpeditions', 'Success', 'O2'])
    
    exps = find_exped_by_peak( peakid )
    
    for i, expedition in exps.iterrows():
        
        members = find_members_by_exped( expedition.EXPID )
        
        result = expedition.TERMREASON
        
        if result == 1 or result == 3:
            result = 1
        else:
            result = 0
        
        for i, member in members.iterrows():
            
            age = member.CALCAGE
            
            if age < 1:
                continue
            
            past_expeditions = df_members[ (df_members['LNAME'] == member['LNAME']) ].LNAME.count()
            
            oxygen =  member.MO2USED*1
            
            insert = {'Age': age, 'PastExpeditions': past_expeditions , 'Success': result, 'O2': oxygen}
            df.loc[len(df)] = insert 

    
    return df


###### Getter functions

def find_exped_by_peak( peak ):
    
    df_select = df_exped.copy()
    
    df_select =  df_select[ df_select.PEAKID == peak ]
    
    return df_select


def find_members_by_exped( expedition ):
    
    
    df_exped_members = df_members.copy()

    df_exped_members =  df_exped_members[ df_exped_members.EXPID == expedition ]
    return df_exped_members

def find_member( lname ):
    
    
    df_exped_members = df_members.copy()

    df_exped_members =  df_exped_members[ df_exped_members.LNAME == lname ]
    return df_exped_members


def find_members_by_peak( peak ):
    
    df_peak_expeditions = find_exped_by_peak(peak)
    
    df_peak_members = df_members.copy()
    
    df_peak_members =  df_peak_members.loc[ df_peak_members.EXPID.isin( df_peak_expeditions.EXPID ) ]
    return df_peak_members



def getConfusionMatrix():

    ### load data
    df = pd.read_csv('data/confusion_matrix.data')

    return plotConfusionMatrix( df , 500, 400 )


def plotConfusionMatrix( df , width, height ):

    from bokeh.palettes import Blues8 

    # Had a specific mapper to map color with value
    mapper = LinearColorMapper(
        palette=Blues8[::-1], low=df.value.min(), high=df.value.max())

    TOOLS = "hover,save,reset"


    # Define a figure
    p = figure(
        plot_width=width,
        plot_height=height,
    #     title="",
        x_range=list(df.Treatment.drop_duplicates()),
        y_range=list(df.Prediction.drop_duplicates()),
        toolbar_location='above',
        tools=TOOLS,
        tooltips=[('Counts', '@value')],
        x_axis_location="below")

    p.xaxis.axis_label = "Prediction"
    p.yaxis.axis_label = "Truth"


    # Create rectangle for heatmap
    p.rect(
        x="Prediction",
        y="Treatment",
        width=1,
        height=1,
        source=ColumnDataSource(df),
        line_color=None,
        fill_color=transform('value', mapper))

    # Add legend
    color_bar = ColorBar(
        color_mapper=mapper,
        location=(0, 0),
        label_standoff=12,
        border_line_color=None,
        ticker=BasicTicker(desired_num_ticks=len(Blues8)))

    color_bar.background_fill_alpha = 0.0

    p.add_layout(color_bar, 'right')

    p.background_fill_alpha = 0.0
    p.border_fill_alpha = 0.0

    return p

def  getFeatureImportances():
    
    ### load data
    df = pd.read_csv('data/feat_importances.data', index_col=0)

    return plot_feature_importance_bokeh( df , 10, 600, 400 )


def plot_feature_importance_bokeh( df2, topN, width, height ):
           
    df2 = df2.sort_values(by='Importance',ascending=False)
    df2= df2.iloc[0:topN]
    
    p = figure(plot_width=width, plot_height=height, 
               y_range= list(df2.index),
               toolbar_location='above', 
    #                tooltips=[("MPG", "@mpg_mean"), 
    #                      ("Cyl, Mfr", "@cyl_mfr")]
              )

    p.hbar(y='index', right='Importance', height=0.9, left=0, source=df2,
           line_color="blue", fill_color='blue')

    #     p.y_range.start = 0
    p.x_range.range_padding = 0.05
    p.y_range.range_padding = 0.01

    p.xgrid.grid_line_color = None
    p.xaxis.axis_label = "Relative Importance"
    p.yaxis.axis_label = "Feature"

    p.xaxis.major_label_orientation = 1.2
    p.outline_line_color = None

    p.background_fill_alpha = 0.0
    p.border_fill_alpha = 0.0

    p.xaxis.major_label_text_font_size = "14pt"
    p.yaxis.major_label_text_font_size = "14pt"

    p.xaxis.axis_label_text_font_size = "20pt"
    p.yaxis.axis_label_text_font_size = "20pt"
    return p


def  getModelSelectionFigure():
    
    ### load data
    df = pd.read_csv('data/model_selection.data' )

    return plotModelSelection( df )


def plotModelSelection( df ):
    source = ColumnDataSource(data=df)

    p = figure(x_range=list(df.Model), y_range=(0, 1), plot_height=500, width=600, 
               toolbar_location='above', tools="save,reset" )

    p.vbar(x=dodge('Model', -0.25, range=p.x_range), top='Accuracy', width=0.2, source=source,
           color="#F1C40F", legend=value("Accuracy"))

    p.vbar(x=dodge('Model',  0.0,  range=p.x_range), top='Precision', width=0.2, source=source,
           color="#718dbf", legend=value("Precision"))

    p.vbar(x=dodge('Model',  0.25, range=p.x_range), top='Recall', width=0.2, source=source,
           color="#e84d60", legend=value("Recall"))

    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.legend.location =(1, 1)
    p.legend.orientation = "horizontal"
 
    p.xaxis.axis_label = "Classifier"
    p.yaxis.axis_label = "Score"
    
    p.xaxis.major_label_orientation = pi/4

    p.xaxis.major_label_text_font_size = "14pt"
    p.yaxis.major_label_text_font_size = "14pt"

    p.xaxis.axis_label_text_font_size = "20pt"
    p.yaxis.axis_label_text_font_size = "20pt"

    p.background_fill_alpha = 0.0
    p.border_fill_alpha = 0.0
    return p

def makeRecommendorPlot(  data ):
     

    source = ColumnDataSource( data)

    p = figure( y_range=data.Name.tolist(), plot_height=450, plot_width=600, toolbar_location=None )
    p.hbar(y='Name', right='Probability', height=0.5, source=source, 
        line_color='white' )

    p.xgrid.grid_line_color = None
    p.x_range.start = 0
    p.x_range.end = 1

    p.xaxis.major_label_text_font_size = "14pt"
    p.yaxis.major_label_text_font_size = "14pt"

    p.xaxis.axis_label_text_font_size = "20pt"
    p.yaxis.axis_label_text_font_size = "20pt"

    p.xaxis.axis_label = "Predicted probability of success"
    p.yaxis.axis_label = "Himalayan Mountain"

    p.background_fill_alpha = 0.0
    p.border_fill_alpha = 0.0
    return p



def makePredictions( kind, peak, age, oxygen, season, pmheight, expcount ):
    
    personal = [ age, oxygen, season, pmheight, expcount ]
    
    if kind == 1:
        p = df_peaks[ df_peaks['PEAKID'] ==  peak ]
        data = personal.copy()
        data.extend( [p['HEIGHTM'], p['SUCCESS_ATTEMPTS'], p['FAILED_ATTEMPTS'] ] )
        data = [data]
    elif kind == 2:
        peak_height = peak
        data = personal.copy()
        data.append( peak_height )
        data = [data]

    elif kind == 3:
        data = [] 
 
        for i, p in peaks.iterrows():
            row = personal.copy()
            row.extend( [p['HEIGHTM'], p['SUCCESS_ATTEMPTS'], p['FAILED_ATTEMPTS'] ] )
            
            data.append( row )
    
    df = pd.DataFrame(columns=['CALCAGE', 'MO2USED' , 'MSEASON', 'MPERHIGHPT', 'Past Expeditions', 'HEIGHTM', 'SUCCESS_ATTEMPTS', 'FAILED_ATTEMPTS' ], data=data)
    X = processDataframe( df )
    X = trans_pipe.fit_transform( X, 0 )
    
    for c in col_to_add:
        if c not in X:
            X[c] = 0 
            
    X = X.reindex(sorted(X.columns), axis=1)   


    results = clf.predict_proba(X)
    # print(results)

    if kind == 3:
        peak_names = peaks.PKNAME.tolist()

        top_p = dict()
        for i, res in enumerate(results):
            top_p[peak_names[i]] = res[1]
            
        top_p = sorted(top_p.items(), key=lambda kv: -kv[1])
        df_top_success = pd.DataFrame( top_p[0:5] , columns=['Name' , 'Probability'])

        p = makeRecommendorPlot(  df_top_success )
        return  None,None, p
    
    return  clf.predict( X ), results, None


