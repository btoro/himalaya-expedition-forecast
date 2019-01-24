import dill
import pandas as pd
from data import getExpeditions, getMembers, getPeaks


df_members = getMembers()
df_peaks = getPeaks()
df_exped = getExpeditions()


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

#     print( df_peak_expeditions.EXPID )
    
    df_peak_members =  df_peak_members.loc[ df_peak_members.EXPID.isin( df_peak_expeditions.EXPID ) ]
    return df_peak_members

