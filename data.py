import pandas as pd
# import requests
# from io import StringIO  # got moved to io in python3.

def getPeaks():
    # url = 'https://docs.google.com/spreadsheets/d/11psC5WkkjYXHLYrfWUr69M9ds9Un_4R0O9aGL6B3_dY/export?format=csv&gid=798216563'
    # r = requests.get( url )
    # data = r.content.decode('utf-8')

    url = 'https://docs.google.com/spreadsheets/d/1pix2FmPPww5_3XxGFe9uC-9dM51xPAUCTR3fLAjrgJg/export?gid=743836239&format=csv'
    df = pd.read_csv( url )
    # df = pd.read_csv('data/peaks.DBF.csv')
    return df

def getExpeditions():

    url = 'https://docs.google.com/spreadsheets/d/11psC5WkkjYXHLYrfWUr69M9ds9Un_4R0O9aGL6B3_dY/export?gid=798216563&format=csv'
    df_exped = pd.read_csv( url )
    # df_exped = pd.read_csv('data/exped.DBF.csv')
    return df_exped

def getMembers():

    url = 'https://drive.google.com/uc?authuser=0&id=1dMq6sNy4_Cw1bcdq8RAPqdRIvqh_TnNK&export=download'
    

    df_members = pd.read_csv(url)
    # df_members = pd.read_csv('data/members.DBF.csv')
    return df_members