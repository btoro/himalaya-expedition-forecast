import pandas as pd


def getPeaks():
    df = pd.read_csv('data/peaks.DBF.csv')
    return df

def getExpeditions():
    df_exped = pd.read_csv('data/exped.DBF.csv')
    return df_exped

def getMembers():
    df_members = pd.read_csv('data/members.DBF.csv')
    return df_members