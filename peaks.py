### Plots for peak 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from bokeh.models import Range1d
from bokeh.plotting import figure
from data import getPeaks, getMembers


df_peaks = getPeaks()
df_members = getMembers()

def plot_visits_month_peak( peak ):
    
    visits = df_members[ df_members['PEAKID'] == peak ]
    
    return visits.groupby(['MYEAR'])['FNAME'].count()
    

    
def visits_over_time( peak ):


    data = plot_visits_month_peak( peak )

        
    # title = 'Visits over time'
    TOOLS = "hover,save,reset"

    p = figure(  tools=TOOLS, background_fill_color="#fafafa" , width=1000, height=300)

    p.vbar(x= data.index, top=data.values, width=0.9)


    # p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
    #        fill_color="navy", line_color="white", alpha=0.5)

    interval = 0.1*data.values.max()
    p.y_range = Range1d(0, data.values.max()+interval, bounds=(0,  data.values.max()+interval))
    # p.y_range.start = 0
    # p.legend.location = "center_right"
    # p.legend.background_fill_color = "#fefefe"
    p.xaxis.axis_label = 'Year'
    p.yaxis.axis_label = 'Climbers'
    p.grid.grid_line_color="white"

    p.background_fill_alpha = 0.0
    p.border_fill_alpha = 0.0

    p.xaxis.major_label_text_font_size = "18pt"
    p.yaxis.major_label_text_font_size = "18pt"

    p.xaxis.axis_label_text_font_size = "20pt"
    p.yaxis.axis_label_text_font_size = "20pt"
    return p

def get_peak_info( peakid ):

    return {'name': df_peaks[df_peaks.PEAKID == peakid]['PKNAME'].values, \
            'height': df_peaks[df_peaks.PEAKID == peakid]['HEIGHTM'].values
                     }