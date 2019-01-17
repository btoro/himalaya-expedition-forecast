
import pandas as pd
import numpy as np

from bokeh.models import Range1d
from bokeh.models.sources import ColumnDataSource
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.models.callbacks import CustomJS

from bokeh.sampledata.us_states import data as statesData
from bokeh.models import HoverTool


def createUSMap():

       if 'HI' in statesData:
              del statesData["HI"]

       if 'AK' in statesData:
              del statesData["AK"]

       #Extract longitude and latitude of the state boundaries
       state_xs = [statesData[code]["lons"] for code in statesData]
       state_ys = [statesData[code]["lats"] for code in statesData]

       state_names=[]

       for state in statesData:
              statename = statesData[state]['name']
              state_names.append(statename)


       p = figure(title=" ", toolbar_location="left",
              tools="pan,wheel_zoom,box_zoom,reset,hover,save",
              x_axis_location=None, y_axis_location=None, plot_width=1100, plot_height=700)

       p.grid.grid_line_color = None

       basecolors = ['red', 'green', 'blue', 'orange', 'brown']
       state_colors = np.random.choice( basecolors, len(state_xs))


       data_source = ColumnDataSource(data=dict(
       x=state_xs,
       y=state_ys,
       color=state_colors,
       name=state_names,
       # red_frac=state_red_frac,
       # blue_frac=state_blue_frac,
       # yellow_frac=state_yellow_frac,
       # red_frac_err=state_red_frac_err,
       # blue_frac_err=state_blue_frac_err,
       # yellow_frac_err=state_yellow_frac_err,
       # total_count=state_total_count,
       ))


       p.patches('x', 'y', source=data_source,
              fill_color='color', fill_alpha=0.7,
              line_color="black", line_width=0.5)

       hover = p.select_one(HoverTool)
       hover.point_policy = "follow_mouse"
       hover.tooltips = [
       ("State", "@name"),
       ]
       

       press_state = CustomJS(  code="    link = \"http://stackoverflow.com\"; \
                                          console.log( '-----'); \
                                          console.log( cb_obj ); \
                                          //window.location.href = link; \
                                   " )

       hover.js_on_event('tap', press_state)



       return p 