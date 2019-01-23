### Plots for peak 
from bokeh.plotting import figure

def visits_over_time():


    data = []

        

    # output to static HTML file
    output_file("stocks.html", title="stocks.py example")

    # create a new plot with a a datetime axis type
    p = figure(width=800, height=350, x_axis_type="datetime")

    # add renderers
    p.circle(aapl_dates, aapl, size=4, color='darkgrey', alpha=0.2, legend='close')
    p.line(aapl_dates, aapl_avg, color='navy', legend='avg')

    # NEW: customize by setting attributes
    p.title.text = "AAPL One-Month Average"
    p.legend.location = "top_left"
    p.grid.grid_line_alpha=0
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Price'
    p.ygrid.band_fill_color="olive"
    p.ygrid.band_fill_alpha = 0.1