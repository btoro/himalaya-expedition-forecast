import os
from flask import Flask, render_template, request, redirect, flash
from flask_wtf import FlaskForm as Form
from wtforms import FieldList, FormField, StringField, TextField, FloatField, IntegerField, BooleanField, TextAreaField, SubmitField, RadioField, SelectField, DateField, validators
from wtforms.fields.html5 import IntegerRangeField
app = Flask(__name__)


from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models.sources import ColumnDataSource

from graphs_maps import createUSMap


@app.route('/', methods=['get'])
def index():
    plot = createUSMap()
    script, div = components(plot)

    return render_template('index.html', the_div=div, the_script=script)

@app.route('/overview', methods=['get'])
def overview():
    return render_template('overview.html')

if __name__ == '__main__':
    app.run(port=33507, debug=True)
