import os
from flask import Flask, render_template, request, redirect, flash
from flask_wtf import FlaskForm as Form
from wtforms import FieldList, FormField, StringField, TextField, FloatField, IntegerField, BooleanField, TextAreaField, SubmitField, RadioField, SelectField, DateField, validators
from wtforms.fields.html5 import IntegerRangeField

import requests


from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models.sources import ColumnDataSource

from graphs_maps import createUSMap


app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'



peakList = [('cpp', 'C++'), ('py', 'Python'), ('text', 'Plain Text')]

class OnePeakPredict_Form(Form):
    peak  = SelectField(u'Select Peak', choices=peakList)

    age    = IntegerField('Age', [validators.NumberRange(min=18, max=100)])

    useoxygen = BooleanField( 'Plan to use O2?' )

    exp_over_6000    = IntegerField('6000 meters', [validators.NumberRange(min=0, max=1000)] , default=0)
    exp_over_7000    = IntegerField('7000 meters', [validators.NumberRange(min=0, max=1000)], default=0)
    exp_over_8000    = IntegerField('8000 meters', [validators.NumberRange(min=0, max=1000)], default=0)
    exp_over_4000    = IntegerField('4000 meters', [validators.NumberRange(min=0, max=1000)], default=0)



@app.route('/', methods=['get', 'post'])
def index():
 
    form_onepeak = OnePeakPredict_Form( request.form )

    return render_template('index.html', form_onepeak=form_onepeak )

@app.route('/overview', methods=['get'])
def overview():
    return render_template('overview.html')

@app.route('/peak', methods=['get'])
def peak():
    return render_template('peak.html')

@app.route('/map', methods=['get'])
def map():
    plot = createUSMap()
    script, div = components(plot)

    return render_template('map.html', the_div=div, the_script=script)


if __name__ == '__main__':
    app.run(port=33507, debug=True)
