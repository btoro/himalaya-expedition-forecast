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

from data import getPeaks
from models import predict_one_peak
from peaks import visits_over_time

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


df_peaks = getPeaks()

peakList = list(zip(df_peaks.PEAKID, df_peaks.PKNAME))

class OnePeakPredict_Form(Form):
    peak  = SelectField(u'Select Peak', choices=peakList )

    age    = IntegerField('Age', [validators.NumberRange(min=18, max=100), validators.required()])

    useoxygen = BooleanField( 'Plan to use O2?' )

    exp_over_6000    = IntegerField('6000 meters', [validators.NumberRange(min=0, max=1000)] , default=0)
    exp_over_7000    = IntegerField('7000 meters', [validators.NumberRange(min=0, max=1000)], default=0)
    exp_over_8000    = IntegerField('8000 meters', [validators.NumberRange(min=0, max=1000)], default=0)
    exp_over_4000    = IntegerField('4000 meters', [validators.NumberRange(min=0, max=1000)], default=0)



@app.route('/', methods=['get', 'post'])
def index():
    form_onepeak = OnePeakPredict_Form( request.form )

    if request.method == 'GET':
        return render_template('index.html', form_onepeak=form_onepeak )

    elif  request.method == 'POST' and form_onepeak.validate():

        results, prob = predict_one_peak( form_onepeak.peak.data, form_onepeak.age.data, form_onepeak.useoxygen.data, form_onepeak.exp_over_4000.data )
        return render_template('index.html', form_onepeak=form_onepeak, results=results, prob= round( prob[0][1]*100, 0 ) )

    else:
        print('validation failed?')
        print( form_onepeak.errors )
        return render_template('index.html', form_onepeak=form_onepeak )





@app.route('/overview', methods=['get'])
def overview():
    return render_template('overview.html')

@app.route('/peak/<string:peakid>', methods=['get'])
@app.route('/peak', methods=['get'])
def peak( peakid ):

    if not peakid:
        peakid = 'EVER'

    plot = visits_over_time(peakid )
    script, div = components(plot)


    return render_template('peak.html', the_div_1=div, the_script_1=script)

@app.route('/map', methods=['get'])
def map():
    plot = createUSMap()
    script, div = components(plot)

    return render_template('map.html', the_div=div, the_script=script)


if __name__ == '__main__':
    app.run(port=33507, debug=True)
