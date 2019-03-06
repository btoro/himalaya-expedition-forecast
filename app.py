import os
from flask import Flask, render_template, request, redirect, flash
from flask_wtf import FlaskForm as Form
from wtforms import FieldList, FormField, StringField, TextField, FloatField, IntegerField, BooleanField, TextAreaField, SubmitField, RadioField, SelectField, DateField, validators
from wtforms.fields.html5 import IntegerRangeField

import requests


from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models.sources import ColumnDataSource


from data import getPeaks
from models import predict_one_peak, getConfusionMatrix, getFeatureImportances, getModelSelectionFigure
from models import makePredictions
from peaks import visits_over_time, get_peak_info

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


df_peaks = getPeaks()

peakList = list(zip(df_peaks.PEAKID, df_peaks.PKNAME))



class prediction_from( Form ):

    peak  = SelectField(u'Select Peak', choices=peakList )
    peakHeight = IntegerField('Peak Height (m)' )

    predictType = RadioField('Prediction type', choices=[('1', 'Single Himalayan Peak'), ('2', 'Custom Peak'), ('3', 'All Himalayan peaks')])

    age    = IntegerField('Age', [validators.NumberRange(min=18, max=100), validators.DataRequired()])
    
    useoxygen = BooleanField( 'Plan to use O2?' )

    season  = SelectField(u'Select Season', choices= [('1', 'Spring'), ('2', 'Summer'), ('3', 'Fall'), ('4', 'Winter')] )
    maxPersonalHeight = IntegerField('Personal Max Height (m)', [validators.NumberRange(min=0, max=8840), validators.InputRequired()])

    past_exped = IntegerField('Past Expedition Count', [validators.NumberRange(min=0, max=200), validators.InputRequired()])


@app.route('/', methods=['get', 'post'])
def index():
    form_onepeak = prediction_from( request.form )
    radio = list(form_onepeak.predictType)

    if request.method == 'GET':
        return render_template('index.html', form_onepeak=form_onepeak , radio=radio)

    elif  request.method == 'POST' and form_onepeak.validate():

        kind = int(form_onepeak.predictType.data)

        if kind == 1:
            p = form_onepeak.peak.data
        elif kind == 2:
            p = form_onepeak.peakHeight.data
        elif kind == 3:
            p = ''

        if kind  == 3:
            _, _, p = makePredictions( kind, p, form_onepeak.age.data , form_onepeak.useoxygen.data, int(form_onepeak.season.data) , form_onepeak.maxPersonalHeight.data, form_onepeak.past_exped.data )
            s, d = components(p)
            return render_template('index.html', form_onepeak=form_onepeak , radio=radio, s=s, d=d )

        else:
            results, prob, _ = makePredictions( kind, p, form_onepeak.age.data , form_onepeak.useoxygen.data, int(form_onepeak.season.data) , form_onepeak.maxPersonalHeight.data, form_onepeak.past_exped.data )
        
        
            return render_template('index.html', form_onepeak=form_onepeak , radio=radio, results=results, prob= round( prob[0][1]*100, 0 ))

        # results, prob = predict_one_peak( form_onepeak.peak.data, form_onepeak.age.data, form_onepeak.useoxygen.data, form_onepeak.exp_over_4000.data )
        # return render_template('index.html', form_onepeak=form_onepeak, results=results, prob= round( prob[0][1]*100, 0 ) )

    else:
        print('validation failed?')
        print( form_onepeak.errors )
        return render_template('index.html', form_onepeak=form_onepeak  , radio=radio)





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

    info = get_peak_info( peakid )


    return render_template('peak.html', the_div_1=div, the_script_1=script, info=info)

@app.route('/background', methods=['get'])
def background(  ):
    return render_template('background.html')

@app.route('/mountains', methods=['get'])
def mountains(  ):
    return render_template('mountains.html')



@app.route('/slides', methods=['get'])
def slides(  ):

    plot = visits_over_time( 'EVER' )
    s1_1, d1_1 = components(plot)

    plot = getModelSelectionFigure()
    s3_1, d3_1 = components(plot)

    # plot = getConfusionMatrix()
    # s4_1, d4_1 = components(plot)

    plot = getFeatureImportances()
    s4_2, d4_2 = components(plot)


    

    return render_template('slides.html',   d1_1=d1_1, s1_1=s1_1 ,
                                            d3_1=d3_1, s3_1=s3_1 ,
                                            d4_2=d4_2, s4_2=s4_2  )



if __name__ == '__main__':
    app.run(port=33507, debug=True)
    # app.run(port=33507 )
