from flask.ext.wtf import Form
from wtforms import TextField
from wtforms.validators import Required

class PredictForm(Form):
    example = TextField('example', validators=[Required()])

class TrainForm(Form):
    example = TextField('example', validators=[Required()])
    label = TextField('label', validators=[Required()])