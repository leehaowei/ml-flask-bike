from flask import Flask, render_template, request
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
from joblib import load
import plotly.express as px
import plotly.graph_objects as go
import uuid



app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html', href='static/base_pic.svg')
    else:
        text = request.form['text']
        random_string = uuid.uuid4().hex
        path = f"app/static/{random_string}.svg"
        model = load('app/poly_model.joblib')
        np_arr = floats_string_to_np_arr(text)
        make_picture('app/data_by_hour.csv', model, np_arr, path)

        return render_template('index.html', href=path[4:])


def make_picture(training_data_filename, model, new_input_arr, output_file):
    data = pd.read_csv(training_data_filename)
    poly_reg = PolynomialFeatures(degree =12)
    
    x_new = np.array(list(range(24))).reshape(24, 1)
    x_new_poly = poly_reg.fit_transform(x_new)
    preds_ploy = model.predict(x_new_poly)

    fig = px.scatter(data, x="Hour", y="count")
    fig.add_trace(go.Scatter(x=x_new.reshape(24), y=preds_ploy, mode='lines', name='prediction'))
    
    test_poly = poly_reg.fit_transform(new_input_arr)
    new_preds = model.predict(test_poly)
    
    
    fig.add_trace(go.Scatter(x=new_input_arr.reshape(new_input_arr.shape[0]), y=new_preds, name='New Outputs', mode='markers',
                  marker=dict(color='black', size=5, line=dict(color='black', width=2))))
    
    fig.write_image(output_file, width=800, engine='kaleido')
    fig.show()


def floats_string_to_np_arr(floats_str):
    def is_float(s):
        try:
            float(s)
            return True
        except:
            return False
    floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
    return floats.reshape(len(floats), 1)