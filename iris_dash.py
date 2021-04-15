# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 10:25:46 2021

@author: Amanco
"""
import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from sklearn.datasets import load_iris  # para carregar dataset
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
import math
from sklearn.utils import shuffle
torch.manual_seed(1234) 

# Preparing data
iris = load_iris()
data = iris.data[iris.target==1,::2]  # comprimento das sépalas e pétalas, indices 0 e 2
x_train = torch.FloatTensor(data[:,0:1])
y_train = torch.FloatTensor(data[:,1:2])
n_samples = x_train.shape[0]

xt_min = x_train.min()
xt_max = x_train.max()
x_train_n = (x_train - xt_min)/(xt_max - xt_min)

yt_min = y_train.min()
yt_max = y_train.max()
y_train_n = (y_train - yt_min)/(yt_max - yt_min)

x_train_bias = torch.cat([torch.ones(size=(n_samples,1)), x_train_n],dim=1)

# Creating model
model = torch.nn.Linear(2, 1, bias=False)
torch.nn.init.uniform(model.weight.data, -0.1, 0.1)
model.weight.data
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Initialize variables
dataset = torch.cat((x_train_bias, y_train_n),1)
num_epochs = 100
batch_size = 50
n_batch = math.ceil(len(dataset)/batch_size)

# Train loop
weights = []
best_weights = []
loss_list = []
indexes = []
for epoch in range(num_epochs):
    # Shuffle data every epoch
    s_dataset, index = shuffle(dataset, range(len(dataset)))
    batches = [s_dataset[i:i + batch_size] for i in range(0, len(s_dataset), batch_size)]
    indexes.append(index)
    for batch in batches:
        batch_X, batch_y = batch[:,:2], batch[:,2]
        batch_y = torch.unsqueeze(batch_y, 1)
        out = model(batch_X)
        loss = criterion(out, batch_y)
        # Record losses/iter
        loss_list.append(loss)

        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() 
        # Record weights/iter
        weights.append(model.weight.data[0].clone())

        # Calculate analytical solution weights and record it
        x_bias_t = torch.t(batch_X)
        w_opt = (torch.inverse(x_bias_t.mm(batch_X)).mm(x_bias_t)).mm(batch_y)
        best_weights.append(w_opt)
        
w0_train, w1_train = tuple(torch.stack(weights).T)      # Weight vectors from each epoch
x_bias_t = torch.t(x_train_bias)
w_opt = (torch.inverse(x_bias_t.mm(x_train_bias)).mm(x_train_bias.T)).mm(y_train_n)
best_w0, best_w1 = w_opt.T[0]                           # Weights from analitical solution
padding = 1
        
w0 = torch.linspace(best_w0 - padding, best_w0 + padding, 100)
w1 = torch.linspace(best_w1 - padding, best_w1 + padding, 100)

# create w0 and w1 grid
w0_grid,w1_grid = torch.meshgrid(w0,w1)
# calculate J
w = torch.dstack((w0_grid,w1_grid))                   # 100x100x2
x_train = x_train_bias                                # 50x2 => 100x50x2
y_train = y_train_n                                   # 50x1 => 100x50x100
y_pred = torch.matmul(x_train,torch.swapaxes(w,1,2))  # 50x2 * 100x2x100 = 100x100x50
J=((y_pred - y_train)**2).mean(1)



external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')

available_indicators = df['Indicator Name'].unique()

app.layout = html.Div([
    html.Div(id='hidden-div', style={'display':'none'}),
    html.Div(
        dcc.Markdown('''
##### Batch_size: 
        '''),
        style={'display': 'inline-block'}),
    html.Div(
            dcc.Input(
            id='batch',
            placeholder='Batch size',
            type='text',
            value=50,
            debounce=True
            ),
        style={'display': 'inline-block'}
        ),
    
    html.Div([

        html.Div(
            dcc.Graph(id='contour-plot'),
            style={'width': '41%', 'display': 'inline-block'}
        ),
        html.Div(
            dcc.Graph(id='lines-plot'),
            style={'width': '49%', 'display': 'inline-block'}
        )
    ]),

    
    dcc.Markdown('''
##### Iteration number: 
        '''),
    dcc.Slider(
        id='iteration-slider',
        min=0,
        max=len(weights),
        value=0,
        marks={str(i): str(i) for i in range(0,len(weights),20)},
        step=1
    )
])

@app.callback(
    Output('contour-plot', 'figure'),
    Input('iteration-slider', 'value'))
def update_graph1(iteration):
    fig = go.Figure(data =
    go.Contour(
    z=J,
    x=w0,
    y=w1,
    transpose=True,
    ncontours=60,
    showscale=False
    ))
    
    fig.add_scatter(x=w0_train[:iteration], y=w1_train[:iteration], name= 'Loss')
    fig.add_scatter(x=best_weights[iteration][0], y=best_weights[iteration][1],
                    name = 'Batch optima', mode='markers')
    fig.add_scatter(x=[best_w0], y=[best_w1],
                    name = 'Global optima', mode='markers')
    fig.update_layout(
        autosize=False,
        width=800,
        height=800)
    return fig

@app.callback(
    Output('lines-plot', 'figure'),
    Input('iteration-slider', 'value'))
def update_graph2(iteration):
    # print(weights[20])
    # print(len(weights))
    fig2 = go.Figure(data=
    go.Scatter(
    x=x_train_n.squeeze(),
    y=best_w0 + x_train_n.squeeze() * best_w1,
    mode = 'lines+markers',
    marker_symbol='x',
    marker=dict(size=1, color="black"),
    name='Global optima'
    ))
    fig2.add_scatter(x=x_train_n.squeeze(), y= weights[iteration][0] + x_train_n.squeeze() * weights[iteration][1],
        marker=dict(size=1, color="red"),
        name='Current weights')
    fig2.add_scatter(x=x_train_n.squeeze(), y= best_weights[iteration][0] + x_train_n.squeeze() * best_weights[iteration][1],
        marker=dict(size=1, color="blue"),
        name='Batch optima')
    fig2.add_scatter(x=x_train_n.squeeze(), y=y_train_n.squeeze(),
        mode='markers',
        marker_size=20,
        marker_color="lightskyblue",
        name='All data')
    start = (iteration%n_batch) * batch_size
    end = start + batch_size
    fig2.add_scatter(x=x_train_n.squeeze()[indexes[iteration//n_batch]][start:end],
                    y=y_train_n.squeeze()[indexes[iteration//n_batch]][start:end],
        mode='markers',
        marker_size=20,
        marker_color="midnightblue",
        name='Batch data')
    fig2.update_layout(
        autosize=False,
        width=1100,
        height=800)
    return fig2

@app.callback(
    Output('hidden-div','style'),
    Input('batch', 'value'))
def update_training(batch):
    global weights, best_weights, loss_list, w0_train, w1_train, n_batch, batch_size, indexes
    torch.nn.init.uniform(model.weight.data, -0.1, 0.1)
    batch_size = int(batch)
    if batch_size > 50: batch_size=50
    n_batch = math.ceil(len(dataset)/batch_size)
    weights = []
    best_weights = []
    loss_list = []
    indexes = []
    
    for epoch in range(num_epochs):
        # Shuffle data every epoch
        s_dataset, index = shuffle(dataset, range(len(dataset)))
        batches = [s_dataset[i:i + batch_size] for i in range(0, len(s_dataset), batch_size)]
        indexes.append(index)
        for batch in batches:
            batch_X, batch_y = batch[:,:2], batch[:,2]
            batch_y = torch.unsqueeze(batch_y, 1)
            out = model(batch_X)
            loss = criterion(out, batch_y)
            # Record losses/iter
            loss_list.append(loss)
    
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step() 
            # Record weights/iter
            weights.append(model.weight.data[0].clone())
    
            # Calculate analytical solution weights and record it
            x_bias_t = torch.t(batch_X)
            w_opt = (torch.inverse(x_bias_t.mm(batch_X)).mm(x_bias_t)).mm(batch_y)
            best_weights.append(w_opt)
    w0_train, w1_train = tuple(torch.stack(weights).T)      # Weight vectors from each epoch
    return {}


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))