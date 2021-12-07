import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import warnings
import sys
import re
import numpy as np
from scipy import stats
import statsmodels.api as sm
#import urllib

from sklearn.metrics import auc
#from sklearn.metrics import roc_auc_score

import base64
import io
import json
import ast
import time

import numpy.polynomial.polynomial as poly
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import tensorflow as tf
import random
import ann_fxns

import pAUCc as PAUCC # See: https://github.com/Big-Life-Lab/partial-AUC-C/tree/main/Python3.7

tf.random.set_seed(1)
np.random.seed(1)
random.seed(1)

#########################################################################################
################################# CONFIG APP ############################################
#########################################################################################


warnings.filterwarnings('ignore')
#pd.set_option('display.max_columns', None)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True


#########################################################################################
################################# LOAD DATA #############################################
#########################################################################################

traces_list = ['all quartiles', 'upper quartile', 'lower quartile',
    'upper and lower quartiles', 'middle quartiles']
#########################################################################################
###################### PROCESS UPLOADED FILE ############################################
#########################################################################################

def get_partial_auc(x, y, mfr):

    x_part = []
    y_part = []
    for i, xi in enumerate(x):
        if xi <= mfr:
            x_part.append(xi)
            y_part.append(y[i])
        
    pAUC = PAUCC.concordant_partial_AUC(x_part, y_part)
    #print(mfr, ' -- ', pAUC, ' -- ', pAUC[3])
    return pAUC

def fix_end_points(fpr, tpr):

    if min(fpr) > 0:
        fpr.append(0)
        tpr.append(1)
    if max(fpr) < 1:
        fpr.append(1)
        tpr.append(0)
    if min(tpr) > 0:
        tpr.append(0)
        fpr.append(1)
    if max(tpr) < 1:
        tpr.append(1)
        fpr.append(0)
    
    return fpr, tpr


def parse_contents(contents, filename, date, downsample):
    #global df
    
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if '.csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                #io.StringIO(decoded.decode('ISO-8859-1')))
                io.StringIO(decoded.decode('utf-8')))
        elif '.xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    dwb_col = df.columns.str.replace('\s+', '_')
    df.columns = dwb_col

    df = df.replace(',','', regex=True)
    c = df.select_dtypes(object).columns
    
    #df[c] = df[c].apply(pd.to_numeric, errors='coerce')
    #df = df.astype(str)
    #df = df.applymap(lambda x: float(x) if x.isnumeric() else x)
    
    df.dropna(how='any', axis=0, inplace=True)
    df = df.sample(frac=0.01*downsample, replace=False, random_state=1)
    
    features = list(df)
    for i in features:
        ls = df[i].tolist()
        if all(isinstance(item, str) for item in ls) is True:
            one_hot = pd.get_dummies(df[i])
            one_hot = one_hot.add_prefix(i + ':')
            
            df = df.drop(labels=[i], axis = 1)
            df = df.join(one_hot)
    
    return df.to_json()


#########################################################################################
################# DASH APP CONTROL FUNCTIONS ############################################
#########################################################################################



def description_card0():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card0",
        children=[
            html.H5("ANN Binary Classifier (beta): Classify your data using artificial neural networks",
                    style={
            'textAlign': 'left',
        }),
            dcc.Markdown("Want to make binary (yes/no) predictions using artifical neural" +
                " networks (ANNs)?" +
                " You've come to the right place." +
                " This one-of-a-kind app allows you to construct and test ANNs using your own data." +
                " You can download the results and even download the ANN." +
                " The app performs some cleaning and preprocessing steps. It will detect your data" +
                " types, handle missing data, and convert categorical data and strings to something" +
                " that ANNs can use. But the app isn't magic, so unwonkify your data before uploading.",

            style={
            'textAlign': 'left',
        }),
        ],
    )


def generate_control_card1():
    
    """
    :return: A Div containing controls for graphs.
    """
    
    return html.Div(
        id="control-card1",
        children=[
            html.P("Large files can take several minutes to clean, preprocess, and analyze. Before uploading a large file, choose a random fraction to ensure that all operates as expected."),
            html.H5("% of data to use"),
            html.Div(id='down-sample-container'),
            dcc.Input(id='downsample',
                type='number',
                value=100,
                min=1, max=100, step=1),
                
            html.Br(),
            html.Br(),
            
            html.P("Upload your data, choose your features," +
                       " customize your neural net, and click 'submit'.",
                        style={
                'textAlign': 'left',
            }),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select File'),
                ]),
                style={
                    'lineHeight': '60px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
            html.Div(id='output-data-upload'),

            html.P("Your file (csv or Excel) should only have" +
                   " columns, rows, and column headers." +
                   " Columns that are categorical or nonnumerical will automatically" +
                   " be one-hot encoded.",
                    ),
            
            html.Br(),
            html.H5("Select a target feature"),
            html.P("The thing you want to predict/classify"),
            dcc.Dropdown(
                id="var-select1",
                options=[{"label": i, "value": i} for i in []],
                multi=False,
                value=None,
                #placeholder="Select a target",
                style={
                    'font-size': "100%",
                    }
            ),
            
            html.Br(),
            html.H5("Select your predictors"),
            html.P("The ANN will automatically remove your target"),
            dcc.Dropdown(
                id="var-select2",
                options=[{"label": i, "value": i} for i in []],
                multi=True,
                value=None,
                #placeholder="Select features",
                style={
                    'font-size': "100%",
                    }
            ),
            
            html.H5("Hidden layers"),
            html.P("Choose 0 to 10 hidden layers. The more you choose, the slower it runs." +
            " Sometimes 0 (no deep learning) is sufficient."),
            html.Div(id='hidden-layers-container'),
            dcc.Input(id='hidden_layers',
                type='number',
                value=0,
                min=0, max=10, step=1),
            
            html.H5("Nodes in the input layer"),
            html.P("Choose 1 to 100 nodes." +
                " Should be less than the number of features. Start small."),
            html.Div(id='inodes-container'),
            dcc.Input(id='inodes',
                type='number',
                value=1,
                min=1, max=100, step=1),
                
            html.H5("Nodes per hidden layer"),
            html.P("Choose 2 to 100 nodes. The more you choose, the slower it runs." +
            " Should be less than 2X the size of the input layer."),
            html.Div(id='nodes-container'),
            dcc.Input(id='nodes',
                type='number',
                value=2,
                min=2, max=100, step=1),
            
            html.H5("k-fold cross validation"),
            html.P("Choose 2 to 20 folds. Each folds gets a model. The best performing model" +
            " gets applied to all the data."),
            html.Div(id='folds-container'),
            dcc.Input(id='folds',
                type='number',
                value=10,
                min=2, max=20, step=1),
            
            html.Br(),
            html.Br(),
            html.H5("Fine tuning"),
            
            html.H6("No. of epochs"),
            html.P("Choose 2 to 10,000 training epochs, i.e., the no. of times that the NN algorithm will work through the entire set of training data."),
            html.Div(id='epochs-container'),
            dcc.Input(id='epochs',
                type='number',
                value=10,
                min=2, max=10000, step=1),
            
            html.H6("Batch size"),
            html.P("Choose a batch size from 10 to 100. This is the number of training examples utilized in each epoch. A batch size of 32 is a common default and means that 32 samples from the training data will be used to estimate the error gradient before the model weights are updated."),
            html.Div(id='batch-size-container'),
            dcc.Input(id='batch_size',
                type='number',
                value=32,
                min=10, max=100, step=1),
                
            html.H6("Early stopping"),
            html.P("An optimization technique used to reduce overfitting without compromising accuracy that stops training when a monitored metric (loss) has stopped improving. Choose the number of epochs (1 to 100) to wait for loss to improve once it initially stops improving."),
            html.Div(id='patience-container'),
            dcc.Input(id='patience',
                type='number',
                value=4,
                min=1, max=100, step=1),
                
            html.H6("Learning rate"),
            html.P("A hyper-parameter used to govern the pace at which the NN algorithm updates the values of a parameter estimate. Choose from values between 0.1 and 0.00001"),
            html.Div(id='learning-rate-container'),
            dcc.Slider(id='learning_rate',
                min=-5,
                max=-1,
                step=None,
                marks={
                    -4: '0.0001',
                    -3: '0.001',
                    -2: '0.01',
                    -1: '0.1',
                },
                value=-2
            ),
            
            html.Br(),
            html.Br(),
            html.Button('submit', id='btn'),

        ],
    )

    

def generate_control_card2():

    return html.Div(
        id="control-card2",
        children=[
            html.Div(id='max-false-rate-container',
                children=[
                    html.H6("Maximum false rate (MFR)"),
                    html.P("Choose a maximum acceptable rate of false positive and false negatives. The app will calculate partial AUC values for partial ROC curves based on the MFR. These partial AUC values are calculated using the partial concordant AUC metric (pAUCc)."),
                    dcc.Slider(id='max_false_rate',
                        min=0.1, max=1.0, step=None, value=0.25,
                        marks={
                                0.15: '0.15',
                                0.25: '0.25',
                                0.35: '0.35',
                                0.45: '0.45',
                                0.55: '0.55',
                                0.65: '0.65',
                                0.75: '0.75',
                                0.85: '0.85',
                                0.95: '0.95',
                                1.0: '1'
                               },
                        ),],
                style={'width': '45%', 'display': 'inline-block', 'margin': '10px',
                        },
                ),
                        
            html.Div(id='traces-container',
                children=[
                    html.H6("Choose the prediction quartiles to plot"),
                    html.P("The predicted values for individual observations are probabilities of being a positive. These are divided into quartiles, with predictions of greatest confidence occurring in the lower and upper quartiles, i.e., likely positives and negatives."),
                    dcc.Dropdown(
                        id="traces1",
                        options=[{"label": i, "value": i} for i in traces_list],
                        multi=True,
                        value=traces_list,
                        ),
                    ],
                style={'width': '45%', 'display': 'inline-block', 'margin': '10px'},
                ),
            ],
    )
    

def generate_control_card3():

    return html.Div(
        id="control-card3",
        children=[
            html.Div(id='pcr-traces-container',
                children=[
                    html.H6("Choose the prediction quartiles to plot"),
                    dcc.Dropdown(
                        id="traces2",
                        options=[{"label": i, "value": i} for i in traces_list],
                        multi=True,
                        value=traces_list,
                        ),
                    ],
                style={'width': '45%', 'display': 'inline-block', 'margin': '10px'},
                ),
            ],
    )

#########################################################################################
################################# DASH APP LAYOUT #######################################
#########################################################################################


app.layout = html.Div([
    
    html.Div(
        id='df1',
        style={'display': 'none'}
        ),
    html.Div(
        id='df2',
        style={'display': 'none'}
        ),
    html.Div(
        id='df3',
        style={'display': 'none'}
    ),
    
    # Banner
    html.Div(
            style={'background-color': '#f9f9f9'},
            id="banner1",
            className="banner",
            children=[html.Img(src=app.get_asset_url("RUSH_full_color.jpg"), 
                               style={'textAlign': 'left'}),
                      html.Img(src=app.get_asset_url("plotly_logo.png"),
                               style={'textAlign': 'right'}),
                      ],
        ),
    
    
    # Under banner
    html.Div(
            id="top-column1",
            className="ten columns",
            children=[description_card0()],
            style={'width': '91.3%',
                #'height': '100px',
                'display': 'inline-block',
                'border-radius': '15px',
                'box-shadow': '1px 1px 1px grey',
                'background-color': '#f0f0f0',
                'padding': '10px',
                'margin-bottom': '10px',
                'margin-left': '30px',
            },
            
        ),
        
    # Left column
    html.Div(
            id="left-column1",
            className="five columns",
            children=[generate_control_card1()],
            style={'width': '24%', 'display': 'inline-block',
                                 'border-radius': '15px',
                                 'box-shadow': '1px 1px 1px grey',
                                 'background-color': '#f0f0f0',
                                 'padding': '10px',
                                 'margin-bottom': '10px',
            },
        ),
    
    # Right column
    html.Div(
            id="right-column1",
            className="eight columns",
            children=[
                html.Div(
                        id="Table1",
                        children=html.Div(id="table1",
                                children=[html.B("Data with predictions." +
                                " The first column is the target feature." +
                                " The second column is the ANN's prediction (a probability)." +
                                " If your data contained categorical data, then the ANN" +
                                " will have used one-hot encoding to transform categories into" +
                                " data the ANN can use (i.e., 1's and 0's)."),
                                            html.Hr(),
                                            dcc.Loading(
                                                id="loading-1",
                                                type="default",
                                                fullscreen=False,
                                                children=
                                                    #dash_table.DataTable(id="table_plot1"),
                                                    html.Div(id="table_plot1")
                                                
                                                ),
                                        ]),
                                        style={'width': '98%', 'height': '550px',
                                               'display': 'inline-block',
                                               'border-radius': '15px',
                                               'box-shadow': '1px 1px 1px grey',
                                               'background-color': '#f0f0f0',
                                               'padding': '10px',
                                               'margin-bottom': '10px',
                                               },
                                    ),

                            
                html.Div(
                    id="ROC_box",
                    children=html.Div(
                            children=[
                                html.H5("Receiver operatering characteristic (ROC) curves"),
                                generate_control_card2(),
                                ]),
                                style={'width': '98%',
                                        #'height': '560px',
                                        'display': 'inline-block',
                                        'border-radius': '15px',
                                        'box-shadow': '1px 1px 1px grey',
                                        'background-color': '#f0f0f0',
                                        'padding': '10px',
                                        'margin-bottom': '10px',
                                        },
                                ),
                html.Div(
                id="ROC_Fig1",
                children=html.Div(id="roc_fig1",
                        children=[dcc.Loading(
                                    id="loading-2",
                                    type="default",
                                    fullscreen=False,
                                    children=dcc.Graph(id="roc_fig1_plot"),
                                    ),
                                ]),
                                style={'width': '47%', 'height': '560px',
                                       'display': 'inline-block',
                                       'border-radius': '15px',
                                       'box-shadow': '1px 1px 1px grey',
                                       'background-color': '#f0f0f0',
                                       'padding': '10px',
                                       'margin-bottom': '10px',
                                       },
                            ),
                            
                            
                html.Div(
                id="ROC_Fig2",
                children=html.Div(id="roc_fig2",
                        children=[dcc.Loading(
                                    id="loading-3",
                                    type="default",
                                    fullscreen=False,
                                    children=dcc.Graph(id="roc_fig2_plot"),
                                    ),
                                ]),
                                style={'width': '47%', 'height': '560px',
                                       'display': 'inline-block',
                                       'border-radius': '15px',
                                       'box-shadow': '1px 1px 1px grey',
                                       'background-color': '#f0f0f0',
                                       'padding': '10px',
                                       'margin-bottom': '10px',
                                       'margin-left': '15px',
                                       },
                            ),
                
                
                html.Div(
                id="PRC_box",
                children=html.Div(
                        children=[
                            html.H5("Precision-recall curves (PRCs)"),
                            #html.B("As with the ROC curves above, predicted values for individual observations have been divided into quartiles."),
                            generate_control_card3()
                            ]),
                            style={'width': '98%',
                                    #'height': '560px',
                                    'display': 'inline-block',
                                    'border-radius': '15px',
                                    'box-shadow': '1px 1px 1px grey',
                                    'background-color': '#f0f0f0',
                                    'padding': '10px',
                                    'margin-bottom': '10px',
                                    },
                            ),
                
                html.Div(
                id="PRC_Fig1",
                children=html.Div(id="prc_fig1",
                        children=[dcc.Loading(
                                    id="loading-4",
                                    type="default",
                                    fullscreen=False,
                                    children=dcc.Graph(id="prc_fig1_plot"),
                                    ),
                                ]),
                                style={'width': '47%', 'height': '560px',
                                       'display': 'inline-block',
                                       'border-radius': '15px',
                                       'box-shadow': '1px 1px 1px grey',
                                       'background-color': '#f0f0f0',
                                       'padding': '10px',
                                       'margin-bottom': '10px',
                                       },
                            ),
                            
                html.Div(
                id="PRC_Fig2",
                children=html.Div(id="prc_fig2",
                        children=[dcc.Loading(
                                    id="loading-5",
                                    type="default",
                                    fullscreen=False,
                                    children=dcc.Graph(id="prc_fig2_plot"),
                                    ),
                                ]),
                                style={'width': '47%', 'height': '560px',
                                       'display': 'inline-block',
                                       'border-radius': '15px',
                                       'box-shadow': '1px 1px 1px grey',
                                       'background-color': '#f0f0f0',
                                       'padding': '10px',
                                       'margin-bottom': '10px',
                                       'margin-left': '15px',
                                       },
                            ),
                
                ]),
    
    
])





#########################################################################################
############################    Call backs   #######################################
#########################################################################################


@app.callback(Output('df1', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified'),
               State('downsample', 'value')])
def update_output1(list_of_contents, list_of_names, list_of_dates, downsample):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d, downsample) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        
        json_df = children[0]
        df = pd.read_json(json_df)
        
        return df.to_json()
    
    

@app.callback([Output('var-select1', 'options'),
               Output('var-select1', 'value')],
              [Input('df1', 'children')])
def update_output2(json_df):
    ls = [None]
    
    if json_df is None:
        pass
    else:
        df = pd.read_json(json_df)
        ls = sorted(list(set(list(df))))
    
    options = [{"label": i, "value": i} for i in ls]
    return options, ls[0]



@app.callback([Output('var-select2', 'options'),
               Output('var-select2', 'value')],
               [Input('df1', 'children')])
def update_output3(json_df):
    ls = [None]
    
    if json_df is None:
        pass
    else:
        df = pd.read_json(json_df)
        ls = sorted(list(set(list(df))))
        
    options = [{"label": i, "value": i} for i in ls]
    return options, ls



@app.callback([Output('df3', 'children'),
               Output('df2', 'children')],
               [Input('btn', 'n_clicks')],
               [State('df1', 'children'),
               State('var-select2', 'value'),
               State('var-select1', 'value'),
               State('hidden_layers', 'value'),
               State('inodes', 'value'),
               State('nodes', 'value'),
               State('folds', 'value'),
               State('epochs', 'value'),
               State('patience', 'value'),
               State('batch_size', 'value'),
               State('learning_rate', 'value'),
               ])
def update_results_df(n_clicks, df, predictors, outcome, hidden_layers, inodes, nodes, folds,
                epochs, patience, batch_size, learning_rate):
    
    learning_rate = 10**learning_rate
    
    if isinstance(outcome, str) == True:
        outcome = [outcome]
    
    if df is None:
        return df, df
        
    json_df, ddf = ann_fxns.get_results(df, predictors, outcome, hidden_layers, inodes, nodes, folds,
                    epochs, patience, batch_size, learning_rate)
    
    df = pd.read_json(json_df)
    ddf = pd.read_json(ddf)
    
    col_names = ['prediction', outcome[0],]
    for name in col_names:
        first_col = df.pop(name)
        df.insert(0, name, first_col)
    
    return df.to_json(), ddf.to_json()


@app.callback(
    Output("table_plot1", 'children'),
    [Input('df3', 'children')],
)
def update_table1(json_df):
    
    if json_df is None:
        df = pd.DataFrame(columns=['Nothing uploaded'])
        df['Nothing uploaded'] = np.nan
    else:
        df = pd.read_json(json_df)
        df = df.round(3)
    
    df = df.sample(n=100, replace=False, random_state=1)
    
    return dash_table.DataTable(
        id="table_plot11",
        columns=[
            {"name": i, "id": i, "deletable": True, "selectable": True} for i in df.columns
        ],
        data=df.to_dict('records'),
        editable=False,
        #filter_action="native",
        sort_action="native",
        sort_mode="multi",
        #column_selectable="single",
        #row_selectable="multi",
        row_deletable=False,
        selected_columns=[],
        selected_rows=[],
        page_action="native",
        #page_action='none',
        page_current= 0,
        page_size= 13,
        style_table={'overflowX': 'scroll',
                     #'overflowY': 'auto',
                     #'height': '415px',
                     },
        style_cell={'textOverflow': 'ellipsis',
                    'minWidth': '120px',
                    'width': '120px',
                    'maxWidth': '120px',
                    },
    ),


@app.callback(Output('roc_fig1_plot', 'figure'),
              [Input('df2', 'children'),
              Input('max_false_rate', 'value'),
              Input('traces1', 'value'),
              ],
              )
def update_results_roc_fig1(df2, mfr, traces_ls):
    
    try:
        df2 = pd.read_json(df2)
    except:
        df2 = None
    
    if df2 is None or len(df2['TPR'].tolist()) == 0:
    
        figure = go.Figure(data=[go.Table(
                header=dict(values=[],
                        fill_color='#b3d1ff',
                        align='left'),
                        ),
                    ],
                )

        figure.update_layout(title_font=dict(size=14,
                      color="rgb(38, 38, 38)", 
                      ),
                      margin=dict(l=10, r=10, b=10, t=0),
                      paper_bgcolor="#f0f0f0",
                      plot_bgcolor="#f0f0f0",
                      height=550,
                      )
        
        return figure
        
    
    df2 = df2[df2['FPR'] >= 0]
    df2 = df2[df2['TPR'] >= 0]
    
    fig_data = []
    clrs = ['#262626', '#ff0000', '#0066ff', '#b300b3', '#00cccc']
    
    for ic, c in enumerate(traces_ls):
    
        if c == 'all quartiles':
            clr = clrs[0]
        elif c == 'upper quartile':
            clr = clrs[1]
        elif c == 'lower quartile':
            clr = clrs[2]
        elif c == 'upper and lower quartiles':
            clr = clrs[3]
        elif c == 'middle quartiles':
            clr = clrs[4]
        
        tdf = df2[df2['certainty_category'] == c]
        #print('\n\n-----------------------------', c)
        #print('----------------------------- shape', tdf.shape[0])
        #print('-----------------------------\n\n',)
        
        tdf.sort_values(by=['FPR', 'TPR'], ascending=True, inplace=True)
        
        fpr = tdf['FPR'].tolist()
        tpr = tdf['TPR'].tolist()
        
        #fpr, tpr = fix_end_points(fpr, tpr)
        AUC = auc(fpr, tpr)
        
        pAUC = get_partial_auc(fpr, tpr, mfr)
        pAUCc = pAUC[3]

        fig_data.append(
                go.Scatter(x = tdf['FPR'], y = tdf['TPR'], mode="lines", marker_color = clr,
                name = c + ': AUC = ' + str(np.round(AUC, 2)) + ', pAUCc = ' + str(np.round(pAUCc,2)),
                text = tdf['TP'] + '<br>' + tdf['FP'] + '<br>' + tdf['TN'] + '<br>' + tdf['FN'] + '<br>' + tdf['threshold'] + '<br>' + tdf['N'],
                opacity = 0.75,
                line=dict(color=clr, width=2),
            ))
        
    fig_data.append(go.Scatter(x=[mfr, mfr], y=[0, 1], mode="lines", type = 'scatter',
                                marker_color = '#b3b3b3', name="Max FPR"))
    
    figure = go.Figure(
        data=fig_data,
        layout=go.Layout(
            xaxis=dict(
                title=dict(
                    text="<b>False positive rate (FPR)</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=14,
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            yaxis=dict(
                title=dict(
                    text="<b>True positive rate (TPR)</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=14,
                        
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            margin=dict(l=60, r=30, b=10, t=40),
            showlegend=True,
            height=550,
            paper_bgcolor="rgb(245, 247, 249)",
            plot_bgcolor="rgb(245, 247, 249)",
        ),
    )
    
    ypos = -0.25
    figure.update_layout(
        legend=dict(
            orientation = "h",
            y = ypos,
            yanchor = "top",
            xanchor="left",
            traceorder = "normal",
            font = dict(
                size = 12,
                color = "rgb(38, 38, 38)"
            ),
            
        )
    )

    return figure
    
    

@app.callback(Output('roc_fig2_plot', 'figure'),
              [Input('df2', 'children'),
              Input('max_false_rate', 'value'),
              Input('traces1', 'value'),
              ],
              )
def update_results_roc_fig2(df2, mfr, traces_ls):
    
    try:
        df2 = pd.read_json(df2)
    except:
        df2 = None
    
    if df2 is None or len(df2['TNR'].tolist()) == 0:
    
        figure = go.Figure(data=[go.Table(
                header=dict(values=[],
                        fill_color='#b3d1ff',
                        align='left'),
                        ),
                    ],
                )

        figure.update_layout(title_font=dict(size=14,
                      color="rgb(38, 38, 38)",
                      ),
                      margin=dict(l=10, r=10, b=10, t=0),
                      paper_bgcolor="#f0f0f0",
                      plot_bgcolor="#f0f0f0",
                      height=550,
                      )
        
        return figure
        
    
    df2 = df2[df2['FNR'] >= 0]
    df2 = df2[df2['TNR'] >= 0]
    
    fig_data = []
    clrs = ['#262626', '#ff0000', '#0066ff', '#b300b3', '#00cccc']
    
    for ic, c in enumerate(traces_ls):

        if c == 'all quartiles':
            clr = clrs[0]
        elif c == 'upper quartile':
            clr = clrs[1]
        elif c == 'lower quartile':
            clr = clrs[2]
        elif c == 'upper and lower quartiles':
            clr = clrs[3]
        elif c == 'middle quartiles':
            clr = clrs[4]
            
        tdf = df2[df2['certainty_category'] == c]
        #print('\n\n-----------------------------', c)
        #print('----------------------------- shape', tdf.shape[0])
        #print('-----------------------------\n\n',)
        
        tdf.sort_values(by=['FNR', 'TNR'], ascending=True, inplace=True)
        
        fnr = tdf['FNR'].tolist()
        tnr = tdf['TNR'].tolist()
        
        #fpr, tpr = fix_end_points(fpr, tpr)
        AUC = auc(fnr, tnr)
        
        pAUC = get_partial_auc(fnr, tnr, mfr)
        pAUCc = pAUC[3]

        fig_data.append(
                go.Scatter(x = tdf['FNR'], y = tdf['TNR'], mode="lines", marker_color = clr,
                name = c + ': AUC = ' + str(np.round(AUC, 2)) + ', pAUCc = ' + str(np.round(pAUCc,2)),
                text = tdf['TP'] + '<br>' + tdf['FP'] + '<br>' + tdf['TN'] + '<br>' + tdf['FN'] + '<br>' + tdf['threshold'] + '<br>' + tdf['N'],
                opacity = 0.75,
                line=dict(color=clr, width=2),
            ))
        
    fig_data.append(go.Scatter(x=[mfr, mfr], y=[0, 1], mode="lines", type = 'scatter',
                                marker_color = '#b3b3b3', name="Max FPR"))
    
    figure = go.Figure(
        data=fig_data,
        layout=go.Layout(
            xaxis=dict(
                title=dict(
                    text="<b>False negative rate (FNR)</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=14,
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            yaxis=dict(
                title=dict(
                    text="<b>True negative rate (TNR)</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=14,
                        
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            margin=dict(l=60, r=30, b=10, t=40),
            showlegend=True,
            height=550,
            paper_bgcolor="rgb(245, 247, 249)",
            plot_bgcolor="rgb(245, 247, 249)",
        ),
    )
    
    ypos = -0.25
    figure.update_layout(
        legend=dict(
            orientation = "h",
            y = ypos,
            yanchor = "top",
            xanchor="left",
            traceorder = "normal",
            font = dict(
                size = 12,
                color = "rgb(38, 38, 38)"
            ),
            
        )
    )

    return figure

    

@app.callback(Output('prc_fig1_plot', 'figure'),
              [Input('df2', 'children'),
              Input('traces2', 'value'),
              ],
              )
def update_results_prc_fig1(df2, traces_ls):
    
    try:
        df2 = pd.read_json(df2)
    except:
        df2 = None
    
    if df2 is None or len(df2['PPV'].tolist()) == 0:
    
        figure = go.Figure(data=[go.Table(
                header=dict(values=[],
                        fill_color='#b3d1ff',
                        align='left'),
                        ),
                    ],
                )

        figure.update_layout(title_font=dict(size=14,
                      color="rgb(38, 38, 38)",
                      ),
                      margin=dict(l=10, r=10, b=10, t=0),
                      paper_bgcolor="#f0f0f0",
                      plot_bgcolor="#f0f0f0",
                      height=550,
                      )
        
        return figure
        
    
    df2 = df2[df2['TPR'] >= 0]
    df2 = df2[df2['PPV'] >= 0]
    
    fig_data = []
    clrs = ['#262626', '#ff0000', '#0066ff', '#b300b3', '#00cccc']
    
    for ic, c in enumerate(traces_ls):

        if c == 'all quartiles':
            clr = clrs[0]
        elif c == 'upper quartile':
            clr = clrs[1]
        elif c == 'lower quartile':
            clr = clrs[2]
        elif c == 'upper and lower quartiles':
            clr = clrs[3]
        elif c == 'middle quartiles':
            clr = clrs[4]
            
        tdf = df2[df2['certainty_category'] == c]
        #print('\n\n-----------------------------', c)
        #print('----------------------------- shape', tdf.shape[0])
        #print('-----------------------------\n\n',)
        
        tdf.sort_values(by=['TPR', 'PPV'], ascending=True, inplace=True)
        
        ppv = tdf['PPV'].tolist()
        tpr = tdf['TPR'].tolist()
        
        #fpr, tpr = fix_end_points(fpr, tpr)
        AUC = auc(tpr, ppv)
        
        #pAUC = get_partial_auc(fnr, tnr, mfr)
        #pAUCc = pAUC[3]

        fig_data.append(
                go.Scatter(x = tdf['TPR'], y = tdf['PPV'], mode="lines", marker_color = clr,
                name = c + ': AUC = ' + str(np.round(AUC, 2)),# + ', pAUCc = ' + str(np.round(pAUCc,2)),
                text = tdf['TP'] + '<br>' + tdf['FP'] + '<br>' + tdf['TN'] + '<br>' + tdf['FN'] + '<br>' + tdf['threshold'] + '<br>' + tdf['N'],
                opacity = 0.75,
                line=dict(color=clr, width=2),
            ))
        
    #fig_data.append(go.Scatter(x=[mfr, mfr], y=[0, 1], mode="lines", type = 'scatter',
    #                            marker_color = '#b3b3b3', name="Max FPR"))
    
    figure = go.Figure(
        data=fig_data,
        layout=go.Layout(
            xaxis=dict(
                title=dict(
                    text="<b>True positive rate (TPR)</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=14,
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            yaxis=dict(
                title=dict(
                    text="<b>Positive predictive value (PPV)</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=14,
                        
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            margin=dict(l=60, r=30, b=10, t=40),
            showlegend=True,
            height=550,
            paper_bgcolor="rgb(245, 247, 249)",
            plot_bgcolor="rgb(245, 247, 249)",
        ),
    )
    
    ypos = -0.25
    figure.update_layout(
        legend=dict(
            orientation = "h",
            y = ypos,
            yanchor = "top",
            xanchor="left",
            traceorder = "normal",
            font = dict(
                size = 12,
                color = "rgb(38, 38, 38)"
            ),
            
        )
    )

    return figure




@app.callback(Output('prc_fig2_plot', 'figure'),
              [Input('df2', 'children'),
              Input('traces2', 'value'),
              ],
              )
def update_results_prc_fig2(df2, traces_ls):
    
    try:
        df2 = pd.read_json(df2)
    except:
        df2 = None
    
    if df2 is None or len(df2['NPV'].tolist()) == 0:
    
        figure = go.Figure(data=[go.Table(
                header=dict(values=[],
                        fill_color='#b3d1ff',
                        align='left'),
                        ),
                    ],
                )

        figure.update_layout(title_font=dict(size=14,
                      color="rgb(38, 38, 38)",
                      ),
                      margin=dict(l=10, r=10, b=10, t=0),
                      paper_bgcolor="#f0f0f0",
                      plot_bgcolor="#f0f0f0",
                      height=550,
                      )
        
        return figure
        
    
    df2 = df2[df2['TNR'] >= 0]
    df2 = df2[df2['NPV'] >= 0]
    
    fig_data = []
    clrs = ['#262626', '#ff0000', '#0066ff', '#b300b3', '#00cccc']
    
    for ic, c in enumerate(traces_ls):

        if c == 'all quartiles':
            clr = clrs[0]
        elif c == 'upper quartile':
            clr = clrs[1]
        elif c == 'lower quartile':
            clr = clrs[2]
        elif c == 'upper and lower quartiles':
            clr = clrs[3]
        elif c == 'middle quartiles':
            clr = clrs[4]
            
        tdf = df2[df2['certainty_category'] == c]
        #print('\n\n-----------------------------', c)
        #print('----------------------------- shape', tdf.shape[0])
        #print('-----------------------------\n\n',)
        
        tdf.sort_values(by=['TNR', 'NPV'], ascending=True, inplace=True)
        
        npv = tdf['NPV'].tolist()
        tnr = tdf['TNR'].tolist()
        
        #fpr, tpr = fix_end_points(fpr, tpr)
        AUC = auc(tnr, npv)
        
        #pAUC = get_partial_auc(fnr, tnr, mfr)
        #pAUCc = pAUC[3]

        fig_data.append(
                go.Scatter(x = tdf['TNR'], y = tdf['NPV'], mode="lines", marker_color = clr,
                name = c + ': AUC = ' + str(np.round(AUC, 2)),# + ', pAUCc = ' + str(np.round(pAUCc,2)),
                text = tdf['TP'] + '<br>' + tdf['FP'] + '<br>' + tdf['TN'] + '<br>' + tdf['FN'] + '<br>' + tdf['threshold'] + '<br>' + tdf['N'],
                opacity = 0.75,
                line=dict(color=clr, width=2),
            ))
        
    #fig_data.append(go.Scatter(x=[mfr, mfr], y=[0, 1], mode="lines", type = 'scatter',
    #                            marker_color = '#b3b3b3', name="Max FPR"))
    
    figure = go.Figure(
        data=fig_data,
        layout=go.Layout(
            xaxis=dict(
                title=dict(
                    text="<b>True negative rate (TNR)</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=14,
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            yaxis=dict(
                title=dict(
                    text="<b>Negative predictive value (NPV)</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=14,
                        
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            margin=dict(l=60, r=30, b=10, t=40),
            showlegend=True,
            height=550,
            paper_bgcolor="rgb(245, 247, 249)",
            plot_bgcolor="rgb(245, 247, 249)",
        ),
    )
    
    ypos = -0.25
    figure.update_layout(
        legend=dict(
            orientation = "h",
            y = ypos,
            yanchor = "top",
            xanchor="left",
            traceorder = "normal",
            font = dict(
                size = 12,
                color = "rgb(38, 38, 38)"
            ),
            
        )
    )

    return figure
#########################################################################################
############################# Run the server ############################################
#########################################################################################


if __name__ == "__main__":
    app.run_server(host='0.0.0.0',debug=False) # modified to run on linux server
