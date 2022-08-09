from dash import Dash, Input, Output, callback, dash_table, html, dcc, State
import pandas as pd
import dash_bootstrap_components as dbc


# Constants

T_VAL = 1.96 # t-value for 95% confidence interval
OUTLIER = 50 # threshold for outlier removal determined by looking at distribution of videoboard, broadcast, and social media counts


bc_df = pd.read_excel("./NYJ-Data/Broadcast_Exposures.xlsx") # broadcast exposures
vb_df = pd.read_excel("./NYJ-Data/Videoboard_Features.xlsx") # videoboard features
sm_df = pd.read_csv("./NYJ-Data/AllContentSeriesPosts.csv") # social media posts

# convert Instragram Stories Tap Count to Engagement Rate
for i, row in sm_df.iterrows():
    if row['ServiceType'] == 'Instagram Story':
        new_val = row['TapBackwardCount']/row['TapForwardCount']
        sm_df.at[i, 'EngagementRate'] = new_val

sm_df.dropna(subset= ['EngagementRate'], inplace=True)



bc_df = bc_df[['Asset', 'Audience', 'Media_Value_USD', 'Seen_Hits_Seconds']]
vb_df = vb_df[['Asset', 'Video_Total_Seconds', 'Logo_Average_Clarity', 'Logo_Average_Size', 'Logo_Total_Seconds']]
sm_df = sm_df[['ServiceType','Content_Series_Name', 'EngagementRate', 'PostValue']]


# Min-Max Normalization

df = bc_df.drop('Asset', axis=1)
df_norm = (df - df.min()) / (df.max() - df.min())
bc_df = pd.concat((bc_df['Asset'], df_norm), axis=1)

df = vb_df.drop('Asset', axis=1)
df_norm = (df - df.min()) / (df.max() - df.min())
vb_df = pd.concat((vb_df['Asset'], df_norm), axis=1)

df = sm_df.drop(['Content_Series_Name', 'ServiceType'], axis=1)
df_norm = (df - df.min()) / (df.max() - df.min())
sm_df = pd.concat((sm_df[['Content_Series_Name', 'ServiceType']], df_norm), axis=1)


def VB_Scoring(sec_weight, clarity_weight, size_weight):
    """
        Calculate the score for videoboard assets
        Input:  weights for videoboard variables: Video_Total_Seconds, Logo_Average_Clarity, Logo_Average_Size
        Output: Score for videoboard assets
    """
    vb_vars = vb_df.groupby('Asset').agg(['count', 'mean', 'std']).reset_index()

    # filter out series that don't meet median threshold
    median_count = vb_vars['Video_Total_Seconds'].where(vb_vars['Video_Total_Seconds']['count'] < OUTLIER)['count'].median()

    vid_sec = vb_vars[['Asset', 'Video_Total_Seconds']].where(vb_vars['Video_Total_Seconds']['count'] >= median_count).dropna()
    logo_sec = vb_vars[['Asset', 'Logo_Total_Seconds']].where(vb_vars['Logo_Total_Seconds']['count'] >= median_count).dropna()
    clarity = vb_vars[['Asset', 'Logo_Average_Clarity']].where(vb_vars['Logo_Average_Clarity']['count'] >= median_count).dropna()
    size = vb_vars[['Asset', 'Logo_Average_Size']].where(vb_vars['Logo_Average_Size']['count'] >= median_count).dropna()

    # rename columns
    vid_sec.columns = vid_sec.columns.map(' '.join)
    logo_sec.columns = logo_sec.columns.map(' '.join)
    clarity.columns = clarity.columns.map(' '.join)
    size.columns = size.columns.map(' '.join)

    # recombine into one dataframe
    recombined = pd.merge(logo_sec, clarity, on='Asset ')
    recombined = pd.merge(recombined, size, on='Asset ')
    addSeries = vb_vars[vb_vars.Asset.str.contains('Did You Know?')]
    addSeries.columns = addSeries.columns.map(' '.join)
    recombined = recombined.append(addSeries) ## add this series to connect social media

    Scores = pd.DataFrame(columns=['Asset', 'Low', 'Average', 'High'])

    # calculate 95% confidence interval scores
    for index,row in recombined.iterrows():
        low = sec_weight * (row['Logo_Total_Seconds mean'] - T_VAL*row['Logo_Total_Seconds std']) + clarity_weight * (row['Logo_Average_Clarity mean'] - T_VAL*row['Logo_Average_Clarity std'] )+ size_weight * (row['Logo_Average_Size mean'] - T_VAL*row['Logo_Average_Size std'])
        average = sec_weight* row['Logo_Total_Seconds mean'] + clarity_weight* row['Logo_Average_Clarity mean'] + size_weight * row['Logo_Average_Size mean']
        high = sec_weight * (row['Logo_Total_Seconds mean'] + T_VAL*row['Logo_Total_Seconds std']) + clarity_weight * (row['Logo_Average_Clarity mean'] + T_VAL*row['Logo_Average_Clarity std'] )+ size_weight * (row['Logo_Average_Size mean'] + T_VAL*row['Logo_Average_Size std'])

        Scores.loc[index] = [row['Asset '], low, average, high]
    
    # adjust all scores to get rid of negative values
    min_score = Scores['Low'].min()
    Scores['Low'] = Scores['Low'] + abs(min_score) + abs(Scores['Low'].mean())
    Scores['Average'] = Scores['Average'] + abs(min_score) + abs(Scores['Low'].mean())
    Scores['High'] = Scores['High'] + abs(min_score) + abs(Scores['Low'].mean())

    # sort scores for debugging purposes
    Scores = Scores.sort_values(by='Average', ascending=False)

    return Scores

def BC_Scoring(aud_weight, val_weight, hit_weight):
    """
        Calculate the score for broadcast assets
        Input:  weights for broadcast variables: Audience, Media_Value_USD, Seen_Hits_Seconds
        Output: Score for broadcast assets

    """
    bc_vars = bc_df.groupby('Asset').agg(['count', 'mean', 'std']).reset_index()

    # filter out series that don't meet median threshold
    median_count = bc_vars['Audience'].where(bc_vars['Audience']['count'] < OUTLIER)['count'].median()

    audience = bc_vars[['Asset', 'Audience']].where(bc_vars['Audience']['count'] >= median_count).dropna()
    value = bc_vars[['Asset', 'Media_Value_USD']].where(bc_vars['Media_Value_USD']['count'] >= median_count).dropna()
    hits = bc_vars[['Asset', 'Seen_Hits_Seconds']].where(bc_vars['Seen_Hits_Seconds']['count'] >= median_count).dropna()

    # rename columns
    audience.columns = audience.columns.map(' '.join)
    value.columns = value.columns.map(' '.join)
    hits.columns = hits.columns.map(' '.join)

    # recombine into one dataframe
    recombined = pd.merge(audience, value, on='Asset ')
    recombined = pd.merge(recombined, hits, on='Asset ') 

    Scores = pd.DataFrame(columns=['Asset', 'Low', 'Average', 'High'])

    # calculate 95% confidence interval scores
    for index, row in recombined.iterrows():
        low = aud_weight * (row['Audience mean'] - T_VAL * row['Audience std']) + val_weight * (row['Media_Value_USD mean'] - T_VAL * row['Media_Value_USD std'])+ hit_weight * (row['Seen_Hits_Seconds mean'] - T_VAL * row['Seen_Hits_Seconds std'])
        average = aud_weight *row['Audience mean'] + val_weight * row['Media_Value_USD mean'] + hit_weight * row['Seen_Hits_Seconds mean']
        high = aud_weight * (row['Audience mean'] + T_VAL * row['Audience std']) + val_weight * (row['Media_Value_USD mean'] + T_VAL * row['Media_Value_USD std']) + hit_weight * (row['Seen_Hits_Seconds mean'] + T_VAL * row['Seen_Hits_Seconds std'])
        
        Scores.loc[index] = [row['Asset '], low, average, high]
    
    # adjust all scores to get rid of negative values
    min_score = Scores['Low'].min()
    Scores['Low'] = Scores['Low'] + abs(min_score) + abs(Scores['Low'].mean())
    Scores['Average'] = Scores['Average'] + abs(min_score) + abs(Scores['Low'].mean())
    Scores['High'] = Scores['High'] + abs(min_score) + abs(Scores['Low'].mean())

    # sort scores for debugging purposes
    Scores = Scores.sort_values(by='Average', ascending=False)
    
    return Scores


def SM_Scoring(val_weight, eng_weight):
    """
        Calculate the score for social media assets
        Input:  weights for social media variables: Media_Value_USD, Engagement
        Output: Score for social media assets
    """
    sm_vars = sm_df.groupby('Content_Series_Name').agg(['count', 'mean', 'std']).reset_index()

    # filter out series that don't meet median threshold
    median_count = sm_vars['EngagementRate'].where(sm_vars['EngagementRate']['count'] < OUTLIER)['count'].median()

    eng_rate = sm_vars[['Content_Series_Name', 'EngagementRate']].where(sm_vars['EngagementRate']['count'] >= median_count).dropna()
    post_value = sm_vars[['Content_Series_Name', 'PostValue']].where(sm_vars['PostValue']['count'] >= median_count).dropna()

    # rename columns
    eng_rate.columns = eng_rate.columns.map(' '.join)
    post_value.columns = post_value.columns.map(' '.join)

    # recombine into one dataframe
    recombined = pd.merge(eng_rate, post_value, on='Content_Series_Name ')
    addSeries = sm_vars[sm_vars['Content_Series_Name'].str.contains('Did You Know?')]
    addSeries.columns = addSeries.columns.map(' '.join)
    recombined =recombined.append(addSeries, ignore_index = True) ## add this series to connect videoboard

    Scores = pd.DataFrame(columns=['Asset', 'Low', 'Average', 'High'])

    # calculate 95% confidence interval scores
    for index, row in recombined.iterrows():
        low =  eng_weight * (row['EngagementRate mean'] - T_VAL * row['EngagementRate std']) + val_weight * (row['PostValue mean'] - T_VAL * row['PostValue std'])
        average = eng_weight * (row['EngagementRate mean']) + val_weight * (row['PostValue mean'])
        high = eng_weight * (row['EngagementRate mean'] + T_VAL * row['EngagementRate std']) + val_weight * (row['PostValue mean'] + T_VAL * row['PostValue std'])

        Scores.loc[index] = [row['Content_Series_Name '], low, average, high]
    
    # adjust all scores to get rid of negative values
    min_score = Scores['Low'].min()
    Scores['Low'] = Scores['Low'] + abs(min_score) + abs(Scores['Low'].mean())
    Scores['Average'] = Scores['Average'] + abs(min_score) + abs(Scores['Low'].mean())
    Scores['High'] = Scores['High'] + abs(min_score) + abs(Scores['Low'].mean())

    # sort scores for debugging purposes
    Scores = Scores.sort_values(by='Average', ascending=False)

    return Scores


def BC_Pricing(Scores, initPrice):
    """
        Calculate the price for broadcast assets
        Input:  scores for broadcast assets, price for broadcast assets
        Output: price dataframe for broadcast assets
    """
    # Prices
    Prices = pd.DataFrame(columns=['Asset', 'Bronze ', 'Silver ', 'Gold '])

    # static signage was the most closely related product to our market research
    initScore = Scores[Scores['Asset'] == 'Static Signage']['Average'].values[0]

    # calculate price for each , format it to round to the nearest dollar
    for index, row in Scores.iterrows():
        low = "{:,}".format(int((row['Low']/initScore) * initPrice))
        average = "{:,}".format(int((row['Average']/initScore) * initPrice))
        high = "{:,}".format(int((row['High']/initScore) * initPrice))
        
        Prices.loc[index] = [row['Asset'], low, average, high]

    return Prices, int(Prices[Prices['Asset'] == 'Static Signage']['Silver '].values[0].replace(',',''))


def VB_Pricing(Scores, initPrice):
    """
        Calculate the price for video board assets
        Input:  scores for video board assets, price for video board assets
        Output: price dataframe for video board assets
    """
    # Prices
    Prices = pd.DataFrame(columns=['Asset', 'Bronze ', 'Silver ', 'Gold '])

    initScore = Scores['Average'].mean() # average of the average video board scores

    # calculate price for each tier, format it to round to the nearest dollar
    for index, row in Scores.iterrows():
        low = "{:,}".format(int((row['Low']/initScore) * initPrice))
        average = "{:,}".format(int((row['Average']/initScore) * initPrice))
        high = "{:,}".format(int((row['High']/initScore) * initPrice))
        
        Prices.loc[index] = [row['Asset'], low, average, high]

    return Prices, int(Prices[Prices['Asset'] == 'Did You Know']['Silver '].values[0].replace(',',''))


def SM_Pricing(Scores, initPrice):
    """
        Calculate the price for social media assets
        Input:  scores for social media assets, price for social media assets
        Output: price dataframe for social media assets
    """
    # Prices
    Prices = pd.DataFrame(columns=['Asset', 'Bronze ', 'Silver ', 'Gold '])
    
    initScore = Scores[Scores['Asset']== 'Did You Know']['Average'].values[0] # Did You Know connects social media to video board

    # calculate price for each tier, format it to round to the nearest dollar
    for index, row in Scores.iterrows():
        low = "{:,}".format(int((row['Low']/initScore) * initPrice))
        average = "{:,}".format(int((row['Average']/initScore) * initPrice))
        high = "{:,}".format(int((row['High']/initScore) * initPrice))
        
        Prices.loc[index] = [row['Asset'], low, average, high]

    return Prices


def overall_Pricing(bc_prices, vb_prices, sm_prices):
    """
        Calculate the overall price for each asset
        Input:  prices for broadcast assets, prices for video board assets, prices for social media assets
        Output: price dataframe for each asset category
    """
    # Prices
    
    bc = bc_prices.copy()
    bc.drop(['Asset'], axis=1, inplace=True)
    bc = bc.apply(lambda x: x.str.replace(',','').astype(int), axis=0)

    vb = vb_prices.copy()
    vb.drop(['Asset'], axis=1, inplace=True)
    vb = vb.apply(lambda x: x.str.replace(',','').astype(int), axis=0)
    
    sm = sm_prices.copy()
    sm.drop(['Asset'], axis=1, inplace=True)
    sm = sm.apply(lambda x: x.str.replace(',','').astype(int), axis=0)

    Prices = pd.DataFrame(columns=['Asset Category', 'Bronze ', 'Silver ', 'Gold '], 
                          data=[['Broadcast', "{:,}".format(int(bc['Bronze '].mean())), "{:,}".format(int(bc['Silver '].mean())), "{:,}".format(int(bc['Gold '].mean()))], 
                                ['Video Board', "{:,}".format(int(vb['Bronze '].mean())), "{:,}".format(int(vb['Silver '].mean())),  "{:,}".format(int(vb['Gold '].mean()))],
                                ['Social Media',  "{:,}".format(int(sm['Bronze '].mean())),  "{:,}".format(int(sm['Silver '].mean())),  "{:,}".format(int(sm['Gold '].mean()))]])
    return Prices


def get_prices(initPrice= 525000,
                post_value_weight=1, engagement_rate_weight=.75, # Social Media Weights
                audience_weight=1, value_weight=.75, hit_seconds_weight=.5, # Broadcast Weights
                seconds_weight=.5, clarity_weight=.75, size_weight= 1): # Video Board Weights
    """
        Get the prices for the assets
        Input:  None
        Output: Dataframe with prices for assets
    """

    vb_scores = VB_Scoring(seconds_weight, clarity_weight, size_weight)
    bc_scores = BC_Scoring(audience_weight, value_weight, hit_seconds_weight)
    sm_scores = SM_Scoring(post_value_weight, engagement_rate_weight)

    
    bc_prices, initVBPrice = BC_Pricing(bc_scores, initPrice)
    vb_prices, initSMPrice = VB_Pricing(vb_scores, initVBPrice)
    sm_prices = SM_Pricing(sm_scores, initSMPrice)
    overall_prices = overall_Pricing(bc_prices, vb_prices, sm_prices)

    return [overall_prices, bc_prices, vb_prices, sm_prices]

price_data = get_prices()

colors = {
    'background': '#125740',
    'text': '#000000'
}

all_options = {
    'All Asset Categories': 0,
    'Broadcast Assets': 1,
    'Videoboard Assets': 2,
    'Social Media Assets': 3}

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(style={'backgroundColor': colors['background'], 'textAlign': 'center', 'color': "white"}, children = [
    dcc.Store(id='memory-output', data= {
                                    0: price_data[0].to_dict('records'),
                                    1: price_data[1].to_dict('records'),
                                    2: price_data[2].to_dict('records'),
                                    3: price_data[3].to_dict('records')}),

    html.Img(src=app.get_asset_url('jets.png'), style={'height':'10%', 'width':'10%'}), 

    html.Div([
        html.H1("Jets Rate Card")
    ]), 

    html.Br(),
    
    html.Div([
        dcc.RadioItems(
        id ='breakdown-radio',
        options = [{'label': i, 'value': j} for i, j in all_options.items()],
        value= all_options['All Asset Categories'],
        labelStyle={'display': 'inline', 'margin-right': '20px', 'font-size': '20px'}, # take away display: block to have horizontal buttons
        inputStyle={"margin-right": "10px"}),

        html.Br(),

        html.Div(
        [
            dbc.Button(
                "Open Customization Menu",
                id="collapse-button",
                className="mb-3",
                color="dark",
                n_clicks=0,
            ),
            dbc.Collapse(
                html.Div([
                    html.Div([
                    "Static Signage Price: $",
                    dcc.Input(id="initial-price", type="number", placeholder="525000", debounce=True),]),
                
                    html.Br(),
                    html.Br(),

                    html.Div([
            dbc.Row(
            [
                dbc.Col(html.Div("Broadcast Asset Weights:")),
                dbc.Col(html.Div("Video Board Asset Weights:")),
                dbc.Col(html.Div("Social Media Asset Weights:")),
            ], style={'fontWeight': 'bold', 'fontSize': '20px', 'marginBottom': '20px'},),
            
            dbc.Row(
                [
                    dbc.Col(html.Div(["Audience: ",
                                    dcc.Input(id="audience-weight", type="number", placeholder="1", debounce=True),])),
                    dbc.Col(html.Div(["Seconds Seen: ",
                                    dcc.Input(id="seconds-weight", type="number", placeholder="0.5", debounce=True),])),
                    dbc.Col(html.Div(["Post Value:  ",
                                    dcc.Input(id="post-value-weight", type="number", placeholder="1", debounce=True),])),
                
                ], style={'marginBottom': '20px'}),

            dbc.Row([
                    dbc.Col(html.Div(["Media Value:",
                                        dcc.Input(id="media-value-weight", type="number", placeholder="0.75", debounce=True),])),
                    dbc.Col(html.Div(["Logo Clarity:",
                                        dcc.Input(id="clarity-weight", type="number", placeholder="0.75", debounce=True),])),
                    dbc.Col(html.Div(["Engagement Rate:",
                                            dcc.Input(id="engagement-rate-weight", type="number", placeholder="0.75", debounce=True),])),
                ], style={'marginBottom': '20px'}),
            
            dbc.Row([
                    dbc.Col(html.Div(["Hit Seconds:",
                                            dcc.Input(id="hit-seconds-weight", type="number", placeholder="0.5", debounce=True),])),
                    dbc.Col(html.Div(["Logo Size:",
                                            dcc.Input(id="size-weight", type="number", placeholder="1", debounce=True),])),
                    dbc.Col(html.Div([])),
                ],style={'marginBottom': '20px'} ),

            ]),
                ])
                ,
                id="collapse",
                is_open=False,
            ),
        ]),
    ]),

    html.Br(),
    html.Br(),
    html.Br(),

    html.Div(id='output-table', children=[], style={'display': 'flex', 'justifyContent': 'center'}),

    html.Div([
        html.Br(),
        html.Br(),
        html.Br(), 
    ]), 
]

)




@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output('memory-output', 'data'),
    [Input('initial-price', 'value'), 
    Input('audience-weight', 'value'),
    Input('seconds-weight', 'value'),
    Input('post-value-weight', 'value'),
    Input('media-value-weight', 'value'),
    Input('clarity-weight', 'value'),
    Input('engagement-rate-weight', 'value'),
    Input('hit-seconds-weight', 'value'),
    Input('size-weight', 'value'),])
def update_table_weights(initial_price, 
                    audience_weight, seconds_weight, post_value_weight, 
                    media_value_weight, clarity_weight, engagement_rate_weight, 
                    hit_seconds_weight, size_weight):
    
    staticSignPrice= 525000 # see RateCard.ipynb for justification

    if initial_price == None:
        initial_price = staticSignPrice 

    if audience_weight == None:
        audience_weight = 1
    
    if seconds_weight == None:
        seconds_weight = 0.5
    
    if post_value_weight == None:
        post_value_weight = 1
    
    if media_value_weight == None:
        media_value_weight = 0.75
    
    if clarity_weight == None:
        clarity_weight = 0.75

    if engagement_rate_weight == None:
        engagement_rate_weight = 0.75

    if hit_seconds_weight == None:
        hit_seconds_weight = 0.5
    
    if size_weight == None:
        size_weight = 1

    
    df = get_prices(initPrice=int(initial_price),  post_value_weight=float(post_value_weight),
                    value_weight=float(media_value_weight), clarity_weight=float(clarity_weight),
                    engagement_rate_weight=float(engagement_rate_weight), hit_seconds_weight=float(hit_seconds_weight),
                    size_weight=float(size_weight), audience_weight=float(audience_weight),
                    seconds_weight=float(seconds_weight))  
    return {
            0: df[0].to_dict('records'),
            1: df[1].to_dict('records'),
            2: df[2].to_dict('records'),
            3: df[3].to_dict('records')}


@app.callback(
    Output('output-table', 'children'),
    [Input('breakdown-radio', 'value'),
    Input('memory-output', 'data')])
def set_table_data(value, data):
    df = data[str(value)]

    return dash_table.DataTable(
        id='table',
        data=df,

        style_data = {'border': '1px solid black',
                    'fontFamily': 'Arial'},

        style_header={ 
            'textAlign': 'center',
            'backgroundColor': 'rgb(210, 210, 210)',
            'fontWeight': 'bold',
            'font-size': 20,
            },

        style_cell={
            'textAlign': 'center',
            'color': 'black',
            'fontSize': 15,
        },

         style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(220, 220, 220)',
        }],

        page_action='none',
        style_table={'height': '300px', 'width': '1000px', 'overflowY': 'auto'},
        fixed_rows={'headers': True},
    )


if __name__ == "__main__":
    app.run_server(debug=True)