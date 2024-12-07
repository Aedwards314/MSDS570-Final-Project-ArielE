#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import warnings
warnings.filterwarnings('ignore')


# In[16]:


myocardial_infarction_complications = pd.read_csv('MI.data')


# In[17]:


column_names = [
    'ID', 'AGE', 'SEX', 'INF_ANAM', 'STENOK_AN', 'FK_STENOK', 'IBS_POST', 'IBS_NASL', 
    'GB', 'SIM_GIPERT', 'DLIT_AG', 'ZSN_A', 'nr11', 'nr01', 'nr02', 'nr03', 'nr04', 'nr07', 
    'nr08', 'np01', 'np04', 'np05', 'np07', 'np08', 'np09', 'np10', 'endocr_01', 'endocr_02', 
    'endocr_03', 'zab_leg_01', 'zab_leg_02', 'zab_leg_03', 'zab_leg_04', 'zab_leg_06', 
    'S_AD_KBRIG', 'D_AD_KBRIG', 'S_AD_ORIT', 'D_AD_ORIT', 'O_L_POST', 'K_SH_POST', 
    'MP_TP_POST', 'SVT_POST', 'GT_POST', 'FIB_G_POST', 'ant_im', 'lat_im', 'inf_im', 
    'post_im', 'IM_PG_P', 'ritm_ecg_p_01', 'ritm_ecg_p_02', 'ritm_ecg_p_04', 'ritm_ecg_p_06', 
    'ritm_ecg_p_07', 'ritm_ecg_p_08', 'n_r_ecg_p_01', 'n_r_ecg_p_02', 'n_r_ecg_p_03', 
    'n_r_ecg_p_04', 'n_r_ecg_p_05', 'n_r_ecg_p_06', 'n_r_ecg_p_08', 'n_r_ecg_p_09', 
    'n_r_ecg_p_10', 'n_p_ecg_p_01', 'n_p_ecg_p_03', 'n_p_ecg_p_04', 'n_p_ecg_p_05', 
    'n_p_ecg_p_06', 'n_p_ecg_p_07', 'n_p_ecg_p_08', 'n_p_ecg_p_09', 'n_p_ecg_p_10', 
    'n_p_ecg_p_11', 'n_p_ecg_p_12', 'fibr_ter_01', 'fibr_ter_02', 'fibr_ter_03', 
    'fibr_ter_05', 'fibr_ter_06', 'fibr_ter_07', 'fibr_ter_08', 'GIPO_K', 'K_BLOOD', 
    'GIPER_Na', 'Na_BLOOD', 'ALT_BLOOD', 'AST_BLOOD', 'KFK_BLOOD', 'L_BLOOD', 'ROE', 
    'TIME_B_S', 'R_AB_1_n', 'R_AB_2_n', 'R_AB_3_n', 'NA_KB', 'NOT_NA_KB', 'LID_KB', 
    'NITR_S', 'NA_R_1_n', 'NA_R_2_n', 'NA_R_3_n', 'NOT_NA_1_n', 'NOT_NA_2_n', 
    'NOT_NA_3_n', 'LID_S_n', 'B_BLOK_S_n', 'ANT_CA_S_n', 'GEPAR_S_n', 'ASP_S_n', 
    'TIKL_S_n', 'TRENT_S_n', 'FIBR_PREDS', 'PREDS_TAH', 'JELUD_TAH', 'FIBR_JELUD', 
    'A_V_BLOK', 'OTEK_LANC', 'RAZRIV', 'DRESSLER', 'ZSN', 'REC_IM', 'P_IM_STEN', 'LET_IS'
]


myo_infar = pd.read_csv('MI.data', header=None, names=column_names)

print(myo_infar.head())


# In[18]:


myo_infar.replace('?', pd.NA, inplace=True)


# In[23]:


new_column_names = {col: f"col_{idx}" for idx, col in enumerate(myo_infar.columns)}
myo_infar.rename(columns=new_column_names, inplace=True)


# In[24]:


for column in myo_infar.columns:
    if myo_infar[column].dtype in ['float64', 'int64']:
        myo_infar[column].fillna(myo_infar[column].mean(), inplace=True)
    else:
        myo_infar[column].fillna(myo_infar[column].mode()[0], inplace=True)


# In[27]:


myo_infar.columns = column_names


# In[31]:


myo_infar.head()


# In[38]:


myo_infar['AGE'] = pd.to_numeric(myo_infar['AGE'], errors='coerce')


# In[39]:


myo_infar['AGE'] = myo_infar['AGE'].fillna(myo_infar['AGE'].median())


# In[40]:


myo_infar.describe()


# In[41]:


column_names = column_names[:myo_infar.shape[1]]


# In[42]:


myo_infar.columns = column_names


# In[43]:


print(myo_infar.head())


# In[44]:


myo_infar.info()


# In[46]:


numeric_data = myo_infar.select_dtypes(include=[np.number])


# In[47]:


correlation_matrix = myo_infar.corr()

heatmap_fig = px.imshow(
    correlation_matrix,
    labels=dict(color="Correlation"),
    title="Interactive Correlation Heatmap",
    color_continuous_scale="Viridis"
)

heatmap_fig.show()


# In[48]:


myo_infar.columns = column_names


# In[49]:


myo_infar.columns = myo_infar.columns.str.strip().str.replace(' ', '_').str.replace('[^A-Za-z0-9_]', '')

# Set a grey background
sns.set(style="darkgrid")

# Create a histogram with Kernel Density Estimate (KDE) for the 'AGE' column
sns.histplot(data=myo_infar, x="AGE", kde=True)
plt.title("Histogram of Age with KDE")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# In[51]:


myo_infar.plot(kind='box', subplots=True, layout=(5, 3), sharex=False, sharey=False, figsize=(10, 8))
plt.tight_layout()
plt.show()


# In[52]:


myo_infar.hist(figsize=(20,15))
plt.show()


# In[53]:


correlation_matrix = myo_infar.corr()


# In[54]:


plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap")
plt.show()


# In[55]:


scatter_options = [{'label': col, 'value': col} for col in myo_infar.columns if myo_infar[col].dtype in ['float64', 'int64'] or col == 'AGE']


# In[56]:


options=[{'label': col, 'value': col} for col in myo_infar.select_dtypes(include='number').columns]
print([col for col in myo_infar_cleaned.select_dtypes(include='number').columns]) 


# In[57]:


import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px



app = Dash("Myocardial Infarction Final Project")


app.layout = html.Div([
    html.H1("Myocardial Infarction Complications Analysis", style={'textAlign': 'center'}),

    
    html.Div([
        html.Label("Select X-axis for Scatterplot:"),
        dcc.Dropdown(
            id='scatter-x-axis',
            options=[{'label': col, 'value': col} for col in myo_infar.select_dtypes(include='number').columns],
            value=myo_infar.select_dtypes(include='number').columns[0]
        ),
        html.Label("Select Y-axis for Scatterplot:"),
        dcc.Dropdown(
            id='scatter-y-axis',
            options=[{'label': col, 'value': col} for col in myo_infar.select_dtypes(include='number').columns],
            value=myo_infar.select_dtypes(include='number').columns[1]
        ),
    ], style={'width': '48%', 'display': 'inline-block'}),

    
    dcc.Graph(id='scatter-plot'),

    
    html.Div([
        html.H3("Correlation Heatmap"),
        dcc.Graph(id='heatmap', figure=px.imshow(
            myo_infar.corr(),
            labels=dict(color="Correlation"),
            title="Interactive Correlation Heatmap",
            color_continuous_scale="Viridis"
        ))
    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    
    html.Div([
        html.Label("Select Column for Box Plot:"),
        dcc.Dropdown(
            id='box-plot-column',
            options=[{'label': col, 'value': col} for col in myo_infar.select_dtypes(include='number').columns],
            value=myo_infar.select_dtypes(include='number').columns[0]
        ),
        dcc.Graph(id='box-plot')
    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
])


@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('scatter-x-axis', 'value'),
     Input('scatter-y-axis', 'value')]
)
def update_scatter(x_axis, y_axis):
    
    fig = px.scatter(
        myo_infar,
        x=x_axis,
        y=y_axis,
        title=f"Scatter Plot of {x_axis} vs {y_axis}",
        labels={x_axis: f"{x_axis} (X-axis)", y_axis: f"{y_axis} (Y-axis)"},  
        opacity=0.7
    )
    return fig


@app.callback(
    Output('box-plot', 'figure'),
    [Input('box-plot-column', 'value')]
)
def update_boxplot(column):
    
    fig = px.box(
        myo_infar,
        y=column,
        title=f"Box Plot: {column}",
        labels={column: f"{column} (Y-axis)"}
    )
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)


# In[ ]:




