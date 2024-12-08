#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import warnings
warnings.filterwarnings('ignore')


# In[2]:


myocardial_infarction_complications = pd.read_csv('MI.data')


# In[3]:


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


# ### Overview of the Dataset Structure:
# 
# Columns: 124 (14 are numerical, while 110 are categorical or encoded as objects).
# 
# Rows: 1699 observations.
# 
# ##### Features:
# 
# The features represent clinical and demographic data about patients, their history of myocardial infarction, and observed outcomes.
# 
# ##### Demographics:
# 
# Age: Represents the age of the patient, critical for analyzing age-based risk factors.
# 
# Sex: Indicates the biological sex (1=Male, 0=Female). Gender differences can influence heart disease outcomes.
# 
# ##### Medical History:
# 
# INF_ANAM: Previous history of myocardial infarction. Helps assess recurring cases.
# 
# STENOK_AN: Indicates the presence of angina pectoris.
# 
# FK_STENOK: Functional class of angina, ranging from mild to severe.
# 
# SIM_GIPERT: Simultaneous hypertension (1=Yes, 0=No). Tracks patients with elevated blood pressure.
# 
# DLIT_AG: Duration of arterial hypertension, a risk factor for heart complications.
# 
# ##### Clinical Measurements:
# 
# S_AD_KBRIG and D_AD_KBRIG: Systolic and diastolic blood pressure during hospitalization, 
# indicators of cardiovascular health.
# 
# ALT_BLOOD and AST_BLOOD: Liver enzyme levels, which could reflect organ stress or damage.
# 
# ##### Outcome Variables:
# 
# ZSN: Presence of chronic heart failure (1=Yes, 0=No). Critical for evaluating long-term heart function.
# 
# REC_IM: Recurrence of myocardial infarction during the study period.
# 
# LET_IS: Survival status (0=Survived, 1=Deceased). The main indicator of treatment and condition success.
# 
# ##### Risk Factors and Events:
# 
# RAZRIV: Cardiac rupture following myocardial infarction (1=Yes, 0=No).
# 
# DRESSLER: Dressler syndrome occurrence, a complication of infarction.
# 
# FIBR_PREDS: Pre-existing fibrillation risks.

# In[4]:


myo_infar.replace('?', pd.NA, inplace=True)


# In[5]:


new_column_names = {col: f"col_{idx}" for idx, col in enumerate(myo_infar.columns)}
myo_infar.rename(columns=new_column_names, inplace=True)


# In[6]:


for column in myo_infar.columns:
    if myo_infar[column].dtype in ['float64', 'int64']:
        myo_infar[column].fillna(myo_infar[column].mean(), inplace=True)
    else:
        myo_infar[column].fillna(myo_infar[column].mode()[0], inplace=True)


# In[7]:


myo_infar.columns = column_names


# In[8]:


myo_infar.head()


# In[9]:


myo_infar['AGE'] = pd.to_numeric(myo_infar['AGE'], errors='coerce')


# In[10]:


myo_infar['AGE'] = myo_infar['AGE'].fillna(myo_infar['AGE'].median())


# In[11]:


myo_infar.describe()


# In[12]:


column_names = column_names[:myo_infar.shape[1]]


# In[13]:


myo_infar.columns = column_names


# In[14]:


print(myo_infar.head())


# In[15]:


myo_infar.info()


# In[16]:


numeric_data = myo_infar.select_dtypes(include=[np.number])


# In[17]:


correlation_matrix = myo_infar.corr()

heatmap_fig = px.imshow(
    correlation_matrix,
    labels=dict(color="Correlation"),
    title="Interactive Correlation Heatmap",
    color_continuous_scale="Viridis"
)

heatmap_fig.show()


# In[31]:


## Risk factors

fig = px.parallel_coordinates(
    myo_infar,
    dimensions=['AGE', 'SEX', 'SIM_GIPERT', 'GB', 'ZSN'],
    color='ZSN',  
    color_continuous_scale=px.colors.diverging.Tealrose
)
fig.show()


# In[32]:


## How do age and gender affect post-infarction complications?

# Pairplot using Seaborn
sns.pairplot(
    myo_infar[['AGE', 'SEX', 'ZSN', 'SIM_GIPERT']], 
    hue='SEX',
    diag_kind='kde'
)
plt.show()


# In[33]:


## Age by gender

plt.figure(figsize=(10, 6))
sns.histplot(
    data=myo_infar,
    x='AGE',
    hue='SEX',
    kde=True,
    multiple='stack'
)
plt.title('Age Distribution by Gender')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[34]:


## Complications Across Age Groups

myo_infar['Age_Group'] = pd.cut(myo_infar['AGE'], bins=[0, 40, 60, 80, 100], labels=['<40', '40-60', '60-80', '>80'])


age_group_complications = myo_infar.groupby('Age_Group')['ZSN'].mean().reset_index()


plt.figure(figsize=(10, 6))
sns.lineplot(data=age_group_complications, x='Age_Group', y='ZSN', marker='o')
plt.title('Complication Rate Across Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Complication Rate')
plt.show()


# In[39]:


## visualize the relationships between different risk factors dynamically

fig = px.scatter(
    myo_infar,
    x='AGE',
    y='SIM_GIPERT',
    color='ZSN_A',
    title='Age vs. Hypertension by Complications',
    labels={'AGE': 'Age', 'SIM_GIPERT': 'Hypertension'}
)
fig.show()


# #### Presence of chronic Heart failure (HF) in the anamnesis: 
# 
# Partially ordered attribute: there are two lines of severities: 0<1<2<4, 0<1<3<4. 
# 
# State 4 means simultaneous states 2 and 3 
# 
# 0: there is no chronic heart failure 
# 
# 1: I stage 
# 
# 2: II stage (heart failure due to right ventricular systolic dysfunction) 
# 
# 3: II stage (heart failure due to left ventricular systolic dysfunction) 
# 
# 4: IIB stage (heart failure due to left and right ventricular systolic dysfunction)

# In[28]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
myo_infar['Cluster'] = kmeans.fit_predict(X_scaled)

fig = px.scatter_3d(
    myo_infar, x='AGE', y='SIM_GIPERT', z='GB', color='Cluster',
    title="3D Scatter Plot of Clusters",
    labels={'Cluster': 'Cluster ID'}
)
fig.show()


# In[44]:


## complication types

zsn_a_mapping = {
    0: "No Chronic Heart Failure",
    1: "Stage I",
    2: "Stage II (Right Ventricular Dysfunction)",
    3: "Stage II (Left Ventricular Dysfunction)",
    4: "Stage IIB (Left and Right Ventricular Dysfunction)"
}

myo_infar['ZSN_A'] = pd.to_numeric(myo_infar['ZSN_A'], errors='coerce')

myo_infar['ZSN_A_Description'] = myo_infar['ZSN_A'].map(zsn_a_mapping)

heart_failure_counts = myo_infar['ZSN_A_Description'].value_counts().reset_index()
heart_failure_counts.columns = ['Heart Failure Stage', 'Count']

fig = px.pie(
    heart_failure_counts,
    values='Count',
    names='Heart Failure Stage',
    title='Distribution of Chronic Heart Failure Stages',
    labels={"Heart Failure Stage": "Chronic Heart Failure Stage"}
)
fig.show()



# #### Presence of chronic Heart failure (HF) in the anamnesis: 
# 
# Partially ordered attribute: there are two lines of severities: 0<1<2<4, 0<1<3<4. 
# 
# State 4 means simultaneous states 2 and 3 
# 
# 0: there is no chronic heart failure 
# 
# 1: I stage 
# 
# 2: II stage (heart failure due to right ventricular systolic dysfunction) 
# 
# 3: II stage (heart failure due to left ventricular systolic dysfunction) 
# 
# 4: IIB stage (heart failure due to left and right ventricular systolic dysfunction)

# In[41]:


time_b_s_mapping = {
    1: "Less than 2 hours",
    2: "2-4 hours",
    3: "4-6 hours",
    4: "6-8 hours",
    5: "8-12 hours",
    6: "12-24 hours",
    7: "More than 1 day",
    8: "More than 2 days",
    9: "More than 3 days"
}

myo_infar['TIME_B_S'] = pd.to_numeric(myo_infar['TIME_B_S'], errors='coerce')

myo_infar['TIME_B_S_Description'] = myo_infar['TIME_B_S'].map(time_b_s_mapping)

complication_counts = myo_infar['TIME_B_S_Description'].value_counts().reset_index()
complication_counts.columns = ['Time Elapsed', 'Count']

fig = px.pie(
    complication_counts,
    values='Count',
    names='Time Elapsed',
    title='Time Elapsed from the Beginning of the Attack of CHD to the Hospital',
    labels={"Time Elapsed": "Time Elapsed"}
)
fig.show()


# In[43]:


let_is_mapping = {
    0: "Alive (No Lethal Outcome)",
    1: "Cardiogenic Shock",
    2: "Pulmonary Edema",
    3: "Myocardial Rupture",
    4: "Progress of Congestive Heart Failure",
    5: "Thromboembolism",
    6: "Asystole",
    7: "Ventricular Fibrillation"
}

myo_infar['LET_IS'] = pd.to_numeric(myo_infar['LET_IS'], errors='coerce')

myo_infar['LET_IS_Description'] = myo_infar['LET_IS'].map(let_is_mapping)

lethal_outcome_counts = myo_infar['LET_IS_Description'].value_counts().reset_index()
lethal_outcome_counts.columns = ['Lethal Outcome', 'Count']

fig = px.pie(
    lethal_outcome_counts,
    values='Count',
    names='Lethal Outcome',
    title='Distribution of Lethal Outcomes (Causes)',
    labels={"Lethal Outcome": "Lethal Outcome"}
)
fig.show()


# In[18]:


myo_infar.columns = column_names


# In[25]:


options=[{'label': col, 'value': col} for col in myo_infar.select_dtypes(include='number').columns]
print([col for col in myo_infar.select_dtypes(include='number').columns]) 


# In[24]:


scatter_options = [{'label': col, 'value': col} for col in myo_infar.columns if myo_infar[col].dtype in ['float64', 'int64'] or col == 'AGE']


# In[29]:


import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

app = Dash("Myocardial Infarction Final Project")

app.layout = html.Div([
    html.H1("Myocardial Infarction Complications Analysis", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Filter by Age Range:"),
        dcc.RangeSlider(
            id='age-slider',
            min=int(myo_infar['AGE'].min()),
            max=int(myo_infar['AGE'].max()),
            step=1,
            marks={i: str(i) for i in range(int(myo_infar['AGE'].min()), int(myo_infar['AGE'].max()) + 1, 10)},
            value=[int(myo_infar['AGE'].min()), int(myo_infar['AGE'].max())]
        ),
        html.Label("Filter by Gender:"),
        dcc.RadioItems(
            id='gender-filter',
            options=[
                {'label': 'All', 'value': 'ALL'},
                {'label': 'Male', 'value': 1},
                {'label': 'Female', 'value': 0}
            ],
            value='ALL',
            inline=True
        ),
    ], style={'width': '48%', 'display': 'inline-block'}),

    dcc.Graph(id='scatter-plot'),

    html.Div([
        html.Label("Make a Prediction:"),
        dcc.Input(id='input-age', type='number', placeholder="Enter Age"),
        dcc.Input(id='input-sex', type='number', placeholder="Enter Gender (0 or 1)"),
        dcc.Input(id='input-sim_gipert', type='number', placeholder="Enter Sim_Gipert Value"),
        dcc.Input(id='input-gb', type='number', placeholder="Enter GB Value"),
        html.Button("Predict", id='predict-button'),
        html.Div(id='prediction-output')
    ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    html.Div([
        dcc.Graph(
            figure=fig  
        )
    ])
])


app.layout = html.Div([
    html.H1("Myocardial Infarction Complications Analysis", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Filter by Age Range:"),
        dcc.RangeSlider(
            id='age-slider',
            min=int(myo_infar['AGE'].min()),
            max=int(myo_infar['AGE'].max()),
            step=1,
            marks={i: str(i) for i in range(int(myo_infar['AGE'].min()), int(myo_infar['AGE'].max()) + 1, 10)},
            value=[int(myo_infar['AGE'].min()), int(myo_infar['AGE'].max())]
        ),
        html.Label("Filter by Gender:"),
        dcc.RadioItems(
            id='gender-filter',
            options=[
                {'label': 'All', 'value': 'ALL'},
                {'label': 'Male', 'value': 1},
                {'label': 'Female', 'value': 0}
            ],
            value='ALL',
            inline=True
        ),
    ], style={'width': '48%', 'display': 'inline-block'}),

    
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
     Input('scatter-y-axis', 'value'),
     Input('age-slider', 'value'),
     Input('gender-filter', 'value')]
)
def update_scatter(x_axis, y_axis, age_range, gender_filter):
    filtered_data = myo_infar[
        (myo_infar['AGE'] >= age_range[0]) & (myo_infar['AGE'] <= age_range[1])
    ]
    if gender_filter != 'ALL':
        filtered_data = filtered_data[filtered_data['SEX'] == int(gender_filter)]
    fig = px.scatter(
        filtered_data,
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


# In[20]:


myo_infar.plot(kind='box', subplots=True, layout=(5, 3), sharex=False, sharey=False, figsize=(10, 8))
plt.tight_layout()
plt.show()


# In[ ]:


## Logistic Regression Model to predict risk of complications


# In[27]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

features = ['AGE', 'SEX', 'SIM_GIPERT', 'ZSN', 'GB']  
target = 'ZSN'

X = myo_infar[features]
y = myo_infar[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[ ]:




