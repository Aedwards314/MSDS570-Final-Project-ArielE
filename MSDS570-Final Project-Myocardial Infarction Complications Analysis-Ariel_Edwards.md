```python
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import warnings
warnings.filterwarnings('ignore')
```


```python
myocardial_infarction_complications = pd.read_csv('MI.data')
```


```python
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
```

       ID AGE  SEX INF_ANAM STENOK_AN FK_STENOK IBS_POST IBS_NASL GB SIM_GIPERT  \
    0   1  77    1        2         1         1        2        ?  3          0   
    1   2  55    1        1         0         0        0        0  0          0   
    2   3  52    1        0         0         0        2        ?  2          0   
    3   4  68    0        0         0         0        2        ?  2          0   
    4   5  60    1        0         0         0        2        ?  3          0   
    
       ... JELUD_TAH FIBR_JELUD A_V_BLOK OTEK_LANC RAZRIV DRESSLER ZSN REC_IM  \
    0  ...         0          0        0         0      0        0   0      0   
    1  ...         0          0        0         0      0        0   0      0   
    2  ...         0          0        0         0      0        0   0      0   
    3  ...         0          0        0         0      0        0   1      0   
    4  ...         0          0        0         0      0        0   0      0   
    
      P_IM_STEN LET_IS  
    0         0      0  
    1         0      0  
    2         0      0  
    3         0      0  
    4         0      0  
    
    [5 rows x 124 columns]


### Overview of the Dataset Structure:

Columns: 124 (14 are numerical, while 110 are categorical or encoded as objects).

Rows: 1699 observations.

##### Features:

The features represent clinical and demographic data about patients, their history of myocardial infarction, and observed outcomes.

##### Demographics:

Age: Represents the age of the patient, critical for analyzing age-based risk factors.

Sex: Indicates the biological sex (1=Male, 0=Female). Gender differences can influence heart disease outcomes.

##### Medical History:

INF_ANAM: Previous history of myocardial infarction. Helps assess recurring cases.

STENOK_AN: Indicates the presence of angina pectoris.

FK_STENOK: Functional class of angina, ranging from mild to severe.

SIM_GIPERT: Simultaneous hypertension (1=Yes, 0=No). Tracks patients with elevated blood pressure.

DLIT_AG: Duration of arterial hypertension, a risk factor for heart complications.

##### Clinical Measurements:

S_AD_KBRIG and D_AD_KBRIG: Systolic and diastolic blood pressure during hospitalization, 
indicators of cardiovascular health.

ALT_BLOOD and AST_BLOOD: Liver enzyme levels, which could reflect organ stress or damage.

##### Outcome Variables:

ZSN: Presence of chronic heart failure (1=Yes, 0=No). Critical for evaluating long-term heart function.

REC_IM: Recurrence of myocardial infarction during the study period.

LET_IS: Survival status (0=Survived, 1=Deceased). The main indicator of treatment and condition success.

##### Risk Factors and Events:

RAZRIV: Cardiac rupture following myocardial infarction (1=Yes, 0=No).

DRESSLER: Dressler syndrome occurrence, a complication of infarction.

FIBR_PREDS: Pre-existing fibrillation risks.


```python
myo_infar.replace('?', pd.NA, inplace=True)
```


```python
new_column_names = {col: f"col_{idx}" for idx, col in enumerate(myo_infar.columns)}
myo_infar.rename(columns=new_column_names, inplace=True)
```


```python
for column in myo_infar.columns:
    if myo_infar[column].dtype in ['float64', 'int64']:
        myo_infar[column].fillna(myo_infar[column].mean(), inplace=True)
    else:
        myo_infar[column].fillna(myo_infar[column].mode()[0], inplace=True)
```


```python
myo_infar.columns = column_names
```


```python
myo_infar.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>AGE</th>
      <th>SEX</th>
      <th>INF_ANAM</th>
      <th>STENOK_AN</th>
      <th>FK_STENOK</th>
      <th>IBS_POST</th>
      <th>IBS_NASL</th>
      <th>GB</th>
      <th>SIM_GIPERT</th>
      <th>...</th>
      <th>JELUD_TAH</th>
      <th>FIBR_JELUD</th>
      <th>A_V_BLOK</th>
      <th>OTEK_LANC</th>
      <th>RAZRIV</th>
      <th>DRESSLER</th>
      <th>ZSN</th>
      <th>REC_IM</th>
      <th>P_IM_STEN</th>
      <th>LET_IS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>77</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>55</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>52</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>68</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 124 columns</p>
</div>




```python
myo_infar['AGE'] = pd.to_numeric(myo_infar['AGE'], errors='coerce')
```


```python
myo_infar['AGE'] = myo_infar['AGE'].fillna(myo_infar['AGE'].median())

```


```python
myo_infar.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>AGE</th>
      <th>SEX</th>
      <th>FIBR_PREDS</th>
      <th>PREDS_TAH</th>
      <th>JELUD_TAH</th>
      <th>FIBR_JELUD</th>
      <th>A_V_BLOK</th>
      <th>OTEK_LANC</th>
      <th>RAZRIV</th>
      <th>DRESSLER</th>
      <th>ZSN</th>
      <th>REC_IM</th>
      <th>P_IM_STEN</th>
      <th>LET_IS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1700.000000</td>
      <td>1700.000000</td>
      <td>1700.000000</td>
      <td>1700.000000</td>
      <td>1700.000000</td>
      <td>1700.000000</td>
      <td>1700.000000</td>
      <td>1700.000000</td>
      <td>1700.000000</td>
      <td>1700.000000</td>
      <td>1700.000000</td>
      <td>1700.000000</td>
      <td>1700.000000</td>
      <td>1700.000000</td>
      <td>1700.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>850.500000</td>
      <td>61.862353</td>
      <td>0.626471</td>
      <td>0.100000</td>
      <td>0.011765</td>
      <td>0.024706</td>
      <td>0.041765</td>
      <td>0.033529</td>
      <td>0.093529</td>
      <td>0.031765</td>
      <td>0.044118</td>
      <td>0.231765</td>
      <td>0.093529</td>
      <td>0.087059</td>
      <td>0.477059</td>
    </tr>
    <tr>
      <th>std</th>
      <td>490.892045</td>
      <td>11.233668</td>
      <td>0.483883</td>
      <td>0.300088</td>
      <td>0.107857</td>
      <td>0.155273</td>
      <td>0.200110</td>
      <td>0.180067</td>
      <td>0.291259</td>
      <td>0.175425</td>
      <td>0.205417</td>
      <td>0.422084</td>
      <td>0.291259</td>
      <td>0.282004</td>
      <td>1.381818</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>26.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>425.750000</td>
      <td>54.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>850.500000</td>
      <td>63.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1275.250000</td>
      <td>70.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1700.000000</td>
      <td>92.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>7.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
column_names = column_names[:myo_infar.shape[1]]

```


```python
myo_infar.columns = column_names

```


```python
print(myo_infar.head())

```

       ID  AGE  SEX INF_ANAM STENOK_AN FK_STENOK IBS_POST IBS_NASL GB SIM_GIPERT  \
    0   1   77    1        2         1         1        2        0  3          0   
    1   2   55    1        1         0         0        0        0  0          0   
    2   3   52    1        0         0         0        2        0  2          0   
    3   4   68    0        0         0         0        2        0  2          0   
    4   5   60    1        0         0         0        2        0  3          0   
    
       ... JELUD_TAH FIBR_JELUD A_V_BLOK OTEK_LANC RAZRIV DRESSLER ZSN REC_IM  \
    0  ...         0          0        0         0      0        0   0      0   
    1  ...         0          0        0         0      0        0   0      0   
    2  ...         0          0        0         0      0        0   0      0   
    3  ...         0          0        0         0      0        0   1      0   
    4  ...         0          0        0         0      0        0   0      0   
    
      P_IM_STEN LET_IS  
    0         0      0  
    1         0      0  
    2         0      0  
    3         0      0  
    4         0      0  
    
    [5 rows x 124 columns]



```python
myo_infar.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1700 entries, 0 to 1699
    Columns: 124 entries, ID to LET_IS
    dtypes: int64(15), object(109)
    memory usage: 1.6+ MB



```python
numeric_data = myo_infar.select_dtypes(include=[np.number])
```


```python
correlation_matrix = myo_infar.corr()

heatmap_fig = px.imshow(
    correlation_matrix,
    labels=dict(color="Correlation"),
    title="Interactive Correlation Heatmap",
    color_continuous_scale="Viridis"
)

heatmap_fig.show()
```


        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Config) {window.MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        define('plotly', function(require, exports, module) {
            /**
* plotly.js v2.29.1
* Copyright 2012-2024, Plotly, Inc.
* All rights reserved.
* Licensed under the MIT license
*/
/*! For license information please see plotly.min.js.LICENSE.txt */
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>




<div>                            <div id="af857351-1a8d-48af-a56e-4b71e9be9ced" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("af857351-1a8d-48af-a56e-4b71e9be9ced")) {                    Plotly.newPlot(                        "af857351-1a8d-48af-a56e-4b71e9be9ced",                        [{"coloraxis":"coloraxis","name":"0","x":["ID","AGE","SEX","FIBR_PREDS","PREDS_TAH","JELUD_TAH","FIBR_JELUD","A_V_BLOK","OTEK_LANC","RAZRIV","DRESSLER","ZSN","REC_IM","P_IM_STEN","LET_IS"],"y":["ID","AGE","SEX","FIBR_PREDS","PREDS_TAH","JELUD_TAH","FIBR_JELUD","A_V_BLOK","OTEK_LANC","RAZRIV","DRESSLER","ZSN","REC_IM","P_IM_STEN","LET_IS"],"z":[[1.0,0.21311330967211778,-0.1167935918124476,0.21037129157546636,0.02387846554368808,0.0964389635073201,0.12705757942299253,0.09245855250580036,0.17797634794526127,0.2677763659329296,0.09703603773781015,-0.09278508037647859,0.22880026901878517,0.12426088448429337,0.5040129434267423],[0.21311330967211778,1.0,-0.39331355067606394,0.1528419506156232,0.01833949438678714,-0.03078036488856052,-0.034620778293876955,0.04272792169624834,0.10449539346940681,0.12437683076533317,-0.07771195007902529,0.1457608075460455,0.09082377543746853,-0.018696027831818375,0.15760711744268444],[-0.1167935918124476,-0.39331355067606394,1.0,-0.1033611325134615,-0.017248131029610363,0.04456027878173519,0.027478466281250873,0.008722011370153082,-0.06518648205265093,-0.0750896926331344,0.03561593383022822,-0.09749043353543468,-0.06518648205265105,-0.016035359303576743,-0.08632589281657399],[0.21037129157546636,0.1528419506156232,-0.1033611325134615,1.0,0.09092412093166355,0.022737061792949185,0.0676298025546897,0.014160085902949548,0.020875676471432722,0.017889035775246528,-0.004774099160262838,0.09572519644447097,0.04107794402443217,-0.05424961142651208,0.062311948800665866],[0.02387846554368808,0.01833949438678714,-0.017248131029610363,0.09092412093166355,1.0,0.052924092315976196,0.004491575209301216,0.009983025461129785,0.021160781776313198,-0.01976251995397946,0.0031253815395899764,0.004715224243155486,0.021160781776313208,-0.014342477929842315,0.03735463302175876],[0.0964389635073201,-0.03078036488856052,0.04456027878173519,0.022737061792949185,0.052924092315976196,1.0,0.13725670688902664,0.07561077370457292,0.026963270075191104,-0.007219707985125624,0.03962038047858589,0.01136856526412309,0.05299254839877148,-0.03570757103136137,0.03556176755237865],[0.12705757942299253,-0.034620778293876955,0.027478466281250873,0.0676298025546897,0.004491575209301216,0.13725670688902664,1.0,0.05912089350653141,-0.006469016840519893,0.02925294770845459,0.01242352932490454,-0.010141221754818474,0.04402376841606334,-0.043609487861413984,0.1726881160353002],[0.09245855250580036,0.04272792169624834,0.008722011370153082,0.014160085902949548,0.009983025461129785,0.07561077370457292,0.05912089350653141,1.0,0.00750592764486384,0.05942805386729237,-0.02410258979252685,-0.00163082408858906,0.007505927644863829,-0.05751798263517892,0.056316707452798334],[0.17797634794526127,0.10449539346940681,-0.06518648205265093,0.020875676471432722,0.021160781776313198,0.026963270075191104,-0.006469016840519893,0.00750592764486384,1.0,-0.035141543721842694,-0.00014467144284025562,0.12519623374311437,0.17434974430554373,-0.07052957359809096,0.013377855736496774],[0.2677763659329296,0.12437683076533317,-0.0750896926331344,0.017889035775246528,-0.01976251995397946,-0.007219707985125624,0.02925294770845459,0.05942805386729237,-0.035141543721842694,1.0,-0.022578690609046815,-0.06768884127066464,-0.023621947823822476,-0.05593285190552899,0.33080044048706464],[0.09703603773781015,-0.07771195007902529,0.03561593383022822,-0.004774099160262838,0.0031253815395899764,0.03962038047858589,0.01242352932490454,-0.02410258979252685,-0.00014467144284025562,-0.022578690609046815,1.0,0.05850063914362154,0.009692986670298167,-0.025700077032239265,-0.053455534805156496],[-0.09278508037647859,0.1457608075460455,-0.09749043353543468,0.09572519644447097,0.004715224243155486,0.01136856526412309,-0.010141221754818474,-0.00163082408858906,0.12519623374311437,-0.06768884127066464,0.05850063914362154,1.0,0.08689442110743784,-0.04599285721870127,-0.03830865936314204],[0.22880026901878517,0.09082377543746853,-0.06518648205265105,0.04107794402443217,0.021160781776313208,0.05299254839877148,0.04402376841606334,0.007505927644863829,0.17434974430554373,-0.023621947823822476,0.009692986670298167,0.08689442110743784,1.0,0.029793391476888947,0.09819919041713347],[0.12426088448429337,-0.018696027831818375,-0.016035359303576743,-0.05424961142651208,-0.014342477929842315,-0.03570757103136137,-0.043609487861413984,-0.05751798263517892,-0.07052957359809096,-0.05593285190552899,-0.025700077032239265,-0.04599285721870127,0.029793391476888947,1.0,-0.09607038306913673],[0.5040129434267423,0.15760711744268444,-0.08632589281657399,0.062311948800665866,0.03735463302175876,0.03556176755237865,0.1726881160353002,0.056316707452798334,0.013377855736496774,0.33080044048706464,-0.053455534805156496,-0.03830865936314204,0.09819919041713347,-0.09607038306913673,1.0]],"type":"heatmap","xaxis":"x","yaxis":"y","hovertemplate":"x: %{x}\u003cbr\u003ey: %{y}\u003cbr\u003eCorrelation: %{z}\u003cextra\u003e\u003c\u002fextra\u003e"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"scaleanchor":"y","constrain":"domain"},"yaxis":{"anchor":"x","domain":[0.0,1.0],"autorange":"reversed","constrain":"domain"},"coloraxis":{"colorbar":{"title":{"text":"Correlation"}},"colorscale":[[0.0,"#440154"],[0.1111111111111111,"#482878"],[0.2222222222222222,"#3e4989"],[0.3333333333333333,"#31688e"],[0.4444444444444444,"#26828e"],[0.5555555555555556,"#1f9e89"],[0.6666666666666666,"#35b779"],[0.7777777777777778,"#6ece58"],[0.8888888888888888,"#b5de2b"],[1.0,"#fde725"]]},"title":{"text":"Interactive Correlation Heatmap"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('af857351-1a8d-48af-a56e-4b71e9be9ced');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
## Risk factors

fig = px.parallel_coordinates(
    myo_infar,
    dimensions=['AGE', 'SEX', 'SIM_GIPERT', 'GB', 'ZSN'],
    color='ZSN',  
    color_continuous_scale=px.colors.diverging.Tealrose
)
fig.show()

```


<div>                            <div id="611c5d21-491b-4cd5-92cc-4c57a0ca9d3a" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("611c5d21-491b-4cd5-92cc-4c57a0ca9d3a")) {                    Plotly.newPlot(                        "611c5d21-491b-4cd5-92cc-4c57a0ca9d3a",                        [{"dimensions":[{"label":"AGE","values":[77,55,52,68,60,64,70,65,60,77,71,50,60,57,55,57,52,52,54,71,70,83,70,63,67,77,82,53,44,50,64,65,72,39,65,78,44,59,59,78,73,49,72,41,63,70,43,72,63,56,67,72,43,65,55,58,71,55,44,60,57,48,45,55,58,70,76,62,60,47,63,59,65,61,61,54,63,55,73,56,59,62,80,46,72,47,63,63,68,41,37,74,37,44,42,49,49,42,56,52,50,38,56,63,63,50,44,43,54,60,64,71,53,51,61,59,64,61,60,65,73,65,49,67,52,63,57,76,61,61,73,50,64,57,72,58,78,58,49,59,55,63,73,45,48,55,65,64,61,71,45,62,70,54,53,66,76,72,69,50,40,88,51,76,38,53,80,65,64,80,44,32,65,71,61,62,65,68,66,86,70,62,50,88,55,79,64,75,64,65,58,62,68,51,38,55,69,57,51,63,71,45,50,53,43,42,76,76,65,61,63,65,64,52,63,50,68,76,62,51,35,65,67,63,44,53,62,57,66,51,53,63,50,52,58,77,50,68,73,60,64,65,51,59,42,45,56,73,62,65,78,76,49,57,51,61,53,47,63,59,52,56,56,64,63,74,38,64,58,49,66,59,52,61,55,63,64,74,51,49,52,56,63,49,49,50,61,37,84,64,44,49,70,54,59,79,70,65,81,77,68,64,50,64,70,65,40,61,45,54,74,66,27,50,65,65,51,63,50,38,45,57,69,52,53,39,41,72,64,27,70,49,52,56,62,62,60,51,57,62,52,55,57,58,79,47,46,59,38,54,79,55,61,58,54,57,71,67,54,72,57,61,52,87,69,56,46,52,65,45,53,55,65,62,58,64,69,69,63,62,74,62,45,83,59,41,51,70,60,59,59,62,43,65,77,45,47,75,73,74,55,67,44,45,79,57,65,63,55,75,65,62,37,61,60,53,64,74,70,83,66,65,62,57,50,48,53,57,52,56,68,68,53,83,43,37,63,63,76,76,68,59,68,53,42,72,63,66,62,62,72,63,42,74,44,50,80,65,64,40,48,59,55,73,65,37,67,54,74,80,48,43,64,63,62,41,70,34,63,50,70,49,42,66,63,35,38,64,63,37,61,32,61,77,65,44,53,56,64,55,75,63,62,69,64,64,62,75,70,67,53,73,68,77,82,76,59,49,62,56,35,66,57,72,61,46,60,34,67,70,55,64,75,56,50,68,76,73,53,65,66,66,77,63,39,70,63,62,71,62,70,46,81,55,37,54,55,60,64,61,71,33,32,52,63,42,74,61,46,60,66,58,57,64,70,68,43,63,34,49,50,83,52,61,62,61,71,57,51,52,46,54,79,61,64,70,44,52,67,65,63,56,52,65,80,70,65,63,65,81,73,62,61,42,69,63,62,47,42,65,55,63,60,72,72,77,63,42,51,75,71,60,57,52,59,67,78,49,62,52,59,70,66,60,69,50,59,50,34,59,84,74,49,52,74,63,64,68,43,63,63,52,48,52,56,61,63,55,44,63,53,83,56,78,62,71,66,40,63,50,76,42,78,75,42,50,69,33,53,54,49,57,60,71,53,49,64,67,72,67,63,56,65,42,44,66,51,62,68,54,37,67,46,54,51,69,72,57,49,77,45,37,73,74,60,52,68,66,55,59,62,43,44,50,62,70,38,78,51,78,66,65,52,75,43,70,83,70,64,51,52,80,67,66,69,71,56,41,57,61,44,38,54,45,36,42,60,64,53,65,47,72,70,67,53,70,61,44,54,83,70,66,43,58,56,52,57,59,72,64,55,80,70,64,57,65,74,63,70,61,67,46,58,45,67,67,66,73,75,55,56,55,74,65,73,56,73,75,52,88,57,69,54,63,56,59,63,55,53,62,71,69,70,60,56,59,52,30,81,60,62,54,55,53,56,43,64,43,37,71,60,58,56,51,71,66,76,56,62,46,58,82,62,83,78,54,73,67,35,40,52,77,64,58,70,40,62,56,82,71,52,52,54,70,58,36,43,63,72,50,52,51,71,54,80,56,53,85,57,62,59,67,70,68,61,51,47,67,67,52,52,52,65,50,60,71,37,54,74,58,69,60,67,58,69,54,67,56,78,63,65,49,80,43,65,64,71,71,73,60,71,52,76,53,61,64,67,75,71,62,65,68,55,65,76,57,66,64,88,79,44,42,63,55,72,58,78,45,64,63,52,43,51,87,57,62,72,61,45,74,76,49,37,60,62,92,71,26,67,52,59,59,70,55,61,57,57,53,62,64,53,67,39,76,64,70,55,52,63,61,70,75,63,64,54,41,68,59,60,50,76,43,53,45,43,66,65,79,58,69,67,59,78,79,56,76,38,55,63,62,61,51,51,60,69,61,45,60,50,67,76,57,41,60,70,66,63,65,49,68,74,63,66,77,48,58,58,74,78,62,52,46,54,62,53,66,92,59,70,68,72,66,67,60,61,53,63,64,59,84,55,62,62,52,59,74,60,69,70,62,64,65,61,67,62,62,62,54,66,79,75,54,52,43,68,57,66,57,75,49,62,72,72,73,58,65,78,70,50,47,73,72,81,48,83,69,50,44,43,70,77,55,56,68,81,55,60,62,62,59,51,66,63,66,61,62,67,50,57,75,68,79,44,62,51,54,53,63,52,63,54,56,57,49,66,35,57,55,60,80,62,81,76,67,54,78,59,69,53,59,81,74,75,61,42,49,50,66,68,52,63,62,60,64,65,55,61,58,75,65,73,68,80,56,46,67,33,43,81,51,62,61,56,67,56,60,65,59,62,63,55,55,62,39,54,80,69,43,58,64,60,52,66,64,65,47,60,65,78,81,65,79,57,80,69,63,54,74,63,87,83,74,63,64,82,78,68,70,62,78,38,57,77,76,66,67,82,48,63,52,79,63,59,68,70,74,57,61,40,63,78,63,63,43,63,69,61,69,54,63,61,49,77,63,85,77,74,52,42,75,75,63,62,71,37,57,52,63,50,74,73,56,60,71,57,66,47,76,59,65,64,74,52,38,75,65,66,53,62,64,70,83,42,70,69,72,74,79,62,70,79,69,55,73,64,57,71,68,83,63,84,54,86,82,63,69,72,66,72,68,75,57,65,75,41,74,51,90,56,77,64,59,68,55,63,70,66,66,70,60,63,62,44,53,74,38,55,64,34,77,47,66,34,71,51,63,59,65,67,71,69,56,58,57,60,72,44,80,78,60,70,55,75,55,57,65,65,53,76,56,80,74,69,54,61,75,56,65,50,74,80,81,52,85,65,75,64,76,70,82,69,53,76,73,62,54,82,68,66,64,65,66,80,71,55,63,61,69,69,64,75,71,62,78,67,79,75,52,65,58,82,65,72,76,83,58,52,75,70,69,77,82,67,77,70,77,63,66,55,64,55,63,46,75,74,67,76,66,65,51,68,90,63,64,63,63,43,65,50,67,69,69,66,62,66,73,63,61,61,67,57,53,84,70,70,77,50,62,73,65,45,76,60,73,60,76,64,62,65,66,83,65,52,60,74,68,70,73,57,75,61,61,65,56,65,65,67,67,70,62,63,82,64,70,76,70,43,54,54,57,62,76,72,73,75,64,79,65,72,65,78,41,77,80,81,63,66,78,56,66,61,62,84,60,64,60,67,53,62,65,55,53,73,62,65,70,80,67,73,51,44,63,80,85,84,80,75,77,57,68,51,58,67,67,67,72,55,64,63,57,75,65,70,63,82,72,69,64,77,59,65,68,50,87,75,68,64,75,66,70,75,76,55,57,78,56,68,66,76,46,83,66,60,70,78,75,60,57,63,75,76,61,61,73,66,88,85,54,77,53,77,62,71,70,77,77,70,55,79,63]},{"label":"SEX","values":[1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,0,1,0,0,1,0,1,1,1,0,1,1,1,0,1,0,1,1,0,0,1,0,0,0,1,0,1,0,0,1,0,0,1,1,0,1,0,1,1,0,1,1,1,1,1,1,1,0,1,0,1,1,1,1,0,1,1,1,1,0,1,0,1,1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,0,1,1,0,0,1,1,1,0,0,0,1,1,1,0,1,1,1,0,1,0,1,1,0,1,0,1,1,1,1,1,1,1,1,1,0,0,1,0,0,0,1,1,1,1,0,1,0,1,0,1,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,1,1,0,1,0,1,0,1,1,1,1,0,0,0,1,1,0,1,0,0,1,1,0,0,1,1,0,1,0,1,0,1,0,1,1,0,1,1,1,1,1,1,0,0,0,1,0,0,1,1,1,1,1,0,1,0,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,0,1,1,1,1,1,0,0,1,1,1,1,0,1,0,0,0,0,1,0,1,1,0,0,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,0,1,1,1,1,0,0,1,1,1,0,1,0,0,0,1,0,0,1,1,0,1,1,1,1,1,1,0,1,0,1,0,0,0,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,1,0,1,0,1,1,0,0,1,1,0,1,0,0,0,1,0,1,1,1,1,1,1,0,1,1,0,0,1,1,0,1,1,0,1,0,1,0,1,1,0,1,1,0,1,0,0,0,0,1,1,0,1,0,0,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,0,1,1,1,1,0,1,0,1,1,0,0,1,1,0,0,1,1,0,1,0,1,1,0,1,1,1,0,0,1,1,1,0,1,1,0,0,1,0,1,0,0,0,1,1,1,1,1,1,1,0,1,0,1,1,1,0,1,0,0,1,1,1,0,0,0,1,1,0,1,0,1,0,1,1,0,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,1,0,0,1,1,1,0,1,1,0,1,0,1,1,1,1,0,1,1,1,1,0,0,1,1,1,0,1,1,1,1,1,1,1,1,1,0,0,1,0,0,0,1,0,1,0,0,1,0,1,1,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0,0,1,1,0,1,1,1,1,1,0,0,1,0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,1,1,1,1,1,0,0,1,1,0,1,1,1,1,1,0,1,1,1,0,0,0,0,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,0,0,1,1,0,1,1,1,1,0,1,1,1,1,0,0,1,1,1,1,0,1,1,1,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1,1,1,1,0,1,1,1,1,1,1,0,1,0,1,0,1,0,1,1,0,1,1,1,0,0,0,1,1,1,1,1,1,0,0,1,1,1,1,1,0,1,1,0,1,1,0,1,1,1,1,0,1,1,0,1,1,1,0,1,1,1,0,0,0,1,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,0,1,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,0,0,1,1,0,1,1,0,1,1,0,0,0,1,1,1,1,1,0,0,0,1,0,1,1,1,1,0,1,1,1,1,0,0,1,1,1,1,1,0,1,1,0,0,0,1,1,1,1,1,0,1,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,1,1,0,1,0,1,1,0,0,0,0,1,1,1,1,1,1,0,1,1,1,0,1,1,0,0,0,0,0,1,0,0,1,1,1,1,1,0,0,1,1,1,1,0,1,0,1,1,0,1,1,1,0,1,0,0,1,1,1,1,1,1,0,1,0,0,1,0,1,0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0,1,1,1,1,1,1,0,1,0,1,1,1,1,0,0,1,1,1,0,0,0,1,0,1,0,1,1,1,1,0,0,1,0,1,1,1,0,0,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,0,1,1,0,1,1,1,1,0,1,0,1,1,1,0,0,0,0,0,1,1,0,1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,0,0,0,1,1,0,0,1,1,1,1,1,1,1,1,0,1,1,1,0,0,0,0,0,0,1,0,1,0,1,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,1,0,1,1,1,1,0,0,0,0,1,1,1,1,0,1,1,0,1,0,0,1,1,1,1,0,1,1,1,1,1,1,1,0,1,0,1,0,1,1,1,1,0,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,0,1,0,1,0,1,0,0,0,0,0,0,1,0,1,0,1,1,0,1,0,0,0,0,1,0,0,1,0,1,1,1,0,1,1,0,1,1,1,0,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,1,0,1,0,1,0,0,1,1,0,1,1,1,1,1,1,0,1,0,1,1,0,0,1,0,1,1,1,0,1,0,1,1,0,1,1,1,1,0,1,0,1,1,1,0,0,0,1,0,1,1,1,0,0,1,0,0,0,1,0,0,1,0,1,0,0,1,1,1,0,1,0,0,1,0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,1,0,1,1,0,1,1,0,1,0,0,1,1,0,1,0,1,0,1,0,0,0,0,1,0,1,1,0,0,1,0,0,0,1,1,0,1,0,0,0,0,1,1,0,1,0,1,1,1,0,0,0,0,1,1,0,1,1,0,0,0,1,0,1,1,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,0,1,0,1,1,1,0,1,0,0,0,0,1,1,1,1,1,0,1,0,0,1,0,1,1,0,0,0,1,0,1,0,1,1,0,1,1,0,0,1,1,0,1,1,0,1,1,0,1,1,0,1,0,0,1,0,1,0,1,1,1,1,0,0,0,1,1,1,0,0,1,0,1,1,1,0,0,0,1,0,0,1,1,1,0,1,0,0,1,0,1,1,1,1,1,0,0,1,0,0,0,1,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,1,1,1,1,1,0,1,1,1,1,0,0,0,0,1,0,1,0,1,1,0,0,1,1,0,0,1,1,1,1,0,0,1,0,1,0,0,1,0,1,1,1,1,0,0,0,0,0,1,0,1]},{"label":"ZSN","values":[0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,1,0,0,0,1,0,1,1,0,1,0,0,1,1,1,1,0,1,0,1,1,0,0,1,0,1,0,0,0,1,1,0,1,1,1,0,0,0,1,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,1,0,1,0,0,0,0,0,0,0,1,1,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,1,0,1,0,1,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,1,0,0,1,1,0,1,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,1,0,0,0,0,1,0,0,1,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,1,0,1,0,0,1,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,0,0,1,1,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,1,0,0,1,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,1,1,0,0,1,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0,1,0,0,1,0,1,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,1,1,0,1,1,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,1,1,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,1,0,1,1,0,1,0,0,0,1,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,1,1,0,0,0,1,1,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,1,0,0,1,0,1,1,0,0,0,0,1,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,1,0,1,1,1,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,1,1,1,1,1,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]}],"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"line":{"color":[0,0,0,1,0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,1,1,1,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,1,0,0,0,1,0,1,1,0,1,0,0,1,1,1,1,0,1,0,1,1,0,0,1,0,1,0,0,0,1,1,0,1,1,1,0,0,0,1,0,0,0,1,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,1,0,1,0,0,0,0,0,0,0,1,1,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,1,0,1,0,1,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,1,1,0,0,0,0,0,0,1,0,0,1,1,0,1,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,1,0,0,0,0,1,0,0,1,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,1,0,1,0,0,1,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,0,0,1,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,1,1,1,0,0,1,0,0,0,0,0,0,0,1,1,0,0,1,1,0,1,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,1,0,0,1,0,0,0,1,1,0,0,0,0,0,1,1,0,0,0,1,1,0,0,1,1,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,1,0,0,0,0,0,1,0,1,0,1,0,1,0,0,1,0,1,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,1,1,0,1,1,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1,1,1,0,1,0,0,0,0,0,0,0,0,1,0,0,1,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,0,0,1,0,1,0,0,0,0,1,0,0,1,0,1,1,0,1,0,0,0,1,0,0,0,1,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,1,1,0,0,0,1,1,0,0,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,1,1,0,0,1,0,1,1,0,0,0,0,1,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,1,0,0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,1,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,1,0,1,1,1,0,0,1,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,1,1,1,1,1,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,0,0,1,1,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],"coloraxis":"coloraxis"},"name":"","type":"parcoords"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"coloraxis":{"colorbar":{"title":{"text":"ZSN"}},"colorscale":[[0.0,"rgb(0, 147, 146)"],[0.16666666666666666,"rgb(114, 170, 161)"],[0.3333333333333333,"rgb(177, 199, 179)"],[0.5,"rgb(241, 234, 200)"],[0.6666666666666666,"rgb(229, 185, 173)"],[0.8333333333333334,"rgb(217, 137, 148)"],[1.0,"rgb(208, 88, 126)"]]},"legend":{"tracegroupgap":0},"margin":{"t":60}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('611c5d21-491b-4cd5-92cc-4c57a0ca9d3a');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
## How do age and gender affect post-infarction complications?

# Pairplot using Seaborn
sns.pairplot(
    myo_infar[['AGE', 'SEX', 'ZSN', 'SIM_GIPERT']], 
    hue='SEX',
    diag_kind='kde'
)
plt.show()

```


    
![png](output_19_0.png)
    



```python
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

```


    
![png](output_20_0.png)
    



```python
## Complications Across Age Groups

myo_infar['Age_Group'] = pd.cut(myo_infar['AGE'], bins=[0, 40, 60, 80, 100], labels=['<40', '40-60', '60-80', '>80'])


age_group_complications = myo_infar.groupby('Age_Group')['ZSN'].mean().reset_index()


plt.figure(figsize=(10, 6))
sns.lineplot(data=age_group_complications, x='Age_Group', y='ZSN', marker='o')
plt.title('Complication Rate Across Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Complication Rate')
plt.show()

```


    
![png](output_21_0.png)
    



```python
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

```


<div>                            <div id="87f74787-95c8-4127-b24b-60c842feba78" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("87f74787-95c8-4127-b24b-60c842feba78")) {                    Plotly.newPlot(                        "87f74787-95c8-4127-b24b-60c842feba78",                        [{"hovertemplate":"ZSN_A=0\u003cbr\u003eAge=%{x}\u003cbr\u003eHypertension=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"0","marker":{"color":"#636efa","symbol":"circle"},"mode":"markers","name":"0","showlegend":true,"x":[77,55,52,60,64,65,60,71,50,60,57,55,57,52,54,71,70,63,77,53,44,50,64,65,72,39,65,78,59,72,41,63,70,43,72,63,43,58,71,55,44,60,57,48,45,55,62,60,47,59,61,61,54,55,80,46,63,63,68,41,37,74,37,44,42,49,49,42,56,52,50,38,56,63,63,50,44,43,54,60,64,71,53,51,61,59,61,60,65,65,49,67,52,63,57,76,61,61,73,50,64,57,72,58,78,58,49,59,55,63,73,45,48,55,65,61,71,45,70,54,53,66,76,72,69,50,40,88,51,76,38,53,65,64,80,44,32,65,71,61,62,65,68,66,86,70,62,50,88,55,79,64,75,64,65,58,62,68,51,38,55,57,51,63,50,43,61,65,64,52,50,68,76,62,51,35,63,57,66,51,53,63,50,52,58,77,50,68,73,60,64,65,51,59,42,45,56,78,49,57,51,53,47,63,59,52,56,56,64,63,74,38,64,58,49,66,59,52,61,55,63,64,49,52,56,63,49,49,50,61,37,44,49,70,54,79,70,65,77,68,64,50,64,70,65,40,61,54,74,66,27,50,65,50,38,45,57,53,72,64,27,70,52,56,62,62,60,51,57,55,57,58,79,47,46,38,54,61,54,67,54,72,57,61,52,56,46,52,65,45,55,62,58,64,69,63,62,74,62,45,83,41,51,60,59,59,62,43,65,77,45,47,73,74,55,67,44,45,79,57,65,63,55,75,65,62,37,61,60,53,64,74,70,66,65,62,57,50,48,53,57,52,56,68,68,53,83,43,37,63,63,76,76,68,59,68,53,42,72,63,66,62,62,72,63,42,74,44,50,80,65,64,40,48,59,55,73,65,37,67,54,74,80,48,43,64,63,62,41,70,34,63,50,70,49,42,66,63,35,38,63,37,61,32,61,77,65,44,53,56,64,55,63,62,69,64,64,62,70,67,53,73,82,76,59,49,62,56,35,66,72,61,46,60,34,70,55,75,56,50,68,73,53,65,66,66,77,63,39,70,63,62,71,62,70,46,81,55,37,54,55,60,64,61,71,33,32,52,63,42,61,46,60,66,58,57,64,70,68,43,63,34,49,50,83,52,61,62,71,57,51,52,46,54,61,64,70,44,52,67,65,63,56,52,65,80,70,65,65,81,73,62,61,42,69,63,47,42,65,55,63,60,72,72,77,63,42,51,75,71,60,57,52,59,67,78,49,62,52,59,70,66,60,69,50,50,34,59,84,49,74,63,64,68,43,63,63,52,48,52,56,61,63,55,44,63,53,83,56,78,62,71,66,40,63,50,76,42,78,75,42,50,69,33,53,54,49,57,60,71,53,49,64,67,67,63,56,65,42,44,66,51,62,68,37,67,46,54,51,69,72,49,77,45,37,73,74,60,52,68,66,55,59,62,43,44,50,62,38,78,51,78,66,65,52,75,43,70,83,70,64,51,52,80,67,66,69,71,56,41,57,61,44,38,54,45,36,42,60,64,53,65,47,72,70,67,53,70,61,44,54,83,70,66,43,58,56,52,57,59,72,64,55,80,70,64,57,65,74,63,70,61,67,46,58,45,67,67,66,73,75,55,56,55,74,65,73,56,73,52,88,57,69,54,63,56,63,55,53,62,71,69,60,56,59,52,30,81,60,62,54,55,53,56,43,64,43,37,71,60,58,56,51,71,66,76,56,62,46,82,62,83,78,54,73,67,35,40,52,77,64,58,70,40,62,56,82,71,52,52,54,70,58,36,43,63,72,50,52,51,54,80,56,53,85,57,62,59,67,70,68,61,51,47,67,67,52,52,52,50,60,71,37,54,74,69,60,67,58,69,54,67,56,78,63,65,49,80,43,65,64,71,71,60,71,52,76,53,61,64,67,71,62,65,68,55,65,57,66,64,88,79,44,42,63,55,72,58,45,63,52,43,51,57,62,72,61,45,74,76,49,37,60,62,92,71,26,67,52,59,59,55,61,57,57,53,62,64,53,67,39,76,70,55,52,63,61,70,75,63,64,54,41,68,59,60,50,76,43,53,45,43,66,65,79,58,69,67,78,79,76,38,55,63,62,61,51,51,60,69,61,45,60,50,67,76,57,41,60,70,66,63,65,49,68,74,63,66,77,48,58,58,74,78,62,52,46,54,62,53,92,59,70,68,72,66,67,60,53,63,64,59,84,55,62,62,52,59,74,60,69,70,62,64,65,61,67,62,62,62,54,66,79,75,54,52,43,68,57,66,57,75,49,62,72,72,73,58,65,78,70,50,47,72,81,48,83,69,50,44,43,70,77,55,56,68,81,55,60,62,62,59,51,66,63,66,61,62,67,50,57,75,68,79,44,62,51,54,53,63,52,63,54,56,57,49,66,35,57,55,60,80,62,81,76,67,54,78,59,69,53,59,81,74,75,61,42,49,50,66,68,52,63,62,60,64,65,55,61,58,75,65,73,68,80,56,46,67,33,43,81,51,62,61,56,67,56,60,65,59,62,63,55,55,62,39,54,80,69,43,58,64,60,52,66,64,65,47,60,65,78,81,65,79,57,80,69,63,54,74,63,87,83,74,63,64,82,78,68,70,62,78,38,57,77,76,66,67,82,48,63,52,79,63,59,68,70,74,57,61,40,63,78,63,63,43,63,69,61,54,63,61,49,77,63,85,77,74,52,42,75,75,63,62,71,37,57,52,63,50,74,73,56,60,71,57,66,47,76,59,65,64,74,52,38,75,65,66,53,62,64,70,83,42,70,69,72,74,79,62,79,69,55,73,64,57,71,68,83,63,84,54,86,82,63,69,72,66,72,68,75,57,65,75,41,74,51,90,56,77,64,59,68,63,70,66,66,70,60,63,62,44,53,74,38,55,64,34,77,47,66,34,71,51,63,59,65,67,71,69,56,58,57,60,72,44,78,60,70,55,75,55,57,65,53,76,56,80,74,69,61,75,56,65,50,74,80,81,52,85,65,64,70,82,53,73,62,54,82,68,66,65,66,80,71,55,63,61,69,64,75,71,62,78,67,75,52,65,58,82,65,72,76,83,58,52,75,70,69,67,77,70,63,64,55,63,46,75,74,67,76,66,65,51,68,90,63,64,63,63,43,65,50,67,69,69,66,62,66,63,61,67,57,84,70,70,50,62,65,45,76,60,73,60,76,64,62,65,66,83,65,52,60,74,68,70,73,57,75,61,61,65,56,65,65,67,67,70,62,63,82,64,70,76,70,43,54,54,57,62,76,72,73,75,64,79,65,41,77,80,81,63,66,78,56,66,61,62,84,60,64,67,62,65,55,53,73,62,65,80,67,73,44,63,80,85,84,80,75,77,57,68,51,58,67,67,67,64,63,57,75,70,63,82,64,59,65,68,87,75,68,66,75,78,56,68,66,46,83,66,60,78,75,60,57,63,76,73,66,88,85,54,53,77,62,71,70,77,77,70,55,79],"xaxis":"x","y":["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","1","1","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","1","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","1","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","1","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0"],"yaxis":"y","type":"scattergl"},{"hovertemplate":"ZSN_A=1\u003cbr\u003eAge=%{x}\u003cbr\u003eHypertension=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"1","marker":{"color":"#EF553B","symbol":"circle"},"mode":"markers","name":"1","showlegend":true,"x":[68,70,77,52,83,70,67,82,44,59,78,73,49,56,67,72,65,55,58,70,76,63,65,63,73,56,59,62,72,47,64,73,64,80,69,71,45,53,42,76,76,65,63,63,65,67,44,53,62,73,62,65,76,61,74,51,84,64,59,81,45,65,51,63,69,52,39,41,49,62,52,59,79,55,58,57,71,87,69,53,65,69,59,70,75,74,72,75,58,65,58,75,64,87,64,59,56,66,64,77,82,77,55],"xaxis":"x","y":["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0"],"yaxis":"y","type":"scattergl"},{"hovertemplate":"ZSN_A=2\u003cbr\u003eAge=%{x}\u003cbr\u003eHypertension=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"2","marker":{"color":"#00cc96","symbol":"circle"},"mode":"markers","name":"2","showlegend":true,"x":[62,64,74,73,76,78,70,61,69,70,55,69,79,73,61,77,78,53,51,72,55,72,55,57,76,61,77],"xaxis":"x","y":["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0"],"yaxis":"y","type":"scattergl"},{"hovertemplate":"ZSN_A=3\u003cbr\u003eAge=%{x}\u003cbr\u003eHypertension=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"3","marker":{"color":"#ab63fa","symbol":"circle"},"mode":"markers","name":"3","showlegend":true,"x":[83,75,68,77,57,61,79,63,59,54,57,70,59,70,71,73,80,65,54,76,76,66,53,65,72,60,70,64,75],"xaxis":"x","y":["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","1","0","0"],"yaxis":"y","type":"scattergl"},{"hovertemplate":"ZSN_A=4\u003cbr\u003eAge=%{x}\u003cbr\u003eHypertension=%{y}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"4","marker":{"color":"#FFA15A","symbol":"circle"},"mode":"markers","name":"4","showlegend":true,"x":[75,67,64,76,62,52,75,69,73,65,69,77,50,75,70,76,70,61,63],"xaxis":"x","y":["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0"],"yaxis":"y","type":"scattergl"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"Age"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"Hypertension"}},"legend":{"title":{"text":"ZSN_A"},"tracegroupgap":0},"title":{"text":"Age vs. Hypertension by Complications"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('87f74787-95c8-4127-b24b-60c842feba78');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


#### Presence of chronic Heart failure (HF) in the anamnesis: 

Partially ordered attribute: there are two lines of severities: 0<1<2<4, 0<1<3<4. 

State 4 means simultaneous states 2 and 3 

0: there is no chronic heart failure 

1: I stage 

2: II stage (heart failure due to right ventricular systolic dysfunction) 

3: II stage (heart failure due to left ventricular systolic dysfunction) 

4: IIB stage (heart failure due to left and right ventricular systolic dysfunction)


```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
myo_infar['Cluster'] = kmeans.fit_predict(X_scaled)

fig = px.scatter_3d(
    myo_infar, x='AGE', y='SIM_GIPERT', z='GB', color='Cluster',
    title="3D Scatter Plot of Clusters",
    labels={'Cluster': 'Cluster ID'}
)
fig.show()
```


<div>                            <div id="0d85d0f9-98e8-4243-a91b-4fe7706adb5b" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("0d85d0f9-98e8-4243-a91b-4fe7706adb5b")) {                    Plotly.newPlot(                        "0d85d0f9-98e8-4243-a91b-4fe7706adb5b",                        [{"hovertemplate":"AGE=%{x}\u003cbr\u003eSIM_GIPERT=%{y}\u003cbr\u003eGB=%{z}\u003cbr\u003eCluster ID=%{marker.color}\u003cextra\u003e\u003c\u002fextra\u003e","legendgroup":"","marker":{"color":[1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,0,1,0,0,1,0,2,1,1,0,1,1,1,0,1,0,2,1,0,0,1,0,0,0,1,0,1,0,0,1,0,0,1,1,0,1,0,1,1,0,1,1,1,1,1,1,1,0,1,0,1,1,1,1,0,1,1,1,1,0,1,2,1,1,1,0,1,1,1,1,0,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,2,1,1,1,1,0,0,0,1,0,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,2,1,1,0,0,1,1,1,0,0,0,1,1,1,0,1,1,1,0,1,0,1,1,0,1,0,1,1,1,1,1,1,1,1,1,0,0,1,0,0,0,1,1,1,1,0,1,0,1,0,1,1,1,1,0,1,1,1,0,1,1,1,1,0,1,2,1,1,0,0,1,0,1,0,1,1,1,1,0,0,0,1,1,0,1,0,0,1,1,0,0,1,1,0,1,0,1,0,1,0,1,1,0,1,1,1,1,1,1,0,0,0,1,0,0,1,1,1,1,1,0,1,0,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1,0,1,1,1,1,1,0,0,1,1,1,1,0,1,0,0,0,0,1,0,1,1,2,0,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,2,1,2,1,1,1,1,0,0,1,1,1,0,1,0,0,0,1,0,0,1,1,0,1,1,1,1,1,1,0,1,0,1,0,0,0,1,1,1,1,1,1,1,1,1,0,1,1,1,0,1,1,0,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,0,1,0,1,1,1,2,0,1,0,1,1,0,0,1,1,0,1,0,0,0,1,0,1,1,1,1,1,1,0,1,2,0,0,1,1,0,1,1,0,1,0,1,0,1,2,2,1,1,0,1,0,0,0,0,1,1,2,1,0,0,1,1,1,1,1,0,0,1,2,1,1,0,1,1,1,0,1,1,1,1,0,2,0,1,1,0,0,1,1,0,0,1,1,1,1,0,1,1,0,1,1,1,0,0,1,1,1,0,1,1,0,0,1,0,1,0,0,0,1,1,1,1,1,1,1,0,1,2,1,1,1,2,1,0,0,1,1,1,0,0,0,1,1,0,1,0,1,0,1,1,0,1,0,1,2,1,1,2,1,1,0,1,1,1,1,1,1,0,0,1,1,1,0,1,1,0,1,0,1,1,1,1,0,1,1,1,1,0,0,1,1,1,2,1,1,1,1,1,1,1,1,1,0,0,1,0,0,0,1,0,1,0,0,1,2,1,1,0,0,1,1,1,1,1,1,2,0,0,1,1,1,1,1,1,1,1,0,0,0,1,1,1,0,0,1,1,0,1,1,1,1,1,0,0,1,0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,1,0,1,1,1,1,1,1,0,0,1,1,0,1,1,1,1,1,0,1,1,1,0,0,0,0,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,0,0,1,1,0,1,1,1,1,0,1,1,1,1,0,0,1,1,1,1,0,1,1,2,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1,1,1,1,0,1,1,1,1,1,1,0,1,0,1,0,1,0,1,1,0,1,1,1,0,0,0,1,1,1,1,1,1,0,0,1,1,1,1,1,0,1,1,0,1,1,0,1,1,1,1,0,1,1,0,1,1,1,0,1,1,1,0,0,0,1,0,1,1,1,1,1,1,1,1,0,2,1,1,1,1,1,1,1,1,1,0,1,2,1,1,1,1,1,0,0,1,1,1,0,1,1,1,0,2,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,2,2,0,1,1,0,1,1,2,1,1,0,0,0,1,1,1,1,1,0,0,0,1,0,2,1,1,2,0,1,1,1,1,0,0,1,1,1,1,1,0,1,1,0,0,0,1,1,1,1,1,0,1,1,1,0,1,1,0,0,1,1,1,1,0,1,0,2,1,1,0,1,0,1,1,2,0,0,0,1,1,2,1,1,1,0,1,1,1,0,1,1,0,0,0,2,0,1,0,0,1,1,1,1,0,0,1,1,1,1,1,0,1,0,1,1,0,1,1,1,0,1,0,0,1,1,1,1,1,1,0,1,0,0,1,0,1,0,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,0,1,1,1,1,1,1,0,1,0,1,1,1,1,0,0,1,1,1,0,0,0,1,0,1,0,1,1,1,1,0,0,1,0,1,1,1,0,0,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,0,1,1,0,1,1,1,1,0,1,0,1,1,1,0,0,0,0,0,1,1,0,1,1,1,0,0,1,1,1,1,1,0,0,1,1,1,0,0,0,2,1,0,0,1,1,1,1,1,1,1,1,0,1,1,1,0,0,0,0,0,0,1,0,1,0,1,1,0,1,1,1,0,1,1,1,1,1,0,2,0,1,1,1,1,2,1,1,1,1,0,0,0,0,1,1,1,1,0,1,1,0,1,0,0,1,1,1,1,0,1,1,1,1,1,1,1,0,1,0,1,0,1,1,1,1,0,1,1,0,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,0,1,0,1,0,1,0,0,0,0,0,0,1,0,1,2,1,1,0,1,0,0,0,0,1,0,0,1,0,1,1,1,0,1,1,0,1,1,1,0,1,1,1,1,1,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,1,0,1,0,1,0,0,1,1,0,1,1,1,1,1,1,0,1,0,1,1,0,0,1,0,1,1,1,0,1,0,1,1,0,1,1,2,1,2,1,0,1,1,1,0,0,0,1,0,1,1,1,0,0,1,0,0,0,1,0,0,1,0,1,0,0,2,1,1,0,1,0,0,1,0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,1,0,1,2,0,1,1,0,1,0,0,1,1,0,1,0,1,0,1,0,0,0,0,1,0,1,1,0,0,1,0,0,0,1,1,0,1,0,0,0,0,1,1,0,1,0,1,1,1,0,0,2,0,1,1,0,1,1,0,2,0,1,0,1,1,0,0,2,1,0,1,0,1,0,1,0,0,0,0,0,2,1,0,1,1,1,0,1,0,2,0,0,1,1,1,1,1,0,1,0,0,1,0,1,1,0,0,0,1,0,1,0,1,1,0,1,1,2,0,1,1,0,1,1,0,1,1,0,1,1,0,1,0,0,1,0,1,0,1,1,1,2,0,0,0,1,1,1,0,0,1,0,1,1,1,0,0,0,1,0,0,1,1,1,0,1,0,0,1,0,1,1,1,1,1,2,0,1,0,0,0,1,0,2,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,0,0,0,1,2,0,1,0,0,0,0,1,0,0,0,1,1,1,1,2,0,1,1,2,1,1,0,1,1,1,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,1,1,1,1,1,0,1,1,1,1,0,0,0,0,1,0,1,0,1,1,1,0,1,1,0,0,1,1,1,1,0,0,1,0,1,0,0,0,0,1,1,1,2,0,0,0,0,0,1,0,1],"coloraxis":"coloraxis","symbol":"circle"},"mode":"markers","name":"","scene":"scene","showlegend":false,"x":[77,55,52,68,60,64,70,65,60,77,71,50,60,57,55,57,52,52,54,71,70,83,70,63,67,77,82,53,44,50,64,65,72,39,65,78,44,59,59,78,73,49,72,41,63,70,43,72,63,56,67,72,43,65,55,58,71,55,44,60,57,48,45,55,58,70,76,62,60,47,63,59,65,61,61,54,63,55,73,56,59,62,80,46,72,47,63,63,68,41,37,74,37,44,42,49,49,42,56,52,50,38,56,63,63,50,44,43,54,60,64,71,53,51,61,59,64,61,60,65,73,65,49,67,52,63,57,76,61,61,73,50,64,57,72,58,78,58,49,59,55,63,73,45,48,55,65,64,61,71,45,62,70,54,53,66,76,72,69,50,40,88,51,76,38,53,80,65,64,80,44,32,65,71,61,62,65,68,66,86,70,62,50,88,55,79,64,75,64,65,58,62,68,51,38,55,69,57,51,63,71,45,50,53,43,42,76,76,65,61,63,65,64,52,63,50,68,76,62,51,35,65,67,63,44,53,62,57,66,51,53,63,50,52,58,77,50,68,73,60,64,65,51,59,42,45,56,73,62,65,78,76,49,57,51,61,53,47,63,59,52,56,56,64,63,74,38,64,58,49,66,59,52,61,55,63,64,74,51,49,52,56,63,49,49,50,61,37,84,64,44,49,70,54,59,79,70,65,81,77,68,64,50,64,70,65,40,61,45,54,74,66,27,50,65,65,51,63,50,38,45,57,69,52,53,39,41,72,64,27,70,49,52,56,62,62,60,51,57,62,52,55,57,58,79,47,46,59,38,54,79,55,61,58,54,57,71,67,54,72,57,61,52,87,69,56,46,52,65,45,53,55,65,62,58,64,69,69,63,62,74,62,45,83,59,41,51,70,60,59,59,62,43,65,77,45,47,75,73,74,55,67,44,45,79,57,65,63,55,75,65,62,37,61,60,53,64,74,70,83,66,65,62,57,50,48,53,57,52,56,68,68,53,83,43,37,63,63,76,76,68,59,68,53,42,72,63,66,62,62,72,63,42,74,44,50,80,65,64,40,48,59,55,73,65,37,67,54,74,80,48,43,64,63,62,41,70,34,63,50,70,49,42,66,63,35,38,64,63,37,61,32,61,77,65,44,53,56,64,55,75,63,62,69,64,64,62,75,70,67,53,73,68,77,82,76,59,49,62,56,35,66,57,72,61,46,60,34,67,70,55,64,75,56,50,68,76,73,53,65,66,66,77,63,39,70,63,62,71,62,70,46,81,55,37,54,55,60,64,61,71,33,32,52,63,42,74,61,46,60,66,58,57,64,70,68,43,63,34,49,50,83,52,61,62,61,71,57,51,52,46,54,79,61,64,70,44,52,67,65,63,56,52,65,80,70,65,63,65,81,73,62,61,42,69,63,62,47,42,65,55,63,60,72,72,77,63,42,51,75,71,60,57,52,59,67,78,49,62,52,59,70,66,60,69,50,59,50,34,59,84,74,49,52,74,63,64,68,43,63,63,52,48,52,56,61,63,55,44,63,53,83,56,78,62,71,66,40,63,50,76,42,78,75,42,50,69,33,53,54,49,57,60,71,53,49,64,67,72,67,63,56,65,42,44,66,51,62,68,54,37,67,46,54,51,69,72,57,49,77,45,37,73,74,60,52,68,66,55,59,62,43,44,50,62,70,38,78,51,78,66,65,52,75,43,70,83,70,64,51,52,80,67,66,69,71,56,41,57,61,44,38,54,45,36,42,60,64,53,65,47,72,70,67,53,70,61,44,54,83,70,66,43,58,56,52,57,59,72,64,55,80,70,64,57,65,74,63,70,61,67,46,58,45,67,67,66,73,75,55,56,55,74,65,73,56,73,75,52,88,57,69,54,63,56,59,63,55,53,62,71,69,70,60,56,59,52,30,81,60,62,54,55,53,56,43,64,43,37,71,60,58,56,51,71,66,76,56,62,46,58,82,62,83,78,54,73,67,35,40,52,77,64,58,70,40,62,56,82,71,52,52,54,70,58,36,43,63,72,50,52,51,71,54,80,56,53,85,57,62,59,67,70,68,61,51,47,67,67,52,52,52,65,50,60,71,37,54,74,58,69,60,67,58,69,54,67,56,78,63,65,49,80,43,65,64,71,71,73,60,71,52,76,53,61,64,67,75,71,62,65,68,55,65,76,57,66,64,88,79,44,42,63,55,72,58,78,45,64,63,52,43,51,87,57,62,72,61,45,74,76,49,37,60,62,92,71,26,67,52,59,59,70,55,61,57,57,53,62,64,53,67,39,76,64,70,55,52,63,61,70,75,63,64,54,41,68,59,60,50,76,43,53,45,43,66,65,79,58,69,67,59,78,79,56,76,38,55,63,62,61,51,51,60,69,61,45,60,50,67,76,57,41,60,70,66,63,65,49,68,74,63,66,77,48,58,58,74,78,62,52,46,54,62,53,66,92,59,70,68,72,66,67,60,61,53,63,64,59,84,55,62,62,52,59,74,60,69,70,62,64,65,61,67,62,62,62,54,66,79,75,54,52,43,68,57,66,57,75,49,62,72,72,73,58,65,78,70,50,47,73,72,81,48,83,69,50,44,43,70,77,55,56,68,81,55,60,62,62,59,51,66,63,66,61,62,67,50,57,75,68,79,44,62,51,54,53,63,52,63,54,56,57,49,66,35,57,55,60,80,62,81,76,67,54,78,59,69,53,59,81,74,75,61,42,49,50,66,68,52,63,62,60,64,65,55,61,58,75,65,73,68,80,56,46,67,33,43,81,51,62,61,56,67,56,60,65,59,62,63,55,55,62,39,54,80,69,43,58,64,60,52,66,64,65,47,60,65,78,81,65,79,57,80,69,63,54,74,63,87,83,74,63,64,82,78,68,70,62,78,38,57,77,76,66,67,82,48,63,52,79,63,59,68,70,74,57,61,40,63,78,63,63,43,63,69,61,69,54,63,61,49,77,63,85,77,74,52,42,75,75,63,62,71,37,57,52,63,50,74,73,56,60,71,57,66,47,76,59,65,64,74,52,38,75,65,66,53,62,64,70,83,42,70,69,72,74,79,62,70,79,69,55,73,64,57,71,68,83,63,84,54,86,82,63,69,72,66,72,68,75,57,65,75,41,74,51,90,56,77,64,59,68,55,63,70,66,66,70,60,63,62,44,53,74,38,55,64,34,77,47,66,34,71,51,63,59,65,67,71,69,56,58,57,60,72,44,80,78,60,70,55,75,55,57,65,65,53,76,56,80,74,69,54,61,75,56,65,50,74,80,81,52,85,65,75,64,76,70,82,69,53,76,73,62,54,82,68,66,64,65,66,80,71,55,63,61,69,69,64,75,71,62,78,67,79,75,52,65,58,82,65,72,76,83,58,52,75,70,69,77,82,67,77,70,77,63,66,55,64,55,63,46,75,74,67,76,66,65,51,68,90,63,64,63,63,43,65,50,67,69,69,66,62,66,73,63,61,61,67,57,53,84,70,70,77,50,62,73,65,45,76,60,73,60,76,64,62,65,66,83,65,52,60,74,68,70,73,57,75,61,61,65,56,65,65,67,67,70,62,63,82,64,70,76,70,43,54,54,57,62,76,72,73,75,64,79,65,72,65,78,41,77,80,81,63,66,78,56,66,61,62,84,60,64,60,67,53,62,65,55,53,73,62,65,70,80,67,73,51,44,63,80,85,84,80,75,77,57,68,51,58,67,67,67,72,55,64,63,57,75,65,70,63,82,72,69,64,77,59,65,68,50,87,75,68,64,75,66,70,75,76,55,57,78,56,68,66,76,46,83,66,60,70,78,75,60,57,63,75,76,61,61,73,66,88,85,54,77,53,77,62,71,70,77,77,70,55,79,63],"y":["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","1","1","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","1","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","1","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","1","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","1","0","0","0","0","0","0","0","0"],"z":["3","0","2","2","3","0","2","2","2","3","0","2","2","2","2","2","2","2","2","0","0","0","2","0","0","2","2","3","2","3","2","2","2","0","2","2","3","0","3","0","2","0","2","0","2","3","2","0","2","0","0","3","0","0","0","2","2","2","0","2","2","0","2","2","2","2","2","3","0","2","2","2","2","2","0","2","0","0","0","2","2","3","2","2","0","0","0","2","2","2","0","2","2","0","2","0","2","0","0","0","2","0","3","0","3","2","2","2","2","2","0","0","2","2","1","2","3","2","2","2","2","2","2","2","0","2","2","2","2","2","2","0","0","0","3","0","3","0","0","0","0","2","2","2","0","0","0","2","0","3","2","3","2","0","2","3","2","2","2","2","2","2","2","2","2","2","2","0","3","0","2","2","2","0","2","0","2","2","2","2","0","3","2","0","3","2","0","2","3","0","2","0","0","0","2","0","3","0","0","2","2","2","0","0","0","0","3","2","2","2","2","3","0","2","2","2","2","2","2","1","0","3","2","2","0","2","2","2","2","0","2","2","2","1","2","2","2","3","2","0","2","2","3","2","0","2","0","2","0","0","2","2","2","2","2","0","2","3","2","2","3","0","0","2","2","0","2","2","0","2","0","0","0","2","2","2","2","0","0","2","2","2","0","2","0","3","0","0","2","2","0","0","2","3","3","3","0","2","2","2","2","2","0","2","0","2","0","0","3","2","2","0","2","0","0","0","2","0","0","0","2","2","0","2","0","0","0","2","2","2","0","3","2","0","0","0","2","2","0","0","2","2","2","0","2","0","2","2","2","2","2","2","2","2","0","0","2","0","2","2","3","0","0","2","2","0","2","2","0","2","2","0","2","2","0","0","0","0","2","3","3","2","2","2","0","2","2","2","2","0","2","2","0","2","3","0","3","2","0","2","2","2","2","0","2","0","2","3","0","2","2","2","0","2","2","2","2","2","2","2","2","2","2","0","2","0","2","0","0","3","2","2","0","2","0","0","3","0","2","2","2","2","0","0","0","0","2","2","3","2","3","2","0","2","0","0","0","0","2","0","0","0","2","3","0","0","0","0","0","3","2","0","0","2","3","0","0","0","2","0","2","0","0","2","2","2","0","2","2","0","3","0","2","2","0","2","3","2","2","2","3","2","0","0","2","2","3","2","2","2","0","0","2","2","2","2","0","2","0","2","2","2","2","3","2","0","2","0","0","0","2","3","3","1","0","2","2","2","0","0","2","2","2","2","2","2","3","2","2","0","2","2","0","2","2","0","2","0","2","0","0","0","2","2","2","2","2","0","2","2","2","0","2","2","0","2","0","0","2","0","2","2","2","0","0","2","2","2","0","0","0","2","3","0","2","0","0","0","3","2","2","3","0","0","2","2","2","0","0","2","2","0","0","0","2","3","2","2","0","0","0","3","2","0","3","0","2","0","0","0","0","0","0","0","2","2","2","2","2","0","2","2","0","3","2","2","2","2","2","0","2","2","0","0","2","2","3","2","0","2","3","2","2","0","2","0","2","2","0","3","2","3","0","3","0","3","0","0","0","2","0","0","2","3","0","2","3","0","0","0","0","2","0","0","2","2","2","3","0","0","0","2","0","0","0","2","2","0","2","0","0","3","3","0","0","2","2","2","0","2","2","0","3","2","0","2","0","2","2","0","3","2","2","0","2","2","2","2","0","0","0","2","2","2","2","0","2","2","2","0","2","2","2","2","2","0","2","0","0","2","2","0","2","2","2","0","2","0","3","0","2","2","3","2","2","2","0","2","0","3","2","2","2","2","0","0","0","2","2","2","3","2","0","2","0","2","2","0","2","2","2","2","2","0","2","2","2","0","2","2","2","0","2","0","2","2","2","2","2","0","3","2","0","3","2","2","0","0","0","0","2","0","1","2","2","0","2","0","3","2","0","2","2","0","2","2","0","0","2","2","2","2","0","0","2","0","0","0","2","0","0","2","0","3","0","0","0","2","2","2","3","2","3","3","2","3","2","0","2","0","0","0","0","3","0","0","2","0","3","0","2","2","2","2","3","2","0","2","0","2","2","0","2","2","2","0","2","2","0","3","0","0","2","2","2","2","0","0","2","0","2","2","2","0","0","2","2","0","2","0","2","2","2","0","3","0","2","2","2","2","2","2","2","0","0","2","3","2","0","2","0","2","0","2","2","2","0","2","2","3","2","0","0","0","2","0","2","2","2","0","2","1","0","3","0","0","0","2","0","2","2","2","0","2","0","2","2","0","2","0","0","3","3","2","2","2","0","3","0","2","2","2","0","2","2","0","0","2","0","2","3","0","2","2","2","0","2","0","2","2","0","0","3","0","0","2","0","2","3","0","2","2","2","0","0","2","2","3","2","0","3","2","2","2","3","2","2","0","2","0","2","0","2","0","2","2","2","0","0","0","2","0","2","0","0","2","2","0","0","0","0","3","0","2","0","0","3","2","3","2","2","2","2","2","0","0","0","2","3","2","2","0","2","2","2","2","0","2","2","2","0","2","2","2","2","2","0","2","2","2","2","2","2","0","2","2","2","0","0","2","0","2","2","0","3","0","2","3","2","0","2","2","2","0","0","2","0","2","0","2","2","2","3","2","0","0","3","0","2","0","2","0","2","2","3","0","2","2","0","2","2","2","2","0","2","0","3","0","0","2","0","2","2","0","0","0","0","0","3","0","2","0","2","3","2","0","2","2","2","2","2","3","3","3","0","1","2","0","2","0","0","0","2","0","0","0","0","0","2","2","0","0","3","3","0","3","0","2","3","0","2","2","0","2","0","2","3","0","0","0","2","0","2","0","0","2","2","2","2","2","0","2","2","0","2","2","2","2","0","2","2","2","2","3","3","0","0","0","0","3","2","0","2","2","3","2","2","2","0","0","0","0","0","2","2","2","2","2","0","0","3","2","2","2","2","0","0","2","0","2","2","2","0","0","0","3","0","2","0","2","2","0","0","2","0","2","2","2","0","2","2","0","2","2","0","0","2","0","2","2","0","0","2","2","2","0","2","3","3","2","2","2","1","1","3","0","3","0","2","0","2","2","0","0","0","2","2","3","3","2","2","2","0","2","2","2","3","2","0","0","2","3","0","2","0","2","2","0","3","0","2","2","3","2","0","3","0","2","2","0","2","0","2","2","0","2","2","2","3","2","2","3","2","0","2","0","0","2","0","2","3","2","2","2","0","2","0","2","0","2","2","2","0","0","2","0","1","2","2","0","2","2","0","2","2","2","2","2","3","2","0","0","3","0","2","0","3","3","1","2","2","3","3","0","0","2","2","3","2","0","3","2","3","0","3","2","0","0","2","2","2","3","3","3","0","3","3","3","2","3","0","3","3","2","3","2","3","2","2","0","3","3","2","0","2","2","0","2","2","2","3","0","2","3","2","2","2","2","3","0","0","2","0","0","0","2","2","0","2","2","0","0","2","2","0","2","0","2","2","3","2","2","2","2","2","0","2","2","3","0","0","0","2","2","2","2","0","0","2","3","2","2","0","3","3","0","2","2","3","2","2","2","0","0","2","2","3","2","0","2","0","0","2","2","0","0","2","3","2","2","0","2","3","0","3","2","3","2","0","2","3","0","2","2","0","2","2","2","2","2","0","0","3","2","2","0","0","3","2","2","2","2","2","0","2","2","2","0","2","2","3","0","2","2","0","0","0","2","2","0","0","0","2","2","2","3","0","3","2","0","2","2","3","2","2","2","2","0","2","0","2","0","2","2","0","2","0","2","2","0","0","2","0","2","2","0","2","2","2","2","2","2","0","2","0","2","2","2","2","0","3","2","0","2","2","2","0","0","2","2","2","2","2","3","3","2","3","2","2","0","2","0","2","2","2","2","2","0","2","2"],"type":"scatter3d"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"scene":{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"xaxis":{"title":{"text":"AGE"}},"yaxis":{"title":{"text":"SIM_GIPERT"}},"zaxis":{"title":{"text":"GB"}}},"coloraxis":{"colorbar":{"title":{"text":"Cluster ID"}},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]},"legend":{"tracegroupgap":0},"title":{"text":"3D Scatter Plot of Clusters"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('0d85d0f9-98e8-4243-a91b-4fe7706adb5b');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
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


```


<div>                            <div id="ca0de42d-f2c8-4e53-bbbc-1c5f9ce4439e" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("ca0de42d-f2c8-4e53-bbbc-1c5f9ce4439e")) {                    Plotly.newPlot(                        "ca0de42d-f2c8-4e53-bbbc-1c5f9ce4439e",                        [{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"hovertemplate":"Chronic Heart Failure Stage=%{label}\u003cbr\u003eCount=%{value}\u003cextra\u003e\u003c\u002fextra\u003e","labels":["No Chronic Heart Failure","Stage I","Stage II (Left Ventricular Dysfunction)","Stage II (Right Ventricular Dysfunction)","Stage IIB (Left and Right Ventricular Dysfunction)"],"legendgroup":"","name":"","showlegend":true,"values":[1522,103,29,27,19],"type":"pie"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"legend":{"tracegroupgap":0},"title":{"text":"Distribution of Chronic Heart Failure Stages"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('ca0de42d-f2c8-4e53-bbbc-1c5f9ce4439e');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>


#### Presence of chronic Heart failure (HF) in the anamnesis: 

Partially ordered attribute: there are two lines of severities: 0<1<2<4, 0<1<3<4. 

State 4 means simultaneous states 2 and 3 

0: there is no chronic heart failure 

1: I stage 

2: II stage (heart failure due to right ventricular systolic dysfunction) 

3: II stage (heart failure due to left ventricular systolic dysfunction) 

4: IIB stage (heart failure due to left and right ventricular systolic dysfunction)


```python
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

```


<div>                            <div id="e125d62b-b2b9-441c-9de2-f8431dd9f42c" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("e125d62b-b2b9-441c-9de2-f8431dd9f42c")) {                    Plotly.newPlot(                        "e125d62b-b2b9-441c-9de2-f8431dd9f42c",                        [{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"hovertemplate":"Time Elapsed=%{label}\u003cbr\u003eCount=%{value}\u003cextra\u003e\u003c\u002fextra\u003e","labels":["2-4 hours","More than 3 days","Less than 2 hours","4-6 hours","12-24 hours","More than 1 day","More than 2 days","8-12 hours","6-8 hours"],"legendgroup":"","name":"","showlegend":true,"values":[486,269,198,175,151,141,101,92,87],"type":"pie"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"legend":{"tracegroupgap":0},"title":{"text":"Time Elapsed from the Beginning of the Attack of CHD to the Hospital"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('e125d62b-b2b9-441c-9de2-f8431dd9f42c');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
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

```


<div>                            <div id="5d3245ce-c035-4337-98a7-71556c5aef2e" class="plotly-graph-div" style="height:525px; width:100%;"></div>            <script type="text/javascript">                require(["plotly"], function(Plotly) {                    window.PLOTLYENV=window.PLOTLYENV || {};                                    if (document.getElementById("5d3245ce-c035-4337-98a7-71556c5aef2e")) {                    Plotly.newPlot(                        "5d3245ce-c035-4337-98a7-71556c5aef2e",                        [{"domain":{"x":[0.0,1.0],"y":[0.0,1.0]},"hovertemplate":"Lethal Outcome=%{label}\u003cbr\u003eCount=%{value}\u003cextra\u003e\u003c\u002fextra\u003e","labels":["Alive (No Lethal Outcome)","Cardiogenic Shock","Myocardial Rupture","Ventricular Fibrillation","Asystole","Progress of Congestive Heart Failure","Pulmonary Edema","Thromboembolism"],"legendgroup":"","name":"","showlegend":true,"values":[1429,110,54,27,27,23,18,12],"type":"pie"}],                        {"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmapgl":[{"type":"heatmapgl","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"legend":{"tracegroupgap":0},"title":{"text":"Distribution of Lethal Outcomes (Causes)"}},                        {"responsive": true}                    ).then(function(){

var gd = document.getElementById('5d3245ce-c035-4337-98a7-71556c5aef2e');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })                };                });            </script>        </div>



```python
myo_infar.columns = column_names
```


```python
options=[{'label': col, 'value': col} for col in myo_infar.select_dtypes(include='number').columns]
print([col for col in myo_infar.select_dtypes(include='number').columns]) 
```

    ['ID', 'AGE', 'SEX', 'FIBR_PREDS', 'PREDS_TAH', 'JELUD_TAH', 'FIBR_JELUD', 'A_V_BLOK', 'OTEK_LANC', 'RAZRIV', 'DRESSLER', 'ZSN', 'REC_IM', 'P_IM_STEN', 'LET_IS']



```python
scatter_options = [{'label': col, 'value': col} for col in myo_infar.columns if myo_infar[col].dtype in ['float64', 'int64'] or col == 'AGE']

```


```python
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

```



<iframe
    width="100%"
    height="650"
    src="http://127.0.0.1:8050/"
    frameborder="0"
    allowfullscreen

></iframe>




```python
myo_infar.plot(kind='box', subplots=True, layout=(5, 3), sharex=False, sharey=False, figsize=(10, 8))
plt.tight_layout()
plt.show()
```


    
![png](output_33_0.png)
    



```python
## Logistic Regression Model to predict risk of complications
```


```python
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

```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00       395
               1       1.00      1.00      1.00       115
    
        accuracy                           1.00       510
       macro avg       1.00      1.00      1.00       510
    weighted avg       1.00      1.00      1.00       510
    
    Confusion Matrix:
    [[395   0]
     [  0 115]]



```python

```