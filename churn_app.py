import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
#import matplotlib.pyplot as plt
import seaborn as sns
import lime 
from lime import lime_tabular
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go


st.write("""

# Churn Prediction App

Customer churn is defined as the loss of customers after a certain period of time. Companies are interested in targeting customers
who are likely to churn. They can target these customers with special deals and promotions to influence them to stay with the company. 

This app predicts the probability of a customer churning using Telco Customer data. Here customer churn means the customer does not make another purchase after a period of time. 

This app allows the users to pick from three tree-based machine learning algorithms. Users can provide inputs in the csv file or enter in the left pane.
This app also has functionality which shows model explanations from a popular LIME- python library.

This app is made by Sanket Shah with Streamlit.
""")

df_selected = pd.read_csv("telco_churn.csv")

df_selected_all = df_selected[['gender', 'Partner', 'Dependents', 'PhoneService','tenure', 
                               'SeniorCitizen', 'Partner', 'Dependents', 'MonthlyCharges', 'Churn']].copy()

##Next, let’s define a function that allows us to download the read-in data:
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions

    href = f'<a href="data:file/csv;base64,{b64}" download="churn_data.csv">Download CSV File</a>'
    return href

st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown(filedownload(df_selected_all), unsafe_allow_html=True)

##The next thing we can do is display the output of our model
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

##Type of model
model_option = st.sidebar.selectbox(
     'Which Model would you like to use?',
     ('Random Forest', 'Gradient Boosting', 'AdaBoost'))

st.write('You selected ', model_option, ' Model which will be used to make predictions on new observations.')

## use the default input if the user does not specify input.
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:

##Numerical Input Slider and Categorical Input Select Box
    def user_input_features():

        gender = st.sidebar.selectbox('gender',('Male','Female'))
        PaymentMethod = st.sidebar.selectbox('PaymentMethod',('Bank transfer (automatic)', 'Credit card (automatic)', 'Mailed check', 'Electronic check'))
        seniorcitizen = st.sidebar.selectbox('Senior Citizen', (0,1))
        Partner = st.sidebar.selectbox('Partner', ('Yes', 'No'))
        Dependents = st.sidebar.selectbox('Dependents', ('Yes', 'No'))
        MonthlyCharges = st.sidebar.slider('Monthly Charges', 18.0,118.0, 18.0)
        tenure = st.sidebar.slider('Tenure', 0.0,72.0, 0.0)
        
        data = {'gender':[gender], 
                'PaymentMethod':[PaymentMethod], 
                'MonthlyCharges':[MonthlyCharges], 
                'tenure':[tenure],
                'SeniorCitizen':[seniorcitizen],
                'Partner':[Partner],
                'Dependents':[Dependents],
                }
        features = pd.DataFrame(data)
        return features

    input_df = user_input_features()
    
##Next, we need to display the output of our model. First, let’s display the default input parameters.
##We read in our data:
churn_raw = pd.read_csv('telco_churn.csv')
churn_raw.fillna(0, inplace=True)
churn = churn_raw.drop(columns=['Churn'])
df = pd.concat([input_df,churn],axis=0)

##Encode our features: 
encode = ['gender', 'PaymentMethod', 'Partner', 'Dependents']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

df = df[:1] # Selects only the first row (the user input data)

df.fillna(0, inplace=True)

##Select the features we want to display:
features = ['MonthlyCharges', 'tenure', 'SeniorCitizen', 
       'gender_Female', 'gender_Male',
       'PaymentMethod_Bank transfer (automatic)',
       'PaymentMethod_Credit card (automatic)',
       'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
       'Partner_Yes', 'Partner_No', 'Dependents_No', 'Dependents_Yes']

df = df[features]

#Finally, we display the default input using the write method:
# Displays the user input features

st.subheader('User Input features')
print(df.columns)

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded (currently it will use the first row for making predictions). Currently using the input parameters from left pane (shown below).')
    st.write(df)

##load saved predcited model
load_clf_rf = pickle.load(open('clf_rf.pkl', 'rb'))
load_clf_gb = pickle.load(open('clf_gb.pkl', 'rb'))
load_clf_ada = pickle.load(open('clf_ada.pkl', 'rb'))

if model_option == 'Random Forest':
    my_model = load_clf_rf
elif model_option == 'Gradient Boosting':
    my_model = load_clf_gb
elif model_option == 'AdaBoost':
    my_model = load_clf_ada

#Generate binary scores and prediction probabilities:
#prediction = load_clf.predict(df)
#prediction_proba = load_clf.predict_proba(df)

##And write the output:
def classify_me(model, input_me):
    churn_labels = np.array(['No','Yes'])
    label = model.predict(input_me)[0]
    churn_proba = model.predict_proba(input_me)
    
    if churn_labels[label] == 'Yes':
        st.subheader(churn_labels[label] + ', the customer is predicted to churn with ' + str(round(churn_proba[0,1]*100,0)) +'% probability')
    elif churn_labels[label] == 'No':
        st.subheader(churn_labels[label] + ', the customer is predicted NOT to churn with ' + str(round(churn_proba[0,0]*100,0)) +'% probability')
    else:
        st.subheader("Something is wrong! Check code")
        
    st.subheader('Prediction Probability')
    st.write(churn_proba)

classify_me(my_model, df)
    
# Plot feature importance
feat_importances_2 = pd.DataFrame(data=my_model.feature_importances_, index=df.columns,  columns=['feature_importance']).reset_index().sort_values(by=['feature_importance'], ascending=False).rename(columns={'index': 'feature_name'})

#plotly chart chart
fig = px.bar(feat_importances_2, x='feature_name', y='feature_importance', 
             title= "Feature Importance")

st.write(
"""
### What features are the most important according to the model? 
##### Click on either grid or bar chart to view the results!
""")

#insert radio button to choose to display grid or chart
selection_input = st.radio(
     "Please Select an Option!",
     ('Grid', 'Bar Chart'))

if selection_input == 'Grid':
    st.dataframe(feat_importances_2)
elif selection_input == 'Bar Chart':
    st.plotly_chart(fig)
else:
     st.write("Nothing to display!")
        
        
### import X input
##col names into np array
X_cols = pickle.load(open('churn_input_X.pkl', 'rb')).columns
#st.write(X_cols)

##col inputs into np array
X = pickle.load(open('churn_input_X.pkl', 'rb')).to_numpy()
#st.write(X)

##input row
df_2 = df.to_numpy().copy()
#st.write(df_2[0])
#st.write(df_2[0].shape)

st.subheader("What does it mean for our example? i.e What variables are driving the prediction?")
st.write("Click to explore more")

### Lime Explnation for a single row
explain_pred = st.button('Explain Predictions')

if explain_pred:
    with st.spinner('Generating explanations'):
        lime_explainer = lime_tabular.LimeTabularExplainer(training_data=X, feature_names=X_cols, class_names=[0,1], mode='classification')
        lime_exp = lime_explainer.explain_instance(data_row = df_2[0], predict_fn = my_model.predict_proba, num_features = 13, num_samples = 7042)
        components.html(lime_exp.as_html(), height=600)

        #lime_exp.predict_proba
        #lime_exp.as_list()
        lime_df = pd.DataFrame(lime_exp.as_list(), columns = ['Variable_Threshold','Probability_weight'])
        st.dataframe(lime_df)
        
        fig = lime_exp.as_pyplot_figure()
        #st.plotly_chart(fig)
        st.pyplot(fig)

