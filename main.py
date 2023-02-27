import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
# container -> creates sections in a horizontal way
# columns   -> creates sections in a vertical way

header = st.container()
dataset = st.container()
features = st.container()
model_Training = st.container()


# st.markdown{
#     """
#     <style>.main{
#     background-color: #F5F5F5;
#     }
#     </style>
#     """,
#     unsafe_allow_html = True
# }




#Catching
@st.cache # Until the file name doesnt change the app will dont load again & it will give the saved version instead of loading.
def get_data(filename):
    car_data = pd.read_csv(filename)

    return car_data

with header:
    st.title('Welcome to my Data Science Project')
    st.text('In this Project it predicts or Classify  whether a customer is purchasing a vehicle or not ?')

with dataset:
    st.header('Car Purchase Dataset')
    st.text('I found this Dataset on Kaggle.')

    car_data = get_data('Data/car_data.csv')
    st.write(car_data.head())# Top 5 rows by default

    st.subheader('Annual Salary of the Customers in the Car Dataset')
    annual_Salary = pd.DataFrame(car_data['AnnualSalary'].value_counts()).head(50)
    st.bar_chart(annual_Salary)

with features:
    st.header('Features of the Dataset')
    st.text('These are the features of the dataset that will help in prediction.')

    st.markdown('* **First Feature :** I created this feature because of this.. I calculated it using this.. ')
    st.markdown('* **Second Feature :** I created this feature because of this.. I calculated it using this.. ')

with model_Training:
    st.header('Time to train Model')
    st.text('Here you got to choose the hyperparameters of the model and see how the performance changes')

    sel_col,disp_col = st.columns(2)

    max_depth = sel_col.slider('**What should be the max_depth of the Model?**',value=20,min_value=10,max_value=100,step=10)

    n_estimators = sel_col.selectbox('**How many trees should be there?**', options= [100,200,300,'No Limit'],index=0)


    sel_col.text('Here are the features of the dataset')
    sel_col.write(car_data.columns)

    input_feature = sel_col.text_input('**Which feature should be selected as a input feature?**','AnnualSalary')

    if n_estimators == 'No Limit':
        regr = RandomForestRegressor(max_depth=max_depth)
    else:        
        regr = RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimators)

    X = car_data[[input_feature]]
    y = car_data[['Purchased']]

    regr.fit(X,y)
    prediction = regr.predict(y)

    disp_col.subheader('Mean Absolute Error of the Model is: ')
    disp_col.write(mean_absolute_error(y,prediction))

    disp_col.subheader('Mean Squared Error of the Model is: ')
    disp_col.write(mean_squared_error(y,prediction))

    disp_col.subheader('R2 Score of the Model is: ')
    disp_col.write(r2_score(y,prediction))
