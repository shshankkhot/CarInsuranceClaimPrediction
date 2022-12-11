import streamlit as st
import pandas as pd
import joblib
import main
from st_aggrid import AgGrid, GridOptionsBuilder

test_df = main.df_test
train_df = main.df
proceed = False


#Caching the model for faster loading
@st.cache
def predict(data):
    clf = joblib.load('models/rf_model.sav')
    return clf.predict(data)

def grid_select(df: pd.DataFrame):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=True) #Add pagination
    gb.configure_side_bar() #Add a sidebar
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
    gridOptions = gb.build()

    grid_response = AgGrid(
        df,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT', 
        update_mode='MODEL_CHANGED', 
        fit_columns_on_grid_load=False,
        theme='alpine', #Add theme color to the table
        enable_enterprise_modules=True,
        height=500, 
        width='100%',
        reload_data=True
    )
    return grid_response

st.title('Policy Claim Predictor')
td = st.selectbox("Select the Sample Dataset for predection",(' ','Train Data','Test Data'))
size=st.slider("Select sample size",10,100,10)
st.header('Select the policy below to predict the claim status:')
if td== 'Train Data':
    prd_df=train_df.head(size)
    prd_df=prd_df.drop(["is_claim"],axis=1)
    selection = grid_select(prd_df)
    proceed=True

if td == 'Test Data':
    prd_df=test_df.sample(size)
    selection = grid_select(prd_df)
    proceed=True

if proceed==True:
    if selection.selected_rows:
        #st.write("You selected:")
        #st.json(selection["selected_rows"])
        selected = selection['selected_rows'] 
        selected_df = pd.DataFrame(selected) #Pass the selected rows to a new dataframe df
        selected_df=selected_df.drop(selected_df.columns[[0]],axis=1)
        #st.write(selected_df)

    if selection.selected_rows:
        if st.button("Predict the policy claim"):
            #for i in selected_df:
            #for index, row in selected_df.iterrows():
            #st.write(selected_df.head(0))
            result = predict(selected_df)
            for i in range(len(selected_df)):
                if result[i] == 0 :
                    st.markdown('### We are not expecting any claims on below policy')
                    st.write(selected_df.loc[[i]])
                else:
                    st.markdown('### High posibility of getting claim on below policy')
                    st.write(selected_df.loc[[i]])
    else:
        st.warning("Please select the policy")


st.markdown('''Click on below link to understand how the grid data is sent back to streamlit and reused in other components.
                [streamlit-aggrid](https://github.com/PablocFonseca/streamlit-aggrid)''')
