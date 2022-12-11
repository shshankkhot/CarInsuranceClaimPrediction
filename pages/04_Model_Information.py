import streamlit as st
import pandas as pd
import main  
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from category_encoders import OrdinalEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

df = main.df
df_test=main.df_test
st.title("Model Building and Evaluation Details")

st.header("Seprating Target from source data")

target = "is_claim"
X = df.drop(target, axis=1)
y = df[target]
#st.write(X_train.shape, y_train.shape)

#X_test = df_test #.drop(target, axis=1)
#y_test = df_test[target]
#st.write(X_test.shape, y_test.shape)

#Code to resample the data 
#X = X_train
#y = y_train
st.write(X.shape, y.shape)

st.write(y.dtype)
st.header("Train and Test Split")
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)
#X_train,X_test,y_train,y_test=train_test_split(X_sm,y_sm,test_size = 0.2, random_state = 42)

st.write("Traning Data",X_train.shape, y_train.shape)
st.write("Testing Data",X_test.shape, y_test.shape)

#Resampling
from imblearn.over_sampling import RandomOverSampler
sampler = RandomOverSampler(random_state=1)
X_train_over, y_train_over = sampler.fit_resample(X_train,y_train)

st.write ("Resampling",X_train_over.shape, y_train_over.shape)
#print(X_train_over.shape, y_train_over.shape)

model_baseline = y_train.value_counts(normalize=True).max()
st.header('Model Baseline')
st.write(model_baseline)

def make_model_pl():
    lr = make_pipeline(OneHotEncoder(),
                    LogisticRegression())
    dt = make_pipeline(OrdinalEncoder(),
                    DecisionTreeClassifier(random_state=1))
    rf = make_pipeline(OrdinalEncoder(),
                    RandomForestClassifier(random_state=1))
    gb = make_pipeline(OrdinalEncoder(),
                    GradientBoostingClassifier(random_state=1))

    model_list = [("lr", lr),("dt", dt),("rf", rf),("gb", gb)]


    for name, model in model_list:
        
        model.fit(X_train_over, y_train_over)
        
        y_pred = model.predict(X_test)
        
        score = accuracy_score(y_test, y_pred)
        st.write(f"The test accuracy score of {name} is {score}")

st.header("Test the Model Performance")        
ml = {" ":" ","LogisticRegression":"lr","DecisionTreeClassifier":"dt","RandomForestClassifier":"rf","GradientBoostingClassifier":"gb"}   
Obj = st.selectbox("### Please Select the Model from the list :",ml.keys())

ml_selected = ml.get(Obj)


if ml_selected == 'lr':
    st.markdown('----------------------------------------------------------------')
    st.write('Test Accuracy of LogisticRegression model',0.4637767727621811)
if ml_selected == 'dt':
    st.markdown('----------------------------------------------------------------')
    st.write('Test Accuracy of DecisionTreeClassifier model',0.8827545012373069)
if ml_selected == 'rf':
    st.markdown('----------------------------------------------------------------')
    st.write('Test Accuracy of RandomForestClassifier model',0.9211536820547829)
if ml_selected == 'gb':
    st.markdown('----------------------------------------------------------------')
    st.write('Test Accuracy of GradientBoostingClassifier model',0.5745370765423671)
st.markdown('----------------------------------------------------------------')

st.header('From the model scores, we can see that RandomForestClassifier performed best on the test data with acc_score of 0.92')
def final_model():
    rff = make_pipeline(OrdinalEncoder(),
                    RandomForestClassifier(random_state=1))
    params = {"randomforestclassifier__n_estimators":range(25,125,25)}

    Model = GridSearchCV(rff,
                    param_grid=params,
                    cv=5,
                    n_jobs=-1,
                    verbose=1)
    
    fit_model = Model.fit(X_train_over, y_train_over)

    st.write("Model best estimator",fit_model.best_estimator_)
    st.write("Model best score",fit_model.best_score_)
    st.write("Model best parameter",fit_model.best_params_)

   
    # save the model to disk
    #joblib.dump(fit_model, 'rf_model.sav')

    st.write("MAE:", mean_absolute_error(y_test, fit_model.predict(X_test)))
    st.write("MSE:", mean_squared_error(y_test, fit_model.predict(X_test)))
    st.write("R2:", r2_score(y_test, fit_model.predict(X_test)))

    y_pred_test =  fit_model.predict(X_test)
    st.write("Model Test Accuracy",round(accuracy_score(y_test, y_pred_test),2))
    
    #Model.save_model('rff_model.json')

#Commented option to run model to save streamlit resourcess 
#rf_ml = st.button("Run RandomForestClassifier  Model")
rf_ml=False
if rf_ml == True:
    final_model()
else:
    st.markdown("Results for RandomForestClassifier Model")   
    st.write('Model best score',0.9881158708245547) 
    st.write("Best Parameter \{","\"randomforestclassifier__n_estimators\":",100,"\}")

    st.write("MAE:",0.07884631794521717)
    st.write("MSE:", 0.07884631794521717)
    st.write("R2:", -0.30489884956244895)
    st.write("Model Test Accuracy",0.92)

