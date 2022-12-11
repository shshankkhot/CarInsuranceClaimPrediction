import streamlit as st
import matplotlib.pyplot as plt
import plost 
import seaborn as sns
import numpy as np
import main
import sweetviz as sv
import streamlit.components.v1 as components

# https://pypi.org/project/sweetviz/
# https://github.com/fbdesignpro/sweetviz

df = main.df
#st.header("Exploratory Data Analysis (EDA)")
st.markdown("# Exploratory Data Analysis (EDA)")
st.sidebar.header("Exploratory Data Analysis")
#st.write(df.shape)

tab1,tab2,tab3 = st.tabs(['Collinearity','Univariate Analysis','Sweetviz'])

with tab1:
    if df is not None:
        sns_fig = plt.figure(figsize=(10, 4))
        st.title("Checking for multicollinearity") 
        st.markdown('To use linear regression for modelling,its necessary to remove correlated variables to improve your model. \nWe are using pandas “.corr()” correlations function to do the same and visualizing the correlation matrix using a heatmap in seaborn.')
        correlation = round(df.select_dtypes("number").drop("is_claim", axis=1).corr(),2)
        # applying mask
        mask = np.triu(np.ones_like(df.select_dtypes("number").drop("is_claim", axis=1).corr()))
        sns.heatmap(correlation,annot=True,mask=mask);
        st.pyplot(sns_fig)    
        st.markdown('''
                    -> Lighter shades represents positive correlation while lighter shades represents negative correlation.
                    \n-> Here we can infer that "length" has strong positive correlation with “displacement” and "turning radius".
                    \n-> “age of policy holder” and “population density” has almost no correlation with all other varibales.
                    ''')
with tab2:
    if df is not None:
        st.title("Univariate Analysis")   
        UA_Object = st.selectbox("Select Feature Type:",(' ','Numerical','Categorical'))
        #numerical features
        numerical= df.select_dtypes(include =[np.float64,np.int64])

        #categorical features
        categorical = df.select_dtypes(include =[np.object])
        #st.write(categorical.head())
        if UA_Object == ' ':
            st.markdown('Please select feture type Numerical or Categorical.')

        if UA_Object == 'Numerical':
            
            num_target = [i for i in numerical.columns]
            st.header("Numerical Features")
            num_fig = plt.figure(figsize=(10,15))
            for n,column in enumerate(num_target):
                plot=plt.subplot(8,2,n+1)   
                sns.distplot(df[column],color='green')
                plt.title(f'{column.title()}',weight='bold')
                plt.tight_layout()
        
            st.pyplot(num_fig)   

        if UA_Object == 'Categorical' :
            #categorical=categorical.drop('policy_tenure',axis=1)
            # Drop first column of dataframe
            categorical = categorical.iloc[: , 1:]
            cat_target = [i for i in categorical.columns]
            st.header("Categorical Features")
            cat_fig = plt.figure(figsize=(15,25))
            for n1,column1 in enumerate(cat_target):

                plot=plt.subplot(14,2,n1+1)
                #sns.countplot(df[column1])
                sns.countplot(data=df,x=df[column1]);
                plt.title(f'{column1.title()}',weight='bold')
                plt.tight_layout()
            
            st.pyplot(cat_fig) 


#@st.cache
def app(data=None):
    
    # Use the analysis function from sweetviz module to create a 'DataframeReport' object.
    report = sv.analyze([data, "Train"], target_feat='is_claim')
    report.show_html(filepath='EDA.html', open_browser=False, layout='vertical', scale=1.0)                   

    # Render the output on a web page.
    #analysis.show_html(filepath='EDA.html', open_browser=False, layout='vertical', scale=1.0)
    #path='file:///C:/Users/Sarika/Documents/FALL_2022/CarInsuranceclaimprediction/EDA.html'
    #components.iframe(path) #, width=1100, height=1200, scrolling=True)

with tab3:
    if df is not None:
        #referesh_sv = st.checkbox("Refersh Sweetviz")
        
        #if referesh_sv is True:
        #    st.write('Refreshing Sweetviz...')
        #    app(data=df)
        
        st.title("EDA With Sweetviz")
        st.markdown('Sweetviz is a python library that focuses on exploring the data with the help of beautiful and high-density visualizations. It not only automates the EDA but is also used for comparing datasets and drawing inferences from it.')
        st.code('''
        #We have already loaded the dataset in the variable named “df”, we will just import the dataset and create the EDA report in just a few lines of code.
        import sweetviz as sv
        sweet_report = sv.analyze(df)
        sweet_report.show_html('EDA.html')
        \n#This step will generate the report and save it in a file named “EDA.html”
        ''')

        st.header("Understanding the Report")
        st.markdown("The report contains characteristics of the different attributes along with visualization.")

        # Render the output from sweetviz on a web page .
        path='file:///C:/Users/Sarika/Documents/FALL_2022/CarInsuranceclaimprediction/EDA.html'
        #[This is an external link to genome.gov](https://www.genome.gov/)
        st.markdown("In this report, we can clearly see what are the different attributes of the datasets and their characteristics including the missing values, distinct values, etc.")
        st.markdown('[Sweetviz](https://pypi.org/project/sweetviz/)')
        st.markdown('''
                    On the top, we have a quick summary of the dataset. Number of rows, columns, type of variables, whether the dataset contains duplicates, etc.
                    \nBelow the summary section, we’ll find details on each variable (column) in the dataset. Clicking on a variable will expand that section with more details.
                    \nWe can also click on the ASSOCIATIONS button in the summary section to view correlation matrix where, Squares represent categorical variables (text), and circles represent numerical correlations.
                    ''')
        plot_file = open('EDA.html','r')
        plot = plot_file.read()
        components.html(plot, width=1100, height=1200, scrolling=True)
        plot = plot_file.close()
