# import required libraries

from PIL import Image
import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# from sklearn.preprocessing import label encoder
matplotlib.use('Agg')


# disable warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# set title
st.title('DevOps Team presents')

# import image
image = Image.open('photos/devops.png')
st.image(image, use_column_width=True)


def main():
    activities = ['EDA', 'Visualisation', 'model', 'About us']
    option = st.sidebar.selectbox('Selection option: ', activities)

# Dealing with EDA part
    if option == 'EDA':
        st.subheader('Exploratory Data Analysis')

        data = st.file_uploader('Upload dataset: ', type=[
                                'csv', 'xlsx', 'txt', 'json'])

        if data is not None:
            st.success('Your dataset is loaded')
            df = pd.read_csv(data)
            st.dataframe(df.head(50))

            if st.checkbox('Display Shape'):
                st.write(df.shape)

            if st.checkbox('Display columns'):
                st.write(df.columns)

            if st.checkbox('SElect multiple columns'):
                selected_columns = st.multiselect(
                    'Select preferred columns: ', df.columns)
                df1 = df[selected_columns]
                st.dataframe(df1)

            if st.checkbox('Display summmary'):
                st.write(df.describe().T)

            if st.checkbox('Display Null Value'):
                st.write(df.isnull().sum())

            if st.checkbox('Display data type:'):
                st.write(df.dtypes)

            if st.checkbox('Display Correlation of dataframe'):
                st.write(df.corr())

    # Visualization part
    elif option == 'Visualisation':
        st.subheader('Visualization of Data')

        data = st.file_uploader('Upload dataset: ', type=[
                                'csv', 'xlsx', 'txt', 'json'])

        if data is not None:
            st.success('Your dataset is loaded')
            df = pd.read_csv(data)
            st.dataframe(df.head(50))

            if st.checkbox('Select Multiple columns to plot'):
                selected_columns = st.multiselect(
                    'Select yourt preferred columns', df.columns)
                df1 = df[selected_columns]
                st.dataframe(df1)

            if st.checkbox('Display heatmap'):
                st.write(sns.heatmap(df1.corr(), vmax=1, square=True,
                                     annot=True, linecolor='red', cmap='viridis'))
                st.pyplot()

            if st.checkbox('Display Pairplot'):
                st.write(sns.pairplot(df1, diag_kind='kde'))
                st.pyplot()

            if st.checkbox('Display pie chart'):
                all_columns = df.columns.to_list()
                pie_columns = st.selectbox(
                    'Select columns to display', all_columns)
                pieChart = df[pie_columns].value_counts().plot.pie(
                    autopct='%1.1f%%')
                st.write(pieChart)
                st.pyplot()

            if st.checkbox('Scatter Plot'):

                all_columns = df.columns.to_list()
                x_column = st.selectbox(
                    'Selct x from  dataset', all_columns)
                y_column = st.selectbox(
                    'Select y from dataset', all_columns)
                hue = st.selectbox('Select hue from dataset', all_columns)

                scatter_vis = sns.relplot(data=df, x=x_column,
                                          y=y_column, hue=hue)
                st.pyplot()

    # modelling part
    elif option == 'model':
        st.subheader('Model Building')
        data = st.file_uploader('Upload dataset: ', type=[
                                'csv', 'xlsx', 'txt', 'json'])

        if data is not None:
            st.success('Your dataset is loaded')
            df = pd.read_csv(data)
            st.dataframe(df.head(50))

            if st.checkbox('Select multiple columns'):
                new_data = st.multiselect(
                    'Select your prefered columns', df.columns)
                df1 = df[new_data]
                st.dataframe(df1)

                # dividing my data into x ans y variables
                X = df1.iloc[:, 0: -1]
                y = df1.iloc[:, -1]

            seed = st.sidebar.slider('Seed', 1, 200)

            classifier_name = st.sidebar.selectbox(
                'Select your pereferred classifier: ', ('KNN', 'SVM', 'LR', 'Naive_Baise', 'Decision Tree'))

            # choosing parametrs
            def add_parameter(name_of_clf):
                param = dict()

                if name_of_clf == 'SVM':
                    C = st.sidebar.slider('C', 00.1, 15.0)
                    param['C'] = C

                if name_of_clf == 'KNN':
                    K = st.sidebar.slider('K', 1, 15)
                    param['K'] = K
                    return param

            # calling the function

            params = add_parameter(classifier_name)

            # define function for our classifier

            # def get_classifier(name_of_clf, param):

            # elif option == 'About us':
            #     st.warning('This part is not ready yet')


if __name__ == "__main__":
    main()
