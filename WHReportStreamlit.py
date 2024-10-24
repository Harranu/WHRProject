# Copyright Okwudili Harrison Anuforo
# Email: Harranu@gmail.com
# World Happiness Analysis Project 2024

# ***** Import needed libraries ***** #
import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from scipy.stats import gaussian_kde
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# *** Read the dataset into a dataframe *** #
whr_df = pd.read_csv('world-happiness-report.csv')
whr_df_2021 = pd.read_csv('world-happiness-report-2021.csv')
whr_df_2022_2024 = pd.read_csv("world-happiness-report-new-2024.csv")
# Select rows where the 'year' column is between 2022 and 2024 (I will use the dataset of 2022-2024 to test our trained model)
whr_df_2022_2024 = whr_df_2022_2024[(whr_df_2022_2024['year'] >= 2022) & (whr_df_2022_2024['year'] <= 2024)]

with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar content
st.sidebar.image("whr_image00.jpeg", use_column_width=True)

# Main content
pages = ["Preface", "Introduction", "Data Exploration & Cleaning", "Data Visualization & Analysis", "Machine Learning",
         "Insight & Conclusion"]
page = st.sidebar.radio("Table of contents", pages)

add_selectbox = st.sidebar.selectbox(
    'You can contact me!',
    ('Email: Harranu@gmail.com', 'Telephone: +49 1521 3452 042')
)

if page == pages[0]:
    st.write("<h1 style='text-align: center;'>World Happiness Report Project</h1>", unsafe_allow_html=True)
    #    st.image("whr_image003.png", caption="Caption here", use_column_width=True, clamp=True)
    st.image("whr_image003.png", use_column_width=True, clamp=True)
    st.write("<div class='customPrefaceFont'>Data Analysis Project "
             "<br>Presented by Okwudili Harrison Anuforo </div>", unsafe_allow_html=True)

if page == pages[1]:
    st.write("<h2>Introduction</h2>", unsafe_allow_html=True)
    st.write("<div class='customInnerFont'>The World Happiness Report (WHR) is a partnership programme of Gallup, "
             "the Oxford Wellbeing Research Centre, the UN Sustainable Development Solutions Network, and the WHR’s "
             "Editorial Board. The report is produced under the editorial control of the WHR Editorial Board. <br><br> "
             "The goal of this project is to estimate the happiness of countries around the world using "
             "socio-economic metrics such as gross domestic product (GDP), social support, healthy life expectancy, "
             "freedom to make life choices, generosity, and perception of corruption. As well as to understand how these "
             "variables are correlated to each other to influence the happiness ladder score and consequentially "
             "contribute to the happiness of the people in their countries. <br><br>I will analyse the "
             "data using analytical- and machine learning methods and present it using interactive visualizations, to determine the combinations of factors that explains "
             "why some countries are ranked to be happier than others. I will merge the available dataset of the "
             "world happiness report dataset from 2005 to 2021 as my base dataset and apply various machine learning methods "
             "to train and make predictions on the trained models with a sample data from the base dataset on 80/20 split. To validate or invalidate "
             "the outcome of the trained models prediction, I will make predictions on the trained models with new but "
             "similar dataset of 2022 to 2024. Finally, I will retrain the models and make further prediction with "
             " the new dataset and compare the results.<br><br> The outcome will help countries and policy makers determine "
             "the factors that influence overall happiness of the people and areas that can be improved in view of decision "
             "and policy making.<br><br></div>", unsafe_allow_html=True)

# ***  Clean the Data *** #
if page == pages[2]:

    st.write("<h2>Data Exploration & Cleaning (STEP 1)</h2>", unsafe_allow_html=True)

    # Horizontal divider
    st.markdown("<hr class='custom-hr-dc'>", unsafe_allow_html=True)

    if st.checkbox("World Happiness Report Dataframe (2005-2020)"):
        st.dataframe(whr_df)

    if st.checkbox("World Happiness Report Dataframe (2021)"):
        st.dataframe(whr_df_2021)

    if st.checkbox("World Happiness Report Dataframe (2022-2024)"):
        st.dataframe(whr_df_2022_2024)

    # Horizontal divider
    st.markdown("<hr class='custom-hr-dc'>", unsafe_allow_html=True)

    if st.checkbox("WHR Dataframe Information (2005-2020)"):
        def datacheck_df(dataframe, head=5):
            st.write("__________________________ Dataframe Info ____________________ ")
            buffer = io.StringIO()
            dataframe.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
            st.write("__________________________ Missing Values ______________________ ")
            st.write(dataframe.isnull().sum())
            missing_values_sum = whr_df.isnull().sum().sum()
            st.write("Total sum of missing values:", missing_values_sum)
            st.write("__________________________ Duplicates __________________________ ")
            st.write("Total sum of Duplicates:", dataframe.duplicated().sum())
            st.write("__________________________ Statistics __________________________ ")
            st.write(dataframe.describe())
            st.write("________________________________________________________________ ")


        # Display 'World Happiness Report and World Happiness Report Dataframes')
        datacheck_df(whr_df)

    if st.checkbox("WHR Dataframe Information (2021)"):
        def datacheck_df_2021(dataframe):
            st.write("__________________________ Dataframe Info ____________________ ")
            buffer = io.StringIO()
            dataframe.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
            st.write("__________________________ Missing Values ______________________ ")
            st.write(dataframe.isnull().sum())
            missing_values_sum = whr_df_2021.isnull().sum().sum()
            st.write("Total sum of missing values:", missing_values_sum)
            st.write("__________________________ Duplicates  __________________________ ")
            st.write("Total sum of Duplicates:", dataframe.duplicated().sum())
            st.write("__________________________ Statistics __________________________ ")
            st.write(dataframe.describe())
            st.write("________________________________________________________________ ")

        datacheck_df_2021(whr_df_2021)

    # Columns to drop in WHR dataframe
    whr_df = whr_df.drop(['Positive affect', 'Negative affect'], axis=1)

    # Columns to drop in WHR2021 and create a new column called year in WHR2021 to correspond with
    # WH and fill it with the value "2021"
    whr_df_2021_not_needed_col = ['Ladder score in Dystopia', 'Explained by: Log GDP per capita',
                                  'Explained by: Social support', 'Ladder score in Dystopia',
                                  'Explained by: Healthy life expectancy', 'Explained by: Freedom to make life choices',
                                  'Explained by: Generosity', 'Explained by: Perceptions of corruption',
                                  'Standard error of ladder score', 'upperwhisker', 'lowerwhisker',
                                  'Dystopia + residual']
    whr_df_2021 = whr_df_2021.drop(columns=whr_df_2021_not_needed_col, axis=1)
    whr_df_2021['year'] = 2021

    # WHR does not have regions, so we map regions from WHR2021 to WHR to their corresponding countries
    country_to_region = {}
    for region in whr_df_2021['Regional indicator'].unique():
        countries_in_region = whr_df_2021[whr_df_2021['Regional indicator'] == region]['Country name'].tolist()
        for country in countries_in_region:
            country_to_region[country] = region
    whr_df['Regional indicator'] = whr_df['Country name'].map(country_to_region)

    # Some regions of some countries where missing during mapping, so we check manually for missing Regional indicator column in WHR and fill it for any corresponding country
    missing_regions = {
        'Sub-Saharan Africa': ['Angola', 'Central African Republic', 'Congo (Kinshasa)', 'Somalia', 'Somaliland region',
                               'South Sudan', 'Sudan'],
        'Latin America and Caribbean': ['Cuba', 'Trinidad and Tobago', 'Suriname', 'Belize', 'Guyana'],
        'Middle East and North Africa': ['Djibouti', 'Qatar', 'Oman', 'Syria', 'Bhutan'], 'South Asia': ['Maldives']}
    country_to_region = {country: region for region, countries in missing_regions.items() for country in countries}
    whr_df['Regional indicator'] = whr_df.apply(
        lambda row: country_to_region.get(row['Country name'], row['Regional indicator']), axis=1)

    # Rename following columns, in order to merge both datasets without missing any value, then merge the two dataframes
    whr_df.rename(columns={'Life Ladder': 'Ladder score', 'Log GDP per capita': 'Logged GDP per capita',
                           'Healthy life expectancy at birth': 'Healthy life expectancy'}, inplace=True)
    whr_df_merged = pd.concat([whr_df, whr_df_2021], ignore_index=True, axis=0)

    # Drop specified countries from the 'Country name'
    # countries_to_drop = ['Cuba', 'Guyana', 'Maldives', 'Oman', 'Suriname']
    # whr_df_merged = whr_df_merged[~whr_df_merged['Country name'].isin(countries_to_drop)]

    # Some feattures have few missing values,  so we fill there missing values with the mean of column based on country/Regional indicator
    fill_missing_values = ['Logged GDP per capita', 'Social support', 'Healthy life expectancy',
                           'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
    for column in fill_missing_values:
        # whr_df_merged[column] = whr_df_merged.groupby('Country name')[column].transform(lambda x: x.fillna(x.mean()))
        whr_df_merged[column] = whr_df_merged.groupby('Regional indicator')[column].transform(
            lambda x: x.fillna(x.mean()))

    # *** Rename the preprocessed and clean dataset *** #
    whr_df = whr_df_merged
    whr_df['year'] = whr_df['year'].astype(str)

    # Save the merged and clean dataset to my disk for further analysis
    # file_path = 'cleaned_whr_data.csv'
    # whr_df.to_csv(file_path, index=False, sep=',', encoding='utf-8', header=True)

    # *** Lets clean the test dataset *** #

    # Check the test dataset content
    if st.checkbox("WHR Dataframe Information (2022-2024"):
        def datacheck_df_2022_2024(dataframe):
            st.write("__________________________ Dataframe Info ____________________ ")
            buffer = io.StringIO()
            dataframe.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
            st.write("__________________________ Missing Values ______________________ ")
            st.write(dataframe.isnull().sum())
            missing_values_sum = whr_df_2022_2024.isnull().sum().sum()
            st.write("Total sum of missing values:", missing_values_sum)
            st.write("__________________________ Duplicates  __________________________ ")
            st.write("Total sum of Duplicates:", dataframe.duplicated().sum())
            st.write("__________________________ Statistics __________________________ ")
            st.write(dataframe.describe())
            st.write("________________________________________________________________ ")


        datacheck_df_2022_2024(whr_df_2022_2024)

    # Columns to drop in new WHR2005-2023 dataframe(2005 till 2023)
    whr_df_2022_2024 = whr_df_2022_2024.drop(['Positive affect', 'Negative affect'], axis=1)

    # Rename following feature columns to correspond on both datasets
    whr_df_2022_2024.rename(columns={'Life Ladder': 'Ladder score', 'Log GDP per capita': 'Logged GDP per capita',
                                     'Healthy life expectancy at birth': 'Healthy life expectancy'}, inplace=True)

    # Rename following country columns to correspond on both datasets
    whr_df_2022_2024['Country name'] = whr_df_2022_2024['Country name'].replace(
        {'Czechia': 'Czech Republic', 'State of Palestine': 'Palestinian Territories', 'Türkiye': 'Turkey'})

    # Map countries to regions from WHRBase to WHR2005-2024 to their corresponding regions
    country_to_region = {}
    for region in whr_df['Regional indicator'].unique():
        countries_in_region = whr_df[whr_df['Regional indicator'] == region]['Country name'].tolist()
        for country in countries_in_region:
            country_to_region[country] = region
    # Create Regional indicator column in WHR2005-2024 and assign its value based on mapping with country name to region from WHRBase
    whr_df_2022_2024['Regional indicator'] = whr_df_2022_2024['Country name'].map(country_to_region)

    # Check for missing rows on Regional indicator column in WHR2005-2023 and fill the missing regions to the country
    missing_regions = {'Sub-Saharan Africa': ['Eswatini']}
    country_to_region = {country: region for region, countries in missing_regions.items() for country in countries}
    whr_df_2022_2024['Regional indicator'] = whr_df_2022_2024.apply(
        lambda row: country_to_region.get(row['Country name'], row['Regional indicator']), axis=1)

    # Fill the missing numerical values of the features with the mean. Group by 'Country name'/'Regional indicator' and fill missing values based on mean of own country/regions
    fill_missing_values = ['Logged GDP per capita', 'Social support', 'Healthy life expectancy',
                           'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
    for column in fill_missing_values:
        # whr_df_2022_2024[column] = whr_df_2022_2024.groupby('Country name')[column].transform(lambda x: x.fillna(x.mean()))
        whr_df_2022_2024[column] = whr_df_2022_2024.groupby('Regional indicator')[column].transform(
            lambda x: x.fillna(x.mean()))

    whr_df_2022_2024['year'] = whr_df_2022_2024['year'].astype(str)
    # Save the clean new/test dataset to my disk for further analysis
    # file_path = 'cleaned-world-happiness-report-2022-2024.csv'
    # whr_df_2022_2024.to_csv(file_path, index=False, sep=',', encoding='utf-8', header=True)

    # *** Data after cleaning *** #

    st.markdown("<hr class='custom-hr-dc'>", unsafe_allow_html=True)
    st.write("<h3>After Data Cleaning</h3>", unsafe_allow_html=True)
    st.markdown("<hr class='custom-hr-dc'>", unsafe_allow_html=True)

    if st.checkbox("Final/Base WHR Dataset after data cleaning (2005-2020 & 2021 merged )"):
        # Display 'World Happiness Report (merged and clean Dataframe), this will be the base DF used for my analysis
        st.dataframe(whr_df)

        # Download the dataset (download optional) for third party
        csv = whr_df.to_csv(index=False)
        st.download_button(label="Download cleaned WH Dataset as CSV", data=csv, file_name='cleaned_whr_data.csv',
                           mime='text/csv', )

    if st.checkbox("New/Test dataset after data cleaning (2022-2024)"):
        # World Happiness Report 2022-024, this will be the test DF used for ML
        st.dataframe(whr_df_2022_2024)

        # Download the New/test dataset (download optional) for third party
        csv = whr_df_2022_2024.to_csv(index=False)
        st.download_button(label="Download cleaned WH New-Test Dataset as CSV", data=csv,
                           file_name='cleaned-world-happiness-report-2022-2024.csv',
                           mime='text/csv', )

    st.markdown("<hr class='custom-hr-dc'>", unsafe_allow_html=True)

    if st.checkbox("Final-Base WHR Dataset information after data cleaning"):
        def cleaned_datacheck_df(dataframe):
            st.write("_______________ Dataframe Info after data cleaning _____________")
            buffer = io.StringIO()
            dataframe.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
            st.write("________________ Missing Values after data cleaning ___________ ")
            st.write(dataframe.isnull().sum())
            missing_values_sum = whr_df.isnull().sum().sum()
            st.write("Total sum of missing values:", missing_values_sum)
            st.write("________________ Duplicates after data cleaning ________________ ")
            st.write("Total sum of Duplicates:", dataframe.duplicated().sum())
            st.write("_________________ Statistics after data cleaning ________________")
            whr_df_statistic = whr_df.drop('year', axis=1)
            st.write(whr_df_statistic.describe())
            st.write("________________________________________________________________ ")


        cleaned_datacheck_df(whr_df)

    if st.checkbox("New/Test Dataset Information after data cleaning"):
        def cleaned_datacheck_df_2022_2024(dataframe):
            st.write("_______________ Dataframe Info after data cleaning _____________")
            buffer = io.StringIO()
            dataframe.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)
            st.write("________________ Missing Values after data cleaning ___________ ")
            st.write(dataframe.isnull().sum())
            missing_values_sum = whr_df_2022_2024.isnull().sum().sum()
            st.write("Total sum of missing values:", missing_values_sum)
            st.write("________________ Duplicates after data cleaning ________________ ")
            st.write("Total sum of Duplicates:", dataframe.duplicated().sum())
            st.write("_________________ Statistics after data cleaning ________________")
            whr_df_statistic = whr_df_2022_2024.drop('year', axis=1)
            st.write(whr_df_statistic.describe())
            st.write("________________________________________________________________ ")


        cleaned_datacheck_df_2022_2024(whr_df_2022_2024)

if page == pages[3]:

    st.write("<h2>Data Visualization & Analysis (Step 2)</h2>", unsafe_allow_html=True)

    # ead and extract clean dataset for exploration and analysis
    whr_df = pd.read_csv("cleaned_whr_data.csv")

    if st.checkbox("Visualization and Insight"):
        # Choropleth map of Ladder score distribution over the years by country
        whr_df = whr_df.sort_values(by='year', ascending=True)
        fig = px.choropleth(
            data_frame=whr_df,
            locations='Country name',
            locationmode='country names',
            color='Ladder score',
            color_continuous_scale='turbo',
            animation_frame='year',
            title='Global Happiness Ladder Scores Map Over Time By Country',
            labels={'Ladder score': 'World Happiness Ladder Score'},
            projection='natural earth',
            hover_data=['Logged GDP per capita', 'Social support', 'Healthy life expectancy',
                        'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
        )
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        # fig.update_layout(autosize=True, width=1200, height=800,margin=dict(l=0, r=0, b=10, t=10, pad=1, autoexpand=True ))
        # Create an expander
        with st.expander("Interpretation/Insight"):
            st.write("The essence of this geolocation map is to understand  the spread of WHR project in terms "
                     " of  countries and regions al well as the increasing rates of participation since inception. "
                     " As could be seen from this map at the early stage of the project around 2005-2006 fewer "
                     "countries participated. However at a point from 2007 onward we could see the exponential "
                     "increase of the geographical spread of participants to include more countries mainly from "
                     "Africa and Asia. This transition over time in terms of increase in participation by countries "
                     "could be interpreted as recognition of WHR and validity of the data it provides.\n\n Another important "
                     "insight is the drastic decrease of participation in 2020, most notable in African and some parts of "
                     "Asia, one can assumed that this has to do with the world event, specifically the Covid19 pandemic, "
                     "which restricted the movement of people around the world  for a certain period. This map also give us "
                     "better insight into the the ladder and feature scores of each countrie across time. Hover over the country "
                     "geolocation on the map to view the specific countries scores.")
        st.write("\n")

        # Ladder score over time for top 5
        whr_df_grouped = whr_df.groupby(['Country name', 'year'])['Ladder score'].sum().reset_index()
        latest_year = whr_df_grouped['year'].max()
        top_10_countries = whr_df_grouped[whr_df_grouped['year'] == latest_year].nlargest(10, 'Ladder score')[
            'Country name']
        top_10_data = whr_df_grouped[whr_df_grouped['Country name'].isin(top_10_countries)]

        fig = px.line(top_10_data, x='year', y='Ladder score', color='Country name', markers=True,
                      title='Ladder Score Over Time for Top 10 Countries')

        fig.update_layout(xaxis_title='Year', yaxis_title='Ladder Score', legend_title='Country',
                          legend=dict(x=1, y=1), margin=dict(l=0, r=0, t=30, b=0),
                          xaxis=dict(tickmode='linear', tick0=top_10_data['year'].min(), dtick=2))

        st.plotly_chart(fig)

        with st.expander("Interpretation/Insight"):
            st.write("The plot above shows the changes in ladder scores for top-performing ten countries from 2005 to 2021. "
                     "Here are some detailed observations: Denmark’s score shows an upward trend, peaking around 2013, and "
                     "then stabilizing with slight fluctuations. Performing consistently among the highest scores, indicating "
                     "strong performance. Finland’s score shows a steady increase, especially noticeable from 2015 onwards. "
                     "It has significant improvement over the years, reaching one of the highest scores by 2021. Switzerland’s "
                     "score remains relatively stable with minor fluctuations and it maintains a high score throughout the period."
                     " Iceland’s score shows some volatility but generally remains high. It's noticeable dip was around 2008-2009, "
                     "likely due to the financial crisis, followed by recovery.  Norway’s score shows a slight upward trend with "
                     "minor fluctuations, however it maintain consistent high scores, indicating stable performance. The Netherlands’"
                     " score shows a gradual increase over the years, with steady improvement, reaching high scores by 2021. "
                     "Sweden’s score shows minor fluctuations but remains relatively stable, it maintained a high score throughout "
                     "the period. New Zealand’s score shows a steady increase, especially noticeable from 2010 onwards, "
                     "with significant improvement over the years. Luxembourg’s score shows some fluctuations but generally remains "
                     "high and stable by maintaining a high score throughout the period. Austria’s score shows minor fluctuations"
                     " but remains relatively stable, by maintaining a high score throughout the period. \n\n"
                     "Comparatively, top Performers are Denmark, Finland, and Switzerland, they consistently have the highest "
                     "scores. Finland and New Zealand show significant improvement over the years, while Switzerland, Norway, "
                     "and Sweden maintain high scores with minor fluctuations. \n")
        st.write("\n")

        # Ladder score over time for lowest 5 countries
        whr_df_grouped = whr_df.groupby(['Country name', 'year'])['Ladder score'].sum().reset_index()
        latest_year = whr_df_grouped['year'].max()
        lowest_10_countries= whr_df_grouped[whr_df_grouped['year'] == latest_year].nsmallest(10, 'Ladder score')[
            'Country name']
        lowest_10_data= whr_df_grouped[whr_df_grouped['Country name'].isin(lowest_10_countries)]

        fig = px.line(lowest_10_data, x='year', y='Ladder score', color='Country name', markers=True,
                      title='Ladder Score Over Time for lowest 10 Countries')

        fig.update_layout(xaxis_title='Year', yaxis_title='Ladder Score', legend_title='Country',
                          legend=dict(x=1, y=1), margin=dict(l=0, r=0, t=30, b=0),
                          xaxis=dict(tickmode='linear', tick0=lowest_10_data['year'].min(), dtick=2))

        st.plotly_chart(fig)

        with st.expander("Interpretation/Insight"):
            st.write("The plot above shows the changes in ladder scores for ten countries with lowest score countries from 2007"
                     " to 2021. The key observation is that each country’s score varies over time, with some showing more volatility"
                     " than others. \n\nAfghanistan shows significant fluctuations over the years. It starts just above 3.5 in "
                     "2007, experiences several ups and downs, and ends below this value in 2021. This indicates a period of "
                     "instability and varying conditions affecting the ladder score.  The decline in Afghanistan’s Ladder score "
                     "suggests worsening well-being, likely influenced by ongoing conflict, political instability, and economic "
                     "challenges. \n\nBotswana’s score remains relatively stable compared to other countries. Though it started "
                     "around the 4.5 mark in 2006 and peaked in 2008 to around 5,5 the highest so far among the group of lowest "
                     "countries, it started  fluctuating in 2010.  However it remained stable since 2016 till 2021 although at the "
                     " range of 3.5 ladder score which is significantly low compared to when the started. This suggests however a "
                     "more consistent environment since 2016. Botswana’s stable Ladder score indicates consistent well-being, possibly "
                     " due to relatively stable governance and economic conditions. \n\nBurundi’s score shows a slight upward trend,"
                     " starting above 3.5 in 2007, decrease to 2,9 in 2014, then came back to 3.5 and remained stable since 2018 over "
                     " time. This indicates a gradual increase in the ladder score, possibly due to stabilizing conditions. This slight "
                     " improvement may reflect gradual socio-economic development and improvements in living conditions "
                     "\n\nHaiti’s score fluctuates but shows a general downward trend, starting above 4 in 2007 and ending around 3.6 "
                     " in 2021. This suggests ongoing challenges affecting the ladder score. Haiti’s fluctuating scores indicate "
                     " instability in well-being, likely due to natural disasters, political commotion, and economic hardships. "
                     "\n\nLesotho scored 4.8 at their peak in 2011, it could be considered somewhat a downward trend, although it "
                     "remains stable since around 2018, with minor fluctuations around the 3.5 mark. Although this may indicate "
                     "a relatively consistent environment over the years. However, the decline in Lesotho’s score suggests deteriorating"
                     "well-being, potentially due to health crises, economic issues, and political instability. \n\nMalawi’s score"
                     "shows a slight upward trend, starting below 4 in 2007,  going to their peak in 2009 with 5.1, with downward"
                     "trend in 2018 by 3.3 score and ending close to 3.6 in 2021. This suggests gradual improvement in the ladder "
                     "score once again after brief decline. The decline and slight improvement in Malawi’s score may indicate slow"
                     "but positive developments in health, education, and economic conditions. But this remains to be observed."
                     "\n\nRwanda’s score shows a noticeable upward trend, although it started above 4 in 2007 and went downward 2012,"
                     "it peaked up in 2017 and continued noticeable upward trend and ended around 3.5 in 2021. This indicates significant "
                     "improvement over the years. This could be attributed to effective governance, economic growth, and social reforms."
                     "\n\nTanzania’s score also shows a slow but steady upward trend, starting below 4 in 2007, with their lowest in 2016 by 2.9 "
                     "score and ending above 3.6 in 2021. This suggests a positive change in the ladder score. The gradual but steady "
                     "improvement in Tanzania’s score suggests steady socio-economic progress and improvements in living standards. "
                     "\n\nYemen’s score shows significant fluctuations, with a general downward trend. It starts above 4.5 in 2007 and ends"
                     " around 3.6 in 2021. This indicates instability and worsening conditions over the years. Yemen’s substantial decline"
                     " in well-being is likely due to the severe impact of ongoing conflict, humanitarian crises, and economic collapse."
                     "\n\nZimbabwe’s score remains relatively stable, with minor fluctuations around the 3.5 mark. This suggests a relatively "
                     "consistent environment over the years. Zimbabwe’s stable but low score indicate persistent challenges in governance, "
                     "economic stability, and social conditions. \n\nComparatively, Rwanda and Tanzania show the most significant improvement "
                     "in ladder scores over the years. Botswana and Zimbabwe have relatively stable scores with minor fluctuations. Afghanistan,"
                     "Haiti, and Yemen show a general downward trend, indicating worsening conditions.")
        st.write("\n")
        # # Ladder score across year
        # fig = px.box(whr_df, x='year', y='Ladder score',
        #              title='Distribution of Ladder Scores Across Years',
        #              labels={'year': 'Year', 'Ladder score': 'Ladder Score'},
        #              color='year',
        #              color_discrete_sequence=px.colors.qualitative.Plotly)
        #
        # fig.update_layout(xaxis_title='Year', yaxis_title='Ladder Score',
        #                   xaxis_tickangle=-45, xaxis_tickmode='linear')
        # st.plotly_chart(fig)

        # Aggregate the data by year and calculate the mean Ladder score
        yearly_ladder_score = whr_df.groupby('year')['Ladder score'].mean().sort_index(ascending=True)
        fig = px.bar(yearly_ladder_score,
                     x=yearly_ladder_score.index,
                     y=yearly_ladder_score.values,
                     title='Average Ladder Scores Across Years',
                     # labels={'x': 'Year', 'y': 'Ladder Score'},
                     color=yearly_ladder_score.values,
                     color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.update_layout(xaxis_title='Year',
                          yaxis_title='Ladder Score',
                          xaxis_tickangle=-45,
                          xaxis=dict(tickmode='linear', dtick=1),
                          margin=dict(l=20, r=20, t=25, b=20),
                          width=600,
                          height=300)
        st.plotly_chart(fig)

        with st.expander("Interpretation/Insight"):
            st.write("Between 2005-2007 the scores are relatively low, around 4, indicating lower levels of well-being. "
                     "\n\nBetween 2008-2012 there was a noticeable increase in scores, suggesting improvements in "
                     "socio-economic conditions or policies. \n\nBetween 2013-2016 the upward trend continued, possibly "
                     "reflecting sustained economic growth or social stability. \n\nWhereas 2017-2021 has the highest scores, "
                     "indicating significant improvements in well-being, possibly due to advancements in healthcare, "
                     "education, and economic opportunities.")
        st.write("\n")

        # Calculate average ladder score by region and year / Sort regions legend by average ladder score in descending order
        avg_ladder_score = whr_df.groupby(['year', 'Regional indicator'])['Ladder score'].mean().reset_index()
        avg_ladder_score_by_region = whr_df.groupby('Regional indicator')['Ladder score'].mean().reset_index()
        sorted_regions = avg_ladder_score_by_region.sort_values(by='Ladder score', ascending=True)['Regional indicator']
        fig = px.line(avg_ladder_score, x='year', y='Ladder score', color='Regional indicator',
                      category_orders={'Regional indicator': sorted_regions},
                      title='Average Ladder Score Over Time by Region',
                      labels={'year': 'Year', 'Ladder score': 'Average Ladder Score'})

        fig.update_layout(
            xaxis=dict(tickmode='linear', tick0=2021, dtick=2),
            legend_title='Region', legend=dict(x=5, y=1, traceorder='reversed'),
            margin=dict(l=20, r=20, t=25, b=0),
            paper_bgcolor='White', plot_bgcolor='#f8f9fa')

        fig.update_traces(mode='lines+markers', marker=dict(size=5))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Interpretation/Insight"):
            st.write("The plot above shows the trends in average ladder scores from 2005 to 2021 for various regions. "
                     "\n\nNorth America and ANZ has consistent high scores, remaining above 7 throughout the period. This "
                     "indicates a stable and high level of well-being or happiness. \n\nWestern Europe is similar to North "
                     "America and ANZ, with high scores but a slight decline towards the end. They are generally stable "
                     "but with a minor downward trend in recent years. \n\nLatin America and Caribbean started higher than "
                     "Central and Eastern Europe but experiences more fluctuations, ending slightly lower. This indicates "
                     "varying conditions affecting well-being over time. \n\nCentral and Eastern Europe, shows an overall "
                     "increasing trend. This indicates gradual improvement in well-being or happiness. \n\nEast Asia started "
                     "just below Western Europe and ends slightly higher, However the are generally stable with minor "
                     "fluctuations. \n\nMiddle East and North Africa, shows more variation with an overall downward trend, "
                     "ending just above a score of 5. This indicates worsening conditions over time. \n\nCommonwealth of "
                     "Independent States, shows upward trend, starting below Middle East and North Africa but ending "
                     "above them near a score of 6. This shows significant improvement over time. \n\nSoutheast Asia, shows "
                     "upward trend, starting at about a score of 5 and ending near 5.75. this indicates steady improvement "
                     "in well-being. \n\nSouth Asia started at the lowest point near a score of 4 but shows improvement, "
                     "ending near 4.75. This indicates gradual improvement over time. \n\nSub-Saharan Africa about a score of 4, "
                     "started low, but ends slightly higher around a score of 4.5. This shows some improvement over time."
                     "\n\nComparatively, top performers are North America and ANZ, Western Europe. While improving Regions are"
                     " Central and Eastern Europe, Commonwealth of Independent States, Southeast Asia, South Asia, Sub-Saharan "
                     "Africa. On other hand the declining Regions are Middle East and North Africa.")
        st.write("\n")

        # Bar plot of Logged GDP per capita by region
        regional_gdp = whr_df.groupby('Regional indicator')['Logged GDP per capita'].mean().sort_values(
            ascending=False).reset_index()
        fig = px.bar(regional_gdp, x='Regional indicator', y='Logged GDP per capita',
                     title='Logged GDP per capita by Region',
                     labels={'Regional indicator': 'Region', 'Logged GDP per capita': 'Logged GDP per capita'},
                     color='Logged GDP per capita',
                     color_continuous_scale='plasma')
        fig.update_layout(xaxis_title='Region', yaxis_title='Logged GDP per capita',
                          xaxis_tickangle=-45, xaxis_tickmode='array',
                          xaxis_tickvals=regional_gdp['Regional indicator'],
                          xaxis_ticktext=regional_gdp['Regional indicator'],
                          margin=dict(l=20, r=20, t=25, b=20))

        st.plotly_chart(fig)

        with st.expander("Interpretation/Insight"):
            st.write("The bar chart above provides a visual representation of the economic status of various global regions "
                     "through their logged GDP per capita. The key observations is that Western Europe has the highest logged GDP per capita:among all regions, around 10, this can be interpreted"
                     " as indicating a very high level of economic development and wealth."
                     "\n\nNorth America and ANZ (Australia and New Zealand) is slightly below Western Europe on logged GDP per "
                     "capita, close to 10. this also represents a high level of economic development. \n\nEast Asia has "
                     "logged GDP per capita: Around 9. this indicates a significant economic growth, particularly in "
                     "countries like China and Japan. \n\nCentral and Eastern Europe has logged GDP per capita score around 8, "
                     "indicating a moderate economic development, with some countries transitioning from planned to market "
                     "economies. \n\nLatin America and Caribbean has logged GDP per capita of slightly below 8, which shows "
                     "varied economic performance, with some countries experiencing growth while others face challenges."
                     "\n\nMiddle East and North Africa has  logged GDP per capita, around 7.5, which indicates wealth from oil-rich "
                     "countries, but also economic disparities within the region. \n\nSoutheast Asia has logged GDP per capita, around 7. "
                     "Indicating emerging economies with rapid growth in countries like Indonesia and Vietnam. \n\nSouth Asia has "
                     "a logged GDP per capita, slightly above 6, Reflecting lower economic development, with countries like "
                     "India showing significant growth potential. \n\nSub-Saharan Africa has lowest logged GDP per capita, around "
                     "6. which indicates that the region faces significant economic challenges and lower levels of development.")
        st.write("\n")

        # Calculate the mean Social support by region
        regional_ss = whr_df.groupby('Regional indicator')['Social support'].mean().sort_values(
            ascending=False).reset_index()

        fig = px.bar(regional_ss, x='Regional indicator', y='Social support',
                     title='Social support by Region',
                     labels={'Regional indicator': 'Region', 'Social support': 'Social support'},
                     color='Social support',
                     color_continuous_scale='viridis')

        fig.update_layout(xaxis_title='Region', yaxis_title='Social support',
                          xaxis_tickangle=-45, xaxis_tickmode='array',
                          xaxis_tickvals=regional_ss['Regional indicator'],
                          xaxis_ticktext=regional_ss['Regional indicator'],
                          margin=dict(l=20, r=20, t=25, b=20))

        st.plotly_chart(fig)

        with st.expander("Interpretation/Insight"):
            st.write("The key observation is that North America & ANZ region has the highest level of social support, with "
                     "values close to 0.9. This indicates strong community and social networks. \n\nWestern Europe also shows "
                     "high levels of social support, slightly below North America & ANZ with around 0.87, indicating robust "
                     "social structures. \n\nCentral & Eastern Europe, East Asia and Latin America & Caribbean region shows "
                     "moderate levels of social support around 0.82, indicating developing social networks and strong community"
                     "ties.  \n\nCommonwealth of Independent States region also falls in the moderate range around 0.81, reflecting "
                     "developing social structures. \n\nSoutheast Asia also shows  moderate levels of social support around 0.8, "
                     "indicating developing community networks. \n\nSouth Asia and Middle East & North Africa falls also within moderate "
                     "range around 0.78, indicating varying social conditions, but developing community networks and  ties. "
                     "\n\nSub-Saharan Africa and South Asia region has the lowest level of social support, just above 0.7 and 6.5 "
                     "indicating significant challenges in social structures. \n\nComparatively, top Regions are "
                     "North America & ANZ and Western Europe. Moderate Regions are Latin America & Caribbean, "
                     "Central & Eastern Europe, East Asia, Commonwealth of Independent States, Southeast Asia, "
                     "Middle East & North Africa. Challenged Regions are  Sub-Saharan Africa and South Asia.")
        st.write("\n")

        # Calculate the mean Healthy Life Expectancy by region
        regional_health = whr_df.groupby('Regional indicator')['Healthy life expectancy'].mean().sort_values(
            ascending=False).reset_index()

        fig = px.bar(regional_health, x='Regional indicator', y='Healthy life expectancy',
                     title='Healthy Life Expectancy by Region',
                     labels={'Regional indicator': 'Region', 'Healthy life expectancy': 'Healthy Life Expectancy'},
                     color='Healthy life expectancy',
                     color_continuous_scale='plasma')

        # Update layout for better readability
        fig.update_layout(xaxis_title='Region', yaxis_title='Healthy Life Expectancy',
                          xaxis_tickangle=-45, xaxis_tickmode='array',
                          xaxis_tickvals=regional_health['Regional indicator'],
                          xaxis_ticktext=regional_health['Regional indicator'],
                          margin=dict(l=20, r=20, t=25, b=0))

        st.plotly_chart(fig)

        with st.expander("Interpretation/Insight"):
            st.write("A comprehensive interpretation indicates that Western Europe has the highest healthy life expectancy, "
                     "close to 70 years. This points to excellent healthcare systems, high living standards, and overall "
                     "well-being. North America & ANZ region also shows high healthy life expectancy, slightly below Western "
                     "Europe, reflecting strong healthcare and quality of life. East Asia has a moderate healthy life expectancy"
                     "around 65 years, indicating good healthcare and living conditions. Latin America & Caribbean region shows "
                     "a moderate healthy life expectancy slightly above 60 years, reflecting moderate healthcare and living standards. "
                     "Central & Eastern Europe is similar to Latin America & Caribbean, this region has a healthy life expectancy around "
                     "60 years. Middle East & North Africa region shows a healthy life expectancy slightly below 60 years, indicating "
                     "varying healthcare and living conditions. South Asia has a healthy life expectancy around 55 years, reflecting "
                     "challenges in healthcare and living standards. Sub-Saharan Africa region has the lowest healthy life "
                     "expectancy, slightly above 50 years, indicating significant healthcare challenges and lower living "
                     "standards. Comparatively top Regions are Western Europe, North America & ANZ. East Asia, "
                     "Latin America & Caribbean Central & Eastern Europe, Middle East & North Africa are are Moderate Regions,"
                     " while South Asia, Sub-Saharan Africa are the Challenged Regions.")
        st.write("\n")

        # Calculate the mean Freedom to make life choices by region
        regional_fmlc = whr_df.groupby('Regional indicator')['Freedom to make life choices'].mean().sort_values(
            ascending=False).reset_index()

        # Create a bar plot using Plotly Express
        fig = px.bar(regional_fmlc, x='Regional indicator', y='Freedom to make life choices',
                     title='Freedom to make life choices by Region',
                     labels={'Regional indicator': 'Region',
                             'Freedom to make life choices': 'Freedom to make life choices'},
                     color='Freedom to make life choices',
                     color_continuous_scale='viridis')

        fig.update_layout(xaxis_title='Region', yaxis_title='Freedom to make life choices',
                          xaxis_tickangle=-45, xaxis_tickmode='array',
                          xaxis_tickvals=regional_fmlc['Regional indicator'],
                          xaxis_ticktext=regional_fmlc['Regional indicator'],
                          margin=dict(l=20, r=20, t=25, b=0))

        st.plotly_chart(fig)
        plt.show()

        with st.expander("Interpretation/Insight"):
            st.write("The Key insight reveals that North America & ANZ has the highest level of freedom to make life choices, "
                     "with values above 0.8. This indicates a high degree of individual autonomy and supportive social and "
                     "political environments. What is outstanding here compared to other features of happiness is that "
                     "\n\nSoutheast Asia has a high levels of freedom to make life choice with values close to 0.8, a little bit "
                     "more than Western Europe, indicating a high degrees of individual autonomy across different countries. "
                     "\n\nWestern Europe also shows high levels of freedom, slightly below North America & ANZ and Southeast Asia,"
                     "reflecting strong individual rights and freedoms. \n\nLatin America & Caribbean region and East Asia has "
                     "moderate to high levels of freedom between 0.7.8 and 0.75 indicating a relatively supportive environment "
                     "for personal autonomy. \n\nCommonwealth of Independent States region also falls in the moderate range of 0.7,"
                     "reflecting mixed levels of freedom to make life choices. \n\nSub-Saharan Africa and South Asia has lower levels "
                     "of freedom around 0.65, indicating significant challenges in social and political environments that support "
                     "individual autonomy. \n\nCentral & Eastern Europe and Middle East & North Africa region has the lowest level of "
                     "freedom to make life choices, just above 0.6, reflecting restrictive social and political conditions."
                     "\n\nComparatively, top Regions are North America & ANZ, Southeast Asia and Western Europe. Moderate Regions "
                     "are Latin America & Caribbean, East Asia and Commonwealth of Independent States. Whereas, challenged Regions "
                     "are Sub-Saharan Africa, South Asia, Central & Eastern Europe and Middle East & North Africa. ")
        st.write("\n")

        # Calculate the mean Generosity by region
        regional_generosity = whr_df.groupby('Regional indicator')['Generosity'].mean().sort_values(
            ascending=False).reset_index()

        fig = px.bar(regional_generosity, x='Regional indicator', y='Generosity',
                     title='Generosity by Region',
                     labels={'Regional indicator': 'Region', 'Generosity': 'Generosity'},
                     color='Generosity',
                     color_continuous_scale='plasma')

        # Update layout for better readability
        fig.update_layout(xaxis_title='Region', yaxis_title='Generosity',
                          xaxis_tickangle=-45, xaxis_tickmode='array',
                          xaxis_tickvals=regional_generosity['Regional indicator'],
                          xaxis_ticktext=regional_generosity['Regional indicator'],
                          margin=dict(l=20, r=20, t=25, b=0))

        st.plotly_chart(fig)

        with st.expander("Interpretation/Insight"):
            st.write("Further observations shows that North America and ANZ has the highest level of generosity, with values "
                     "close to 0.2. This indicates a strong culture of charitable giving and social support. \n\nThey are followed"
                     " by Southeast Asia with high levels of generosity, slightly below North America and ANZ, reflecting a "
                     "significant inclination towards charitable activities. \n\nSouth Asia and Western Europe has moderate levels"
                     " of generosity, indicating a balanced approach to charitable giving. \n\nSub-Saharan Africa shows a near 0 "
                     "value, indicating some level of charitable activities, though not as pronounced as in other regions above it. "
                     "\n\nCentral and Eastern Europe has a negative value, indicating lower levels of charitable giving or social support. "
                     "\n\nLatin America and Caribbean is similar to Central and Eastern Europe, this region shows negative values, reflecting"
                     "lower levels of generosity. \n\nMiddle East and North Africa has more negative values compared to Latin America and Caribbean, "
                     "indicating even lower levels of charitable activities. \n\nCentral and Eastern Europe and Commonwealth of Independent States "
                     "has the lowest level of generosity, with the most negative value, indicating significant challenges in charitable giving "
                     "and social support. \n\nComparatively, top Regions are North America and ANZ, Southeast Asia. While the Moderate Regions are "
                     "Western Europe, South Asia. Whereas the Challenged Regions are Sub-Saharan Africa, East Asia, Central and Eastern Europe, "
                     "Latin America and Caribbean, Middle East and North Africa and Commonwealth of Independent States.")
        st.write("\n")

        # Calculate the mean Perceptions of corruption by region
        regional_pc = whr_df.groupby('Regional indicator')['Perceptions of corruption'].mean().sort_values(
            ascending=False).reset_index()

        fig = px.bar(regional_pc, x='Regional indicator', y='Perceptions of corruption',
                     title='Perceptions of corruption by Region',
                     labels={'Regional indicator': 'Region', 'Perceptions of corruption': 'Perceptions of corruption'},
                     color='Perceptions of corruption',
                     color_continuous_scale='viridis')

        fig.update_layout(xaxis_title='Region', yaxis_title='Perceptions of corruption',
                          xaxis_tickangle=-45, xaxis_tickmode='array',
                          xaxis_tickvals=regional_pc['Regional indicator'],
                          xaxis_ticktext=regional_pc['Regional indicator'],
                          margin=dict(l=20, r=20, t=20, b=20))

        st.plotly_chart(fig)

        with st.expander("Interpretation/Insight"):
            st.write("The bar chart above provides a visual comparison of perceived corruption levels across different global "
                     "regions. This can be interpreted that North America and ANZ has the lowest levels of perceived corruption,"
                     " indicating strong governance and effective anti-corruption measures. \n\nFollowed by Western Europe which "
                     "shows low levels of perceived corruption, reflecting strong institutions and effective anti-corruption measures. "
                     "\n\nEast Asia and South East Asia  has moderate level of perceived corruption, indicating varying degrees of "
                     "governance challenge. \n\nMiddle East & North Africa and Commonwealth of Independent States (CIS), has a moderate "
                     "to high levels of perceived corruption, indicating ongoing challenges in governance and transparency."
                     "\n\nSub-Saharan Africa is similar to the Commonwealth of Independent States (CIS), with high to moderate "
                     "levels of perceived corruption, indicating widespread governance and transparency issues. \n\nSouth Asia and "
                     "Latin America & Caribbean shows high levels of perceived corruption, reflecting significant governance "
                     "challenges. \n\nCentral and Eastern Europe shows the highest levels of perceived corruption, reflecting "
                     "significant governance issues. \n\nComparatively, regions with low perceptions of corruption are North "
                     "America and ANZ and Western Europe. Regions with moderate perceptions of corruption are East Asia and "
                     "Southeast Asia. While Regions with moderate to high perceptions of corruption are Middle East & North "
                     "Africa and Commonwealth of Independent States (CIS) and Sub-Saharan Africa. The regions with high "
                     "perceptions of Corruption are South Asia and Latin America & Caribbean. Whereas the regions with the "
                     "highest perception of corruption is Central and Eastern Europe.")
        st.write("\n")

    st.markdown("<hr class='custom-hr-dc'>", unsafe_allow_html=True)

    # *** Feature distribution plots *** #

    if st.checkbox("Feature distribution"):

        # Drop unnecessary columns not needed for feature for now
        whr_df_cor = whr_df.drop(['Country name', 'year', 'Regional indicator'], axis=1)

        features = whr_df_cor.columns
        selected_features = st.multiselect("Select features", options=features, default=features)

        bins = st.slider("Number of bins", min_value=2, max_value=50, value=60, key="bins_slider")

        colors = ['orange', 'red', 'blue', 'purple', 'gold', 'brown', 'pink', 'green']

        for i, feature in enumerate(selected_features):
            fig = go.Figure()

            # Add histogram trace
            hist_data = whr_df_cor[feature]
            hist_counts, hist_edges = np.histogram(hist_data, bins=bins)
            fig.add_trace(go.Histogram(x=hist_data, nbinsx=bins, name='Histogram',
                                       marker_color=colors[i % len(colors)]))

            # Add KDE trace
            kde = gaussian_kde(hist_data)
            x_vals = np.linspace(hist_data.min(), hist_data.max(), 1000)
            kde_vals = kde(x_vals)

            # Normalize KDE to match histogram counts
            kde_vals_normalized = kde_vals * hist_counts.max() / kde_vals.max()
            fig.add_trace(go.Scatter(x=x_vals, y=kde_vals_normalized, mode='lines', name='KDE',
                                     line=dict(color=colors[(i + 1) % len(colors)])))

            fig.update_layout(
                title=f'Histogram and KDE of {feature}',
                xaxis_title=feature,
                yaxis_title='Frequency count',
                bargap=0.2,
                bargroupgap=0.1
            )

            st.plotly_chart(fig)

        with st.expander("Interpretation/Insight"):
            st.write("The Histogram and KDE of Ladder score shows the distribution frequency count of ladder scores, with "
                     "the x-axis ranging from approximately 2 to 8.  Frequency Peaks, in other words the highest frequency "
                     "count is around the ladder score of 5, indicating that this score is the most common in the dataset. "
                     "Whereas the Kernel Density Estimate (KDE) Plot, which provides a smooth curve, in this sense the red "
                     "KDE line provides a smooth estimate of the probability density function of the ladder score variable. "
                     "The  KDE plot in this context shows that the values are most concentrated around the ladder score of "
                     "5, with a gradual decrease in density as the scores move away from this central value. In conclusion, "
                     "the most common scores are around 5, with fewer occurrences as the scores move towards the extremes. "
                     "This indicates a central tendency around the middle of the scale, with a relatively normal distribution. "
                     "\n\nThe  Histogram and KDE of Logged GDP per capita shows the distribution frequency count of "
                     "logged GDP per capita values, with the x-axis ranging from approximately 6 to 11.5. The highest "
                     "frequency count is around the logged GDP per capita value of 10, indicating that this value is "
                     "the most common in the dataset.  The blue KDE line provides a smooth estimate of the probability "
                     "density function of the logged GDP per capita data. The KDE plot shows that the values are most "
                     "concentrated around the logged GDP per capita value of 10, with a gradual decrease in density as "
                     "the values move away from this central point. In conclusion, the most common values are around 10, "
                     "with fewer occurrences as the values move towards the extremes. This indicates a central tendency "
                     "around the middle of the scale, with a relatively normal distribution. \n\nThe Histogram and KDE of "
                     "Social support display the frequency count of social support values, with the x-axis ranging from "
                     "0.3 to 1. The highest frequency count is around the social support value of 0.9, indicating that "
                     "these values are the most common in the dataset. The red KDE line provides a smooth estimate of "
                     "the probability density function of the social support data. The KDE plot shows that the values "
                     "are most concentrated around the social support value of 0.9, with a sharp peak indicating a high "
                     "density of data points in this range. In conclusion, the most common values are around 0.9, "
                     "indicating that high levels of social support are prevalent in the dataset. This suggests a strong "
                     "presence of social networks and community support within the sample. \n\nThe Histogram and KDE of Healthy "
                     "life expectancy shows the frequency count of healthy life expectancy values, with the x-axis ranging "
                     "from approximately 40 to 77 years. The highest frequency count  is around the healthy life expectancy "
                     "value of 66 years, indicating that this value is the most common in the dataset. The yellow KDE line "
                     "provides a smooth estimate of the probability density function of the healthy life expectancy data. "
                     "The KDE plot shows that the values are most concentrated around the healthy life expectancy value of "
                     "66 years, with a slight right-skew, indicating a higher density of data points in this range. In "
                     "conclusion, the most common values are around 66 years, indicating that this is the typical healthy "
                     "life expectancy in the dataset. The slight right-skew suggests that there are more observations below "
                     "the mode. \n\nThe Histogram and KDE of Freedom to make life choices reveals the frequency count of "
                     "Freedom to make life choices values, with the x-axis ranging from 0.2 to 1. The peak of the frequency "
                     "count is around the value of 07 and 0.9, indicating that this value is the most common in the dataset. "
                     "The red KDE line provides a smooth estimate of the probability density function of the “Freedom to make "
                     "life choices” data. The KDE plot shows that the values are most concentrated around 0.7 and 0.9, "
                     "indicating a high density of data points in this range. In conclusion, the most common values are "
                     "around 0.7 and 0.9, indicating that a high level of freedom to make life choices is prevalent in the "
                     "dataset. This suggests that individuals in the sample generally feel they have significant autonomy "
                     "in their decisions. \n\nThe Histogram and KDE of Generosity shows the frequency count of Generosity values, "
                     "with the x-axis ranging from approximately -0.3 to 0.6.  The highest frequency count is around the "
                     "Generosity value of -0.06, indicating that this value is the most common in the dataset. The pink KDE "
                     "line provides a smooth estimate of the probability density function of the Generosity data. The KDE "
                     "plot shows that the values are most concentrated around the Generosity value of -0.06, with a peak "
                     "indicating a high density of data points in this range. In conclusion, the most common values "
                     "are around -0.06, indicating that moderate levels of generosity are prevalent in the dataset. This "
                     "suggests that individuals in the sample generally exhibit low generosity.  The slight left-skew "
                     "suggests that there are more on the negative side. \n\nThe Histogram and KDE of Perceptions of "
                     "corruption shows the frequency count of perceptions of corruption values, with the x-axis ranging "
                     "from 0.03 to 1. The highest frequency count is around the perceptions of corruption value of 0.8, "
                     "indicating that these values are the most common in the dataset. The green KDE line provides a smooth "
                     "estimate of the probability density function of the perceptions of corruption data. The KDE plot "
                     "shows that the values are most concentrated around the perceptions of corruption value of 0.8, "
                     "indicating a high density of data points in this range. In conclusion, the most common values are "
                     "around 0.8, indicating that high perceptions of corruption are prevalent in the dataset. This "
                     "suggests that individuals in the sample generally perceive a significant level of corruption.")

    st.markdown("<hr class='custom-hr-dc'>", unsafe_allow_html=True)

    # The correlation Table of the Variables and there importance to the World happiness
    whr_df_cor = whr_df.drop(['Country name', 'year', 'Regional indicator'], axis=1)
    feature_correlation = whr_df_cor.corr()

    if st.checkbox("Feature Correlation Table "):
        st.dataframe(feature_correlation)

    if st.checkbox("Feature Correlation Heatmap"):
        colormap = st.selectbox("Select a colormap", ["Plasma", "Viridis", "Cividis", "Inferno", "Magma"])
        fig = px.imshow(feature_correlation, color_continuous_scale=colormap, text_auto=True)
        fig.update_traces(textfont_size=10)
        st.plotly_chart(fig)

        with st.expander("Interpretation/Insight"):
            st.write("The feature correlation (Table and Heatmap Matrix) visually represents the correlation coefficients between "
                 "various well-being and societal metrics. Key insights from it is as follows: \n\nLadder Score has strong "
                 "positive correlations with Logged GDP per capita (0.78), this indicates that higher economic performance is "
                 " strongly associated with higher ladder scores. Social Support (0.70) suggests that strong social networks "
                 "are linked to higher ladder scores. Healthy Life Expectancy (0.74) shows that better health outcomes are "
                 "associated with higher ladder scores. Freedom to Make Life Choices (0.52) indicates that greater personal "
                 "freedom is linked to higher ladder scores. While Ladder Score has a low correlation with Generosity and "
                 "negative correlation with  perceptions of Corruption (-0.42), this suggests that higher perceptions of "
                 "corruption are associated with lower ladder scores. \n\nLogged GDP per capita has strong positive correlations "
                 " with healthy life expectancy (0.84), this indicates that higher economic performance is strongly linked to "
                 " better health outcomes. Logged GDP per capita also has positive correlation with social support (0.69), "
                 "which suggests that economic performance is positively associated with social support. Logged GDP per capita "
                 " has negative correlation with perceptions of corruption (-0.33), which indicates that higher economic "
                 "performance is associated with lower perceptions of corruption. \n\nSocial support has strong positive "
                 "correlations with healthy life expectancy (0.61), which, indicates that strong social networks are "
                 "linked to better health outcomes. \n\nFreedom to make life choices (0.69) is positively correlated to Logged "
                 "GDP per capita  and social support (0.61), which suggests that better health outcomes and social support is "
                 "positively associated with personal freedom. Freedom to make life choices has moderate correlations with "
                 "Social support (0.41), which indicates that better Social support are slightly linked to personal freedom. "
                 "Freedom to make life choices has negative correlation with perceptions of corruption (-0.47), which suggests "
                 "that greater personal freedom is associated with lower perceptions of corruption.  \n\nGenerosity shows weak "
                 "correlations with other metrics, indicating it may be influenced by different or other factors. \n\nIn "
                 "conclusion, the feature correlation highlights the interrelationships between various well-being and societal "
                 "metrics. Strong positive correlations suggest that economic performance, social support, health outcomes, and "
                 "personal freedom are closely linked to higher well-being. In the opposite, higher perceptions of corruption "
                 "are associated with lower well-being and personal freedom.")
            
# *** Machine Learning *** #

elif page == pages[4]:

    st.write("<h2>Machine Learning: Model Evaluation (STEP 3)</h2>", unsafe_allow_html=True)

    # Read and extract the clean dataset for machine learning
    whr_df = pd.read_csv("cleaned_whr_data.csv")

    # Define the features and target
    X_features = whr_df.drop(['Ladder score', 'Country name', 'year', 'Regional indicator'], axis=1)
    y_target = whr_df['Ladder score']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2,
                                                        random_state=42)  # stratify=X['Regional indicator']

    # Create a preprocessor for numerical and categorical data
    modelLR = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    modelDTR = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('regressor', DecisionTreeRegressor(random_state=42))
    ])

    modelRFR = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Fit the pipeline on the training data
    modelLR.fit(X_train, y_train)
    modelDTR.fit(X_train, y_train)
    modelRFR.fit(X_train, y_train)

    # Save the trained models to a file
    joblib.dump(modelLR, 'trained_modelLR.pkl', compress='gzip')
    joblib.dump(modelDTR, 'trained_modelDTR.pkl', compress='gzip')
    joblib.dump(modelRFR, 'trained_modelRFR.pkl', compress='gzip')

    # Make predictions on the test data display the predictions result
    predictionsLR = modelLR.predict(X_test)
    predictionsDTR = modelDTR.predict(X_test)
    predictionsRFR = modelRFR.predict(X_test)

    # *** Model Evaluation *** #

    # I used various regression evaluation metric, in other to compare how the models performed on each

    # Calculate evaluation metrics for LinearRegression
    evaluation_result_LR = {'MAE': mean_absolute_error(y_test, predictionsLR),
                            'MSE': mean_squared_error(y_test, predictionsLR),
                            'RMSE': np.sqrt(mean_squared_error(y_test, predictionsLR)),
                            'R-squared (R²) Score': r2_score(y_test, predictionsLR)
                            }

    # Calculate evaluation metrics for DecisionTreeRegressor
    evaluation_result_DTR = {'MAE': mean_absolute_error(y_test, predictionsDTR),
                             'MSE': mean_squared_error(y_test, predictionsDTR),
                             'RMSE': np.sqrt(mean_squared_error(y_test, predictionsDTR)),
                             'R-squared (R²) Score': r2_score(y_test, predictionsDTR)
                             }

    # Calculate evaluation metrics for RandomForestRegressor
    evaluation_result_RFR = {'MAE': mean_absolute_error(y_test, predictionsRFR),
                             'MSE': mean_squared_error(y_test, predictionsRFR),
                             'RMSE': np.sqrt(mean_squared_error(y_test, predictionsRFR)),
                             'R-squared (R²) Score': r2_score(y_test, predictionsRFR)
                             }

    evaluation_results = {'Linear Regressor': evaluation_result_LR,
                          'Decision Tree Regressor': evaluation_result_DTR,
                          'Random Forest Regressor': evaluation_result_RFR
                          }

    evaluation_results = pd.DataFrame(evaluation_results)
    evaluation_results = evaluation_results

    # Display specific model results
    model_option = st.selectbox(
        'Select Specific Model Result (80/20 Split)',
        ("... (De)select Specific Model Result (80/20 Split)", "Linear Regression", "Decision Tree Regression",
         "Random Forest Regression")
                               )

    if model_option == "Linear Regression":
        st.write(evaluation_results.iloc[0])
    elif model_option == "Decision Tree Regression":
        st.write(evaluation_results.iloc[1])
    elif model_option == "Random Forest Regression":
        st.write(evaluation_results.iloc[2])

    if st.checkbox("Show All Model Results (80/20 Split)"):
        st.write(evaluation_results)

#    st.markdown("<hr class='custom-hr-dc'>", unsafe_allow_html=True)

    if st.checkbox("Visualize Model Results (80/20 Split)"):
        evaluation_results = pd.DataFrame(evaluation_results)
        evaluation_results = evaluation_results

        models = ['Linear Regressor', 'Decision Tree Regressor', 'Random Forest Regressor']
        metrics = ['MAE', 'MSE', 'RMSE', 'R-squared']

        # # Create a subplot figure with 2 rows and 2 columns, adding vertical spacing and specifying subplot sizes
        fig = make_subplots(
            rows=2, cols=2, subplot_titles=metrics, vertical_spacing=0.2
        )

        # Define colors for models
        colors = ['blue', 'orange', 'green']

        # Iterate through metrics
        for i, metric in enumerate(metrics):
            row = i // 2 + 1
            col = i % 2 + 1
            for j, model in enumerate(models):
                value = evaluation_results[model].iloc[i]  # Extract the value
                fig.add_trace(
                    go.Bar(
                        x=[model],
                        # y=[value],
                        y=[f'{value:.2f}'],
                        name=model,
                        marker_color=colors[j],
                        text=f'{value:.4f}',
                        textposition='outside'
                    ),
                    row=row, col=col
                )

            fig.update_xaxes(title_text='', row=row, col=col) # Model
            fig.update_yaxes(title_text='', row=row) # Metrics

        fig.update_layout(
            title_text='', showlegend=False, # Model Evaluation Metrics
            height=900, width=600,
            plot_bgcolor='White', paper_bgcolor='White',
            margin=dict(l=0, r=0, t=20, b=0))

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='custom-hr-dc'>", unsafe_allow_html=True)

    # Calculate the prediction rates
    overall_prediction_rate_df = pd.DataFrame({'Actual Value': y_test,
                                 'Prediction Rate (%) LR': 100 * (1 - abs(y_test - predictionsLR) / y_test),
                                 'Prediction Rate (%) DTR': 100 * (1 - abs(y_test - predictionsDTR) / y_test),
                                 'Prediction Rate (%) RFR': 100 * (1 - abs(y_test - predictionsRFR) / y_test)})

    if st.checkbox("Overall Prediction Rates"):
        # Calculate the overall prediction rate in percentage for each model
        mean_prediction_rates = {
            'Prediction Rate Linear Regression': overall_prediction_rate_df['Prediction Rate (%) LR'].mean(),
            'Prediction Rate Decision Tree Regression': overall_prediction_rate_df['Prediction Rate (%) DTR'].mean(),
            'Prediction Rate Random Forest Regressor': overall_prediction_rate_df['Prediction Rate (%) RFR'].mean()
        }

        # Create a list of dictionaries for the dataframe
        formatted_rates = [{'Models': model, 'Rates in Percentage (%)': f'{rate:.2f}%'} for model, rate in
                           mean_prediction_rates.items()]

        # Convert the list to a dataframe
        mean_prediction_rates_df = pd.DataFrame(formatted_rates)

        # Display the dataframe
        st.dataframe(mean_prediction_rates_df)

    if st.checkbox("Actual Value and Prediction Rate in Percentage"):
        st.dataframe(overall_prediction_rate_df)

    if st.checkbox("Actual and Predicted Value (80/20 Split)"):
        # Create a DataFrame to compare actual and predicted values

        actual_vs_predicted_80_20_split = pd.DataFrame({'Actual Value': y_test,
                                                        'Predicted LR Value': predictionsLR,
                                                        'Predicted DTR Value': predictionsDTR,
                                                        'Predicted RFR Value': predictionsRFR
                                                        })
        st.write(actual_vs_predicted_80_20_split)

    if st.checkbox("Visualize the Actual vs. Predicted Values"):

        # Plot actual vs. predicted values for Linear Regressor
        fig_lr = px.scatter(x=y_test, y=predictionsLR,
                            labels={'x': 'Actual Values', 'y': 'Predicted Values Linear Regressor'},
                            title='Actual vs. Predicted Values - Linear Regressor')
        fig_lr.add_scatter(x=y_test, y=predictionsLR, mode='markers', name='Predicted',
                           marker=dict(color='blue', size=5))
        fig_lr.add_scatter(x=y_test, y=y_test, mode='markers', name='Actual', marker=dict(color='green', size=8))
        fig_lr.add_scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines',
                           name='Ideal Fit', line=dict(color='red', width=2))
        fig_lr.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_lr)

        # Plot actual vs. predicted values for Decision Tree Regressor
        fig_dtr = px.scatter(x=y_test, y=predictionsDTR,
                             labels={'x': 'Actual Values', 'y': 'Predicted Values Decision Tree Regressor'},
                             title='Actual vs. Predicted Values - Decision Tree Regressor')
        fig_dtr.add_scatter(x=y_test, y=predictionsDTR, mode='markers', name='Predicted',
                            marker=dict(color='orange', size=5))
        fig_dtr.add_scatter(x=y_test, y=y_test, mode='markers', name='Actual', marker=dict(color='green', size=8))
        fig_dtr.add_scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines',
                            name='Ideal Fit', line=dict(color='red', width=2))
        fig_dtr.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_dtr)

        # Plot actual vs. predicted values for Random Forest Regressor
        fig_rfr = px.scatter(x=y_test, y=predictionsRFR,
                             labels={'x': 'Actual Values', 'y': 'Predicted Values Random Forest Regressor'},
                             title='Actual vs. Predicted Values - Random Forest Regressor')
        fig_rfr.add_scatter(x=y_test, y=predictionsRFR, mode='markers', name='Predicted',
                            marker=dict(color='purple', size=5))
        fig_rfr.add_scatter(x=y_test, y=y_test, mode='markers', name='Actual', marker=dict(color='green', size=8))
        fig_rfr.add_scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode='lines',
                            name='Ideal Fit', line=dict(color='red', width=2))
        fig_rfr.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_rfr)

    if st.checkbox("Compare Countries Actual and Predicted Ladder Scores"):

        # Make a copy of the df, this will ensure the original data remains unchanged while working with the predictions
        whr_df_copy = whr_df.copy()

        # Make predictions on the test data
        predicted_score_LR = modelLR.predict(X_features)
        predicted_score_DTR = modelDTR.predict(X_features)
        predicted_score_RFR = modelRFR.predict(X_features)

        # Add predicted scores to the copied dataframe
        whr_df_copy['Predicted Ladder score LR'] = predicted_score_LR
        whr_df_copy['Predicted Ladder score DTR'] = predicted_score_DTR
        whr_df_copy['Predicted Ladder score RFR'] = predicted_score_RFR

        # Remove duplicates based on 'Country name'
        whr_df_copy = whr_df_copy.drop_duplicates(subset=['Country name'])

        # Group by top ten countries based on actual 'Ladder score'
        top_ten_countries = whr_df_copy.nlargest(10, 'Ladder score')
        compare_predicted_scores = top_ten_countries[['Country name', 'Ladder score',
                                                      'Predicted Ladder score LR',
                                                      'Predicted Ladder score DTR',
                                                      'Predicted Ladder score RFR']]

        st.write("Model prediction Comparison")
        st.dataframe(compare_predicted_scores)

    st.markdown("<hr class='custom-hr-dc'>", unsafe_allow_html=True)

    # ********* Identify Key Factors - Feature Importance  ********* #

    if st.checkbox("Identify Key Factors - Feature Importance"):

        # Linear Regression Feature Importance
        linear_regressor = modelLR.named_steps['regressor']
        linear_feature_importance = pd.DataFrame(
            {'Feature': X_test.columns, 'Importance': linear_regressor.coef_}).sort_values(by='Importance',
                                                                                           ascending=False)

        # Decision Tree Regressor Feature Importance
        decision_tree_regressor = modelDTR.named_steps['regressor']
        decision_tree_feature_importance = pd.DataFrame(
            {'Feature': X_test.columns, 'Importance': decision_tree_regressor.feature_importances_}).sort_values(
            by='Importance', ascending=False)

        # Random Forest Regressor Feature Importance
        random_forest_regressor = modelRFR.named_steps['regressor']
        random_forest_feature_importance = pd.DataFrame(
            {'Feature': X_test.columns, 'Importance': random_forest_regressor.feature_importances_}).sort_values(
            by='Importance', ascending=False)

        # Create a dictionary of Feature Importance DataFrames
        FeatureImportance = {
            'Linear Regression Feature Importance': linear_feature_importance,
            'DecisionTree Regressor Feature Importance': decision_tree_feature_importance,
            'RandomForest Regressor Feature Importance': random_forest_feature_importance
        }

        st.write('**( a. Feature Importance Table )**')
        # Display the feature importance result table for all models
        for model, dataframe in FeatureImportance.items():
            st.write(f"\n{model}:")
            st.write(dataframe.T)

        st.write('**( b. Feature Importance Plot )**')
        # Plotting the feature importance for Linear Regression Model
        fig_lr = px.bar(linear_feature_importance, x='Importance', y='Feature', orientation='h',
                        title='Linear Regression Feature Importance')
        fig_lr.update_layout(yaxis={'categoryorder': 'total ascending'}, margin=dict(l=0, r=0, t=30, b=0),
                             width=600, height=300)
        st.plotly_chart(fig_lr)

        # Plotting the feature importance for Decision Tree Regression Model
        fig_dtr = px.bar(decision_tree_feature_importance, x='Importance', y='Feature', orientation='h',
                         title='Decision Tree Regressor Feature Importance')
        fig_dtr.update_layout(yaxis={'categoryorder': 'total ascending'}, margin=dict(l=0, r=0, t=30, b=0),
                              width=600, height=300)
        st.plotly_chart(fig_dtr)

        # Plotting the feature importance for Random Forest Regression Model
        fig_rfr = px.bar(random_forest_feature_importance, x='Importance', y='Feature', orientation='h',
                         title='Random Forest Regressor Feature Importance')
        fig_rfr.update_layout(yaxis={'categoryorder': 'total ascending'}, margin=dict(l=0, r=0, t=30, b=0),
                              width=600, height=300)
        st.plotly_chart(fig_rfr)

    if st.checkbox("Cluster Analysis (K-means)"):
        # Use clustering techniques to group countries with similar Ladder scores and other feature scores)*

        whr_df_copy = whr_df.copy()

        # Select relevant features for clustering
        features = ['Ladder score', 'Logged GDP per capita', 'Social support', 'Healthy life expectancy',
                    'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']

        # Standardize the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(whr_df[features])

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=5, random_state=42)
        whr_df_copy['Cluster'] = kmeans.fit_predict(scaled_features)

        # Visualize the clusters using PCA
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_features)

        # Add PCA components to the DataFrame
        whr_df_copy['PCA1'] = principal_components[:, 0]
        whr_df_copy['PCA2'] = principal_components[:, 1]

        # Get centroids
        centroids = kmeans.cluster_centers_
        centroids_pca = pca.transform(centroids)

        # Define colors for clusters
        cluster_colors = ['blue', 'green', 'orange', 'purple', 'pink']

        # Plot
        fig = go.Figure()

        # Add scatter plot for clusters
        for cluster in range(5):
            cluster_data = whr_df_copy[whr_df_copy['Cluster'] == cluster]
            fig.add_trace(go.Scatter(x=cluster_data['PCA1'], y=cluster_data['PCA2'],
                        mode='markers', marker=dict(size=5, color=cluster_colors[cluster]),
                        text=cluster_data['Country name'], hoverinfo='text', name=cluster))

        # Add centroids to the plot
        fig.add_trace(go.Scatter(x=centroids_pca[:, 0], y=centroids_pca[:, 1],
                        mode='markers', marker=dict(color='red', size=10, symbol='x'), name='Centroids'))

        # Update layout to show legend in the top right corner
        fig.update_layout(
            title='Cluster groups of Countries based on similar feature scores',
            xaxis_title='PCA1', yaxis_title='PCA2', legend=dict(x=1, y=1, traceorder='normal',
            bgcolor='rgba(255, 255, 255, 0.5)', bordercolor='Black', borderwidth=1,
            ), margin=dict(l=0, r=0, t=30, b=0),
            showlegend=True
        )

      #  st.title("Cluster Analysis of Countries Based on Happiness Report Features")
        st.plotly_chart(fig)

        # Display the results (Cluster groups, PCA of each country) in dataframe
        st.dataframe(whr_df_copy[['Country name', 'Ladder score', 'Cluster', 'PCA1', 'PCA2']])


    if st.checkbox("Future Trend Analysis (Observe future trends over time)"):
        # Future Trend Analysis (Observe future trends over time by predicting Ladder scores for different years.)

        # Define future years
        future_years = [2025, 2026, 2027, 2028, 2029, 2030]

        # Create future data - generate 6 random numbers between 0 and 10 for each feature
        future_data = pd.DataFrame({
            'Logged GDP per capita': np.random.rand(len(future_years)) * 10,
            'Social support': np.random.rand(len(future_years)) * 10,
            'Healthy life expectancy': np.random.rand(len(future_years)) * 10,
            'Freedom to make life choices': np.random.rand(len(future_years)) * 10,
            'Generosity': np.random.rand(len(future_years)) * 10,
            'Perceptions of corruption': np.random.rand(len(future_years)) * 10
        }, index=future_years)

        # Predict future scores using the trained models
      #  future_predictions_LR = modelLR.predict(future_data)
        future_predictions_DTR = modelDTR.predict(future_data)
        future_predictions_RFR = modelRFR.predict(future_data)

        # Combine historical and future data
        historical_data = whr_df.groupby('year')['Ladder score'].mean()
    #    future_data['Ladder score LR'] = future_predictions_LR
        future_data['Ladder score DTR'] = future_predictions_DTR
        future_data['Ladder score RFR'] = future_predictions_RFR

        # Prepare data for plotting
      #  combined_data_LR = pd.concat([historical_data, future_data['Ladder score LR']])
        combined_data_DTR = pd.concat([historical_data, future_data['Ladder score DTR']])
        combined_data_RFR = pd.concat([historical_data, future_data['Ladder score RFR']])

        # Plot the DataFrame
        plot_data = pd.DataFrame({
          # 'Year': combined_data_LR.index,
            'Year': combined_data_DTR.index,
          # 'Linear Regressor': combined_data_LR.values,
            'Decision Tree Regressor': combined_data_DTR.values,
            'Random Forest Regressor': combined_data_RFR.values
        })

        plot_data = plot_data.reset_index().melt(id_vars='Year',
                                                 value_vars=['Decision Tree Regressor', # value_vars=['Linear Regressor', 'Decision Tree Regressor',
                                                             'Random Forest Regressor'],
                                                 var_name='Model', value_name='Ladder Score')

        # Plot with Plotly Express
        fig = px.line(plot_data, x='Year', y='Ladder Score', color='Model', markers=True,
                      title='Predicted Average Future Ladder Scores over Time')
        st.plotly_chart(fig)


    if st.checkbox("Scenario Analysis"):
        # Scenario Analysis (performing scenario analysis by changing the values of certain factors to see how they impact the predicted Ladder scores

        # Calculate mean values for the columns to create a base scenario
        base_scenario = whr_df[['Logged GDP per capita', 'Social support', 'Healthy life expectancy',
                                'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']].mean()

        # Create scenarios by modifying the base scenario
        scenarios = {
            'Scenario 1 (+10%)': base_scenario * 1.1,  # Increase all values by 10%
            'Scenario 2 (-10%)': base_scenario * 0.9,  # Decrease all values by 10%
            'Scenario 3 (+20%)': base_scenario * 1.2,  # Increase all values by 20%
            'Scenario 4 (-20%)': base_scenario * 0.8,  # Decrease all values by 20%
            'Scenario 5 (+30%)': base_scenario * 1.3,  # Increase all values by 30%
            'Scenario 6 (-30%)': base_scenario * 0.7,  # Decrease all values by 30%
            'Scenario 7 (+40%)': base_scenario * 1.4,  # Increase all values by 40%
            'Scenario 8 (-40%)': base_scenario * 0.6,  # Decrease all values by 40%
            'Scenario 9 (+50%)': base_scenario * 1.5,  # Increase all values by 50%
            'Scenario 10 (-50%)': base_scenario * 0.5,  # Decrease all values by 50%
        }

        # Convert scenarios to DataFrame
        scenario_df = pd.DataFrame(scenarios).T  # Transpose to get scenarios as rows

        # Create a copy of the scenario_df for predictions to avoid modifying the original DataFrame
        scenario_predictions = scenario_df.copy()

        # Predict ladder scores for each scenario using the trained models
        scenario_predictions['Predicted Ladder Score LR'] = modelLR.predict(scenario_df)
        scenario_predictions['Predicted Ladder Score DTR'] = modelDTR.predict(scenario_df)
        scenario_predictions['Predicted Ladder Score RFR'] = modelRFR.predict(scenario_df)

        # st.title('Scenario Predictions')

        st.write('Base Scenario')
        st.write(base_scenario)

        st.write('Scenario Predictions *(scroll left to see predicted ladder score)*')
        st.write(scenario_predictions)

    # ********* Compare and Validate ********* #

    ### Use the trained models to make prediction on new similar dataset ####

    st.markdown("<hr class='custom-hr-dc'>", unsafe_allow_html=True)

    # *** Verify or not: Make predictions on our previously trained models with our new/test dataset *** #

    # Load the trained pipeline (model) from the file
    trained_modelLR = joblib.load('trained_modelLR.pkl')
    trained_modelDTR = joblib.load('trained_modelDTR.pkl')
    trained_modelRFR = joblib.load('trained_modelRFR.pkl')

    #   st.dataframe(whr_df_2022_2024)
    whr_df_2022_2024 = pd.read_csv('cleaned-world-happiness-report-2022-2024.csv')

    # Drop not needed features from the new WHR2022-2024 dataframe (2022 till 2024)
    whr_df_2022_2024_pred = whr_df_2022_2024.drop(['Ladder score', 'Country name', 'year', 'Regional indicator'],
                                                  axis=1)
    new_target = whr_df_2022_2024['Ladder score']

    # Make predictions on the new dataset
    new_predictionsLR = trained_modelLR.predict(whr_df_2022_2024_pred)
    new_predictionsDTR = trained_modelDTR.predict(whr_df_2022_2024_pred)
    new_predictionsRFR = trained_modelRFR.predict(whr_df_2022_2024_pred)

    # Calculate evaluation metrics for LinearRegression
    new_eval_result_LR = {'MAE': mean_absolute_error(new_target, new_predictionsLR),
                          'MSE': mean_squared_error(new_target, new_predictionsLR),
                          'RMSE': np.sqrt(mean_squared_error(new_target, new_predictionsLR)),
                          'R-squared (R²) Score': r2_score(new_target, new_predictionsLR)
                          }

    # Calculate evaluation metrics for DecisionTreeRegressor
    new_eval_result_DTR = {'MAE': mean_absolute_error(new_target, new_predictionsDTR),
                           'MSE': mean_squared_error(new_target, new_predictionsDTR),
                           'RMSE': np.sqrt(mean_squared_error(new_target, new_predictionsDTR)),
                           'R-squared (R²) Score': r2_score(new_target, new_predictionsDTR)
                           }

    # Calculate evaluation metrics for RandomForestRegressor
    new_eval_result_RFR = {'MAE': mean_absolute_error(new_target, new_predictionsRFR),
                           'MSE': mean_squared_error(new_target, new_predictionsRFR),
                           'RMSE': np.sqrt(mean_squared_error(new_target, new_predictionsRFR)),
                           'R-squared (R²) Score': r2_score(new_target, new_predictionsRFR)
                           }

    new_eval_results = {'Linear Regressor': new_eval_result_LR,
                        'Decision Tree Regressor': new_eval_result_DTR,
                        'Random Forest Regressor': new_eval_result_RFR
                        }

    # Validate: model results of sample from new dataset (period: 2022 - 2024)
    if st.checkbox("Use Trained Model to make Prediction on New Dataset (Period: 2022 - 2024)"):
        st.write("Evaluation result: New dataset (2022 - 2024) predicted on trained models")
        new_evaluation_results = pd.DataFrame(new_eval_results)
        new_evaluation_results = new_evaluation_results
        st.write(new_evaluation_results)

      #  if st.checkbox("Overall Prediction Rate in Percentage for each Model"):
        # Calculate the prediction rates
        overall_prediction_new_df = pd.DataFrame({'Actual Value': new_target,
                                                 'Prediction Rate (%) LR': 100 * (
                                                             1 - abs(new_target - new_predictionsLR) / new_target),
                                                 'Prediction Rate (%) DTR': 100 * (
                                                             1 - abs(new_target - new_predictionsDTR) / new_target),
                                                 'Prediction Rate (%) RFR': 100 * (
                                                             1 - abs(new_target - new_predictionsRFR) / new_target)
                                                 })

        # Calculate the overall prediction rate in percentage for each model
        mean_prediction_rates = {
            'Prediction Rate Linear Regression': overall_prediction_new_df['Prediction Rate (%) LR'].mean(),
            'Prediction Rate Decision Tree Regression': overall_prediction_new_df['Prediction Rate (%) DTR'].mean(),
            'Prediction Rate Random Forest Regressor': overall_prediction_new_df['Prediction Rate (%) RFR'].mean()
        }

        # Create a list of dictionaries for the dataframe
        formatted_rates = [{'Models': model, 'Rates in Percentage (%)': f'{rate:.2f}%'} for model, rate in
                           mean_prediction_rates.items()]

        # Convert the list to a dataframe
        mean_prediction_rates_df = pd.DataFrame(formatted_rates)

        # st.title('Prediction Rates Comparison')

        st.write('Overall Mean Prediction Rates')
        st.write(mean_prediction_rates_df)

        st.write('Actual Value and Prediction Rate in Percentage')
        st.write(overall_prediction_new_df)

        # Display actual and predicted values
        st.write('Actual vs Predicted Values')
        actual_vs_predicted = pd.DataFrame({
            'Actual Values': new_target,
            'Predicted LR Values': new_predictionsLR,
            'Predicted DTR Values': new_predictionsDTR,
            'Predicted RFR Values': new_predictionsRFR
        })
        st.write(actual_vs_predicted)


    # Horizontal divider
    st.markdown("<hr class='custom-hr-dc'>", unsafe_allow_html=True)


    # Retrain the model and make further predictions with new dataset #

    # Load the trained models from the file
    load_trained_modelLR = joblib.load('trained_modelLR.pkl')
    load_trained_modelDTR = joblib.load('trained_modelDTR.pkl')
    load_trained_modelRFR = joblib.load('trained_modelRFR.pkl')

    # Historical Trend Analysis(Observe trends over time by predicting Ladder scores for different years - historical and new data.) *
    if st.checkbox("Historical Trend Analysis"):
        #
        # # Copy the specified feature columns
        # new_data = whr_df_2022_2024.loc[:, ['Logged GDP per capita', 'Social support', 'Healthy life expectancy',
        #                                     'Freedom to make life choices', 'Generosity',
        #                                     'Perceptions of corruption']].copy()
        #
        # # Predict using the trained models
        # new_predictionsLR = load_trained_modelLR.predict(new_data)
        # new_predictionsDTR = load_trained_modelDTR.predict(new_data)
        # new_predictionsRFR = load_trained_modelRFR.predict(new_data)
        #
        # # Create a copy of the whr_df for trend analysis to avoid modifying the original DataFrame
        # whr_df_trend = whr_df.copy()
        #
        # # Create a dataframe to store the results
        # historical_data = whr_df_trend.groupby('year')['Ladder score'].mean()
        # new_data['Ladder score LR'] = new_predictionsLR
        # new_data['Ladder score DTR'] = new_predictionsDTR
        # new_data['Ladder score RFR'] = new_predictionsRFR
        #
        # # Combine historical and new data into a single DataFrame for each model
        # combined_data_LR = pd.concat([historical_data, new_data.set_index(whr_df_2022_2024['year'])['Ladder score LR']])
        # combined_data_DTR = pd.concat([historical_data, new_data.set_index(whr_df_2022_2024['year'])['Ladder score DTR']])
        # combined_data_RFR = pd.concat([historical_data, new_data.set_index(whr_df_2022_2024['year'])['Ladder score RFR']])
        #
        # # Ensure unique years for all models
        # combined_data_LR = combined_data_LR.groupby(combined_data_LR.index).mean()
        # combined_data_DTR = combined_data_DTR.groupby(combined_data_DTR.index).mean()
        # combined_data_RFR = combined_data_RFR.groupby(combined_data_RFR.index).mean()
        #
        # # Plot the trends for each model using Plotly Express
        # fig = px.line(combined_data_LR, labels={'value': 'Ladder Score', 'index': 'Year'},
        #               title='Predicted Average Ladder Scores Over Time (Historical and New Data)')
        # fig.add_scatter(x=combined_data_LR.index, y=combined_data_LR.values, mode='lines+markers',
        #                 name='Linear Regressor', line=dict(color='red'))
        # fig.add_scatter(x=combined_data_DTR.index, y=combined_data_DTR.values, mode='lines+markers',
        #                 name='Decision Tree Regressor', line=dict(color='orange'))
        # fig.add_scatter(x=combined_data_RFR.index, y=combined_data_RFR.values, mode='lines+markers',
        #                 name='Random Forest Regressor', line=dict(color='blue'))
        #
        # fig.update_layout(title='Predicted Average Ladder Scores Over Time (Historical and New Data)',
        #                   xaxis_title='Year', yaxis_title='Ladder Score', legend_title='Model',
        #                   margin=dict(l=0, r=0, t=25, b=0),
        #                   width=800, height=300,
        #                   xaxis=dict(tickmode='linear'))
        #
        # # Remove legend entries with 0 values
        # # fig.for_each_trace(lambda trace: trace.update(showlegend=False) if trace.y[0] == 0 else ())
        #
        # st.plotly_chart(fig)
        #
        # import pandas as pd
        # import plotly.express as px
        # import streamlit as st

        # Copy the specified feature columns
        new_data = whr_df_2022_2024.loc[:, ['Logged GDP per capita', 'Social support', 'Healthy life expectancy',
                                            'Freedom to make life choices', 'Generosity',
                                            'Perceptions of corruption']].copy()

        # Predict using the trained models
        new_predictionsLR = load_trained_modelLR.predict(new_data)
        new_predictionsDTR = load_trained_modelDTR.predict(new_data)
        new_predictionsRFR = load_trained_modelRFR.predict(new_data)

        # Create a copy of the whr_df for trend analysis to avoid modifying the original DataFrame
        whr_df_trend = whr_df.copy()

        # Create a dataframe to store the results
        historical_data = whr_df_trend.groupby('year')['Ladder score'].mean()
        new_data['Ladder score LR'] = new_predictionsLR
        new_data['Ladder score DTR'] = new_predictionsDTR
        new_data['Ladder score RFR'] = new_predictionsRFR

        # Combine historical and new data into a single DataFrame for each model
        combined_data_LR = pd.concat([historical_data, new_data.set_index(whr_df_2022_2024['year'])['Ladder score LR']])
        combined_data_DTR = pd.concat(
            [historical_data, new_data.set_index(whr_df_2022_2024['year'])['Ladder score DTR']])
        combined_data_RFR = pd.concat(
            [historical_data, new_data.set_index(whr_df_2022_2024['year'])['Ladder score RFR']])

        # Ensure unique years for all models
        combined_data_LR = combined_data_LR.groupby(combined_data_LR.index).mean()
        combined_data_DTR = combined_data_DTR.groupby(combined_data_DTR.index).mean()
        combined_data_RFR = combined_data_RFR.groupby(combined_data_RFR.index).mean()

        # Streamlit slider for selecting the range of years
        min_year = int(combined_data_LR.index.min())
        max_year = int(combined_data_LR.index.max())
        selected_years = st.slider('Select the range of years to plot:', min_year, max_year, (min_year, max_year))

        # Filter data based on selected years
        filtered_data_LR = combined_data_LR.loc[selected_years[0]:selected_years[1]]
        filtered_data_DTR = combined_data_DTR.loc[selected_years[0]:selected_years[1]]
        filtered_data_RFR = combined_data_RFR.loc[selected_years[0]:selected_years[1]]

        # Plot the trends for each model using Plotly Express
        fig = px.line(filtered_data_LR, labels={'value': 'Ladder Score', 'index': 'Year'},
                      title='Predicted Average Ladder Scores Over Time (Historical and New Data)')
        fig.add_scatter(x=filtered_data_LR.index, y=filtered_data_LR.values, mode='lines+markers',
                        name='Linear Regressor', line=dict(color='red'))
        fig.add_scatter(x=filtered_data_DTR.index, y=filtered_data_DTR.values, mode='lines+markers',
                        name='Decision Tree Regressor', line=dict(color='orange'))
        fig.add_scatter(x=filtered_data_RFR.index, y=filtered_data_RFR.values, mode='lines+markers',
                        name='Random Forest Regressor', line=dict(color='blue'))

        fig.update_layout(title='Predicted Average Ladder Scores Over Time (Historical and New Data)',
                          xaxis_title='Year', yaxis_title='Ladder Score', legend_title='Model',
                          margin=dict(l=0, r=0, t=25, b=0),
                          width=800, height=400,
                          xaxis=dict(tickmode='linear'))

        # Remove legend entries with 0 values
        fig.for_each_trace(lambda trace: trace.update(showlegend=False) if trace.name == '0' else ())

        # Ensure 2004 is not shown on the x-axis
        fig.update_xaxes(range=[selected_years[0], selected_years[1]])

        st.plotly_chart(fig)

    # Horizontal divider
    st.markdown("<hr class='custom-hr-dc'>", unsafe_allow_html=True)

    # Select rows where the 'year' column is between 2022 and 2024 (We will use the dataset of 2022-2024 to test our trained model)
    #   whr_df_2022_2024 = whr_df_2022_2024[(whr_df_2022_2024['year'] >= 2022) & (whr_df_2022_2024['year'] <= 2024)]

    # Define the feature and target variables
    y_target_new = whr_df_2022_2024['Ladder score']
    X_features_new = whr_df_2022_2024.drop(['Ladder score', 'Country name', 'year', 'Regional indicator'], axis=1)

    # Split the new features and target
    X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(X_features_new, y_target_new, test_size=0.2,
                                                                        random_state=42)

    # Fit the models on the new training data
    retrained_modelLR = load_trained_modelLR.fit(X_new_train, y_new_train)
    retrained_modelDTR = load_trained_modelDTR.fit(X_new_train, y_new_train)
    retrained_modelRFR = load_trained_modelRFR.fit(X_new_train, y_new_train)

    # Make predictions with new test data
    new_retrain_predictionsLR = retrained_modelLR.predict(X_new_test)
    new_retrain_predictionsDTR = retrained_modelDTR.predict(X_new_test)
    new_retrain_predictionsRFR = retrained_modelRFR.predict(X_new_test)

    # Calculate evaluation metrics for LinearRegression
    retrained_eval_result_LR = {
        'MAE': mean_absolute_error(y_new_test, new_retrain_predictionsLR),
        'MSE': mean_squared_error(y_new_test, new_retrain_predictionsLR),
        'RMSE': np.sqrt(mean_squared_error(y_new_test, new_retrain_predictionsLR)),
        'R-squared (R²) Score': r2_score(y_new_test, new_retrain_predictionsLR)
    }

    # Calculate evaluation metrics for DecisionTreeRegressor
    retrained_eval_result_DTR = {
        'MAE': mean_absolute_error(y_new_test, new_retrain_predictionsDTR),
        'MSE': mean_squared_error(y_new_test, new_retrain_predictionsDTR),
        'RMSE': np.sqrt(mean_squared_error(y_new_test, new_retrain_predictionsDTR)),
        'R-squared (R²) Score': r2_score(y_new_test, new_retrain_predictionsDTR)
    }

    # Calculate evaluation metrics for RandomForestRegressor
    retrained_eval_result_RFR = {
        'MAE': mean_absolute_error(y_new_test, new_retrain_predictionsRFR),
        'MSE': mean_squared_error(y_new_test, new_retrain_predictionsRFR),
        'RMSE': np.sqrt(mean_squared_error(y_new_test, new_retrain_predictionsRFR)),
        'R-squared (R²) Score': r2_score(y_new_test, new_retrain_predictionsRFR)
    }

    retrained_eval_results = {
        'Linear Regressor': retrained_eval_result_LR,
        'Decision Tree Regressor': retrained_eval_result_DTR,
        'Random Forest Regressor': retrained_eval_result_RFR
    }

    # Validate: New sample data on retrained models:
    if st.checkbox("Retrain Models with New Dataset (Period: 2022 - 2024)"):
        st.write("Retrained Models Evaluation Results")
        retrained_eval_results = pd.DataFrame(retrained_eval_results)
        retrained_eval_results = retrained_eval_results
        st.write(retrained_eval_results)

    # Horizontal divider
    st.markdown("<hr class='custom-hr-dc'>", unsafe_allow_html=True)

    # *** Compare the three results *** #

    if st.checkbox("Compare All Model Results (80/20 Split, Predict and Retrained with New Dataset (2022-2024)"):
        st.write("All Model Evaluation Result (80/20 Split)")
        evaluation_results = pd.DataFrame(evaluation_results)
        evaluation_results = evaluation_results
        st.write(evaluation_results)
        st.write("\n")

        # Display evaluation metrics result of the new predictions from the first trained models: (period: 2022 - 2024)
        st.write("All Model Evaluation Results: New Dataset (2022-2024) Predict with Trained Models")
        new_evaluation_results = pd.DataFrame(new_eval_results)
        new_evaluation_results = new_evaluation_results
        st.write(new_evaluation_results)

        # Display evaluation metrics result of the predictions retrained models:
        st.write("All Model Evaluation Results: Retrained with New Dataset (2022-2024)")
        retrained_eval_results = pd.DataFrame(retrained_eval_results)
        retrained_eval_results = retrained_eval_results
        st.write(retrained_eval_results)

    # Horizontal divider
    st.markdown("<hr class='custom-hr-dc'>", unsafe_allow_html=True)

    if st.checkbox("Evaluations Metrics Definitions (optional)"):
        st.write("(1) Mean Absolute Error (MAE)", unsafe_allow_html=True)
        st.write("(2) Mean Squared Error (MSE)", unsafe_allow_html=True)
        st.write("(3) Root Mean Squared Error (RMSE)", unsafe_allow_html=True)
        st.write("(4) R - squared(R²) Score", unsafe_allow_html=True)


elif page == pages[5]:
    st.write("<h2>Insight & Conclusion (Step 4)</h2>", unsafe_allow_html=True)
    st.write("My goal for this project is to use interactive virtualisation to present the World Happiness Report, "
             "While the most important purpose is to apply explanatory analysis and machine learning to determine the factors "
             "that influences the ladder score and consequentially the happiness of the people. As well as to "
             "apply machine learning methods and train models for prediction, to see how the models respond "
             "to my findings. <br><br> At first sight the exploratory analysis reveals that over time the WHR and the data it provides has been "
             "validated along country and regional lines, this can be observed from the increase in number "
             "of countries that participate in the project over the years.<br><br> The trend of world happiness "
             "score (also refered as Ladder score) has shown over the year from regional perspective that  North America and ANZ and "
             "Western European has always been on the lead, while from country perspective the Western European and "
             "Scandinavian countries such as Finland, Denmark,  Iceland has held the top 3 positions on "
             "world happiness score in recent years. Whereas, Regions like Middle East and North Africa has been in "
             "decline, while regions like Central and Eastern Europe, Commonwealth of Independent States, "
             "Southeast Asia, South Asia and Sub-Saharan Africa seems to be improving.  Some of the countries"
             " with low ladder score such as Botswana, Lesotho, and Zimbabwe have relatively stable scores "
             "with minor fluctuations. While some countries like Afghanistan, Burundi, Haiti, and Yemen show a general "
             " downward trend, indicating worsening conditions.<br><br> The  visual analysis and model results together "
             " provided a comprehensive understanding of global happiness trends and feature importance, by highlighting the importance of economic stability, social support and "
             "healthy life expectancy, as well as freedom to make life choices and perception of corruption, sequentially in enhancing global happiness.<br><br> "
              "The Random Forest Regressor is the most accurate model, making it the best choice for predictive tasks in this context. "
             "However, retraining the model is crucial for improving model performance and generalization. <br><br> The implication of this findings is that "
             "people from non-productive countries and regions, with bad and non-affordable health care system, little or know social "
             "support and high corruption will continue to be unhappier unless there is a significant positive "
             "change in policies and good governance in this areas. Such policies should be aimed to improve the log GDP per capita, social support and healthy life expectancy at birth which clearly  "
             "stands out as the 3 key factors that influence the happiness of the people as showned in my finding.<br><br>"
            " I believe to have achieved the goal of the project by identifying the significant variables and their magnitude in "
             "determining the ladder score and consequently the reason why some countries seems to be ranked "
             "happier than the others.<br><br> However, It is worth note at this point that from my perspective "
             "although this socio-economic metrics serves as good indicator for measuring happiness there "
             "could be other factors, which I believe might contribute to the happiness which is not "
             "considered here, such as religious belief, culture of the people,  as well as world view of "
             "the individual, countries and regions in general.", unsafe_allow_html=True)






