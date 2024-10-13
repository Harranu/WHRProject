# WHR-Repository
<B>World Happiness Report Project</B>

1. Intro
The goal of this project is to estimate the happiness of countries around the world using socio-economic metrics such as gross domestic product (GDP), social support, healthy life expectancy, freedom to make life choices, generosity and perception of corruption. As well as, to understand how this variables are correlated to each other to influence the the happiness ladder score and consequentially contribute to the happiness of the people in there countries.
I will analyse and present this data using analytical interactive visualizations to determine the combinations of factors that explains why some countries are ranked to be happier than others. I will use the available dataset of the world happiness report from 2005 till 2021 and apply various machine learning methods. I will train 3 models with a sample data from the 2005 till 2021 database, which i will merge as base dataset. To validate or invalidate the outcome of the first prediction, I will make predictions on the trained models with new but similar dataset of 2022 to 2024. Finally, I will retrain the model and make another prediction with the new dataset and compare the results of the models.
The outcome will help countries and policy makers determine the factors that influence overall happiness of the people and areas that can be improved in view of decision and policy making.

2. Included Files: 
- world-happiness-report.csv
- world-happiness-report-2021.csv
- world-happiness-report-new-2024.csv
- cleaned_whr_data.csv
- cleaned_whr_new-test_data.csv
- cleaned-world-happiness-report-2022-2024.csv
- trained_modelLR.pkl
- trained_modelDTR.pkl
- trained_modelRFR.pkl
- styles.css
-  whr_image00.jpeg
- whr_image003.png

3. How to:
- Download all files in one directory. 
- Depending on which version of the source code you want to run, either way you need a python interpreter. Incase of streamlit version, you will need to install streamlit package. Read more on streamlit website on how to run a streamlit app. See this link for further instructions https://docs.streamlit.io/develop/concepts/architecture/run-your-app.
- You may have to adjust the dataset (.csv) files directory accordingly depending on the platform you want to run the code, i.e on Desktop, Google Collab or other Notebooks
