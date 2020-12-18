# Understanding Customer Churning using Spark for Big Data
The demand for audio streaming services has been increased, the offer for these services has increased too, and the competence is fierce. Customer loyalty to a specific app or service is crucial for this model business.

Sparkify is a fictional popular digital media service created by Udacity, similar to Spotify or Pandora; many users use their services every day. It also has two modalities: using a free tier or a premium subscription model on which a user is free to upgrade, downgrade or cancel the service as they will like. It is important that the user likes the service to be loyal to Sparkify.

We track each event; if a user plays a song, visits a specific page, or gets an error, it generates a lot of data that will easily scale up to GB of data. Fortunately, for large amounts of data, we can use big data technologies like Spark, and we can analyze and predict customer churn to avoid that they leave.

*Medium:* https://erick-ramirez.medium.com/understanding-customer-churning-using-spark-for-big-data-30708fefa226

### Prerequisites
The environment needed for this project:
1. [Python 3.7](https://www.python.org/downloads/release/python-370/)
2. [seaborn: statistical data visualization](https://seaborn.pydata.org/)
3. [pyspark package](https://spark.apache.org/docs/latest/api/python/index.html)
4. [user_agents](https://pypi.org/project/user-agents/)
5. [functools](https://docs.python.org/3/library/functools.html)
6. [statsmodels](https://www.statsmodels.org/stable/index.html)

### Explanation of the files in the repository
1. Sparkify.ipynb: notebook file that contains all the process in which are explained in  the section **Steps performed**
2. Sparkify.html: html version of the notebook: Sparkify.ipynb
3. results.csv: sumary of results according to the models created and its evaluation related to execution time, accuracy and F1-score.
4. images: folder that contains the images used in this file.
  
### Instructions to run the notebook
1. clone the github repository: `git clone https://github.com/Erickramirez/Sparkify.git`
2. verify the Prerequisites
3. Open and run the file  file: Sparkify.ipynb

## What is customer churn?
Is when a user stops using our company’s product, in this case, the Sparkify service. It is a metric that helps if the business is growing or not, and it is important because normally it costs more to acquire new customers than it does to retain the existing ones. To reduce customer churn, Sparkify has decided to analyze the data and offer incentives to keep the customers.

Spark MLlib implementation to build machine learning models with large datasets for predicting churn rates

## Project definition
It is a realistic dataset with Spark to engineer relevant features for predicting churn. 
Using Spark MLlib to build machine learning models with large datasets. 
The [full dataset is 12GB](s3n://udacity-dsnd/sparkify/sparkify_event_data.json), of which you can analyze a [mini subset 128MB](s3n://udacity-dsnd/sparkify/mini_sparkify_event_data.json)
The code for this project is in the Github repository.
create a spark session:
```# create a Spark session
spark = SparkSession \
        .builder \
        .master("local[*]") \
        .config("spark.ui.port",3000) \
        .appName("Sparkify") \
        .getOrCreate()
```
### Steps performed
**Load and Clean Dataset:** Remove data with mission UserIds, drop first name and last name.

For the mini subset of data, it is a small dataset for Machine learning; it contains 286,500, there are missing userIds, normally the cookies are disabled, or we cannot track that specific user, then the data related to them has been removed. There are 225 unique users, which has the following distribution:
![level](/images/level.png)

*Customer count free/paid distribution by gender(F/M) and Churn (cancelled)*

**Exploratory Data Analysis:** exploratory data to check data distribution and features. Define Churn  as `Cancellation Confirmation` events and exploratory data analysis to observe the behavior for users who stayed vs users who churned.
23.11% of users have canceled the service, and the user interactions about the visit pages are the following:

![level](/images/pages1.png)

![level](/images/pages2.png)

![level](/images/pages3.png)


There is a small difference in behavior related to users who have churn and do not churn the service. `Thumbs Up` , `Thumps down` , and `Roll Advert` , however, there is still not a statistically significant difference to conclude. This evaluation of the box plot was performed to visualize the percentiles and mean to check distribution.
To avoid overfitting for the model, we have removed the pages related to the account's cancelation, which means a churn.
the behavior of the users related to the sessions (duration in hours and number of songs played)

 
**Feature engineering:** actions performed:
- Extract the necessary features from the smaller subset of data (final result at User level and not at event level)
- Continue using spark for scale up, using the lazy evaluation in Apache Spark

Features to extract based on the data exploration and Churn definition
The data will be present by userId

1. Categorical Features (need label encoding)
    - gender
    - pages (remove Cancellation Confirmation and Cancel)
    - Browser (extracted from userAgent)
    - OS (extracted from userAgent)
    
    ![level](/images/categorical.png)
    
    *example of categorical feature labeled*

2. Numerical Features (need to be scaled)
    - mean of songs in a session
    - mean of session duration (in hours)
    - mean of events (registers) in a session
    - days of use
    
    ![level](/images/scaled.png)
    
    *example of numerical features scaled*
    
The final features to train are the following:
```
root
 |-- userId: string (nullable = true)
 |-- Churn: integer (nullable = true)
 |-- level: integer (nullable = true)
 |-- gender: integer (nullable = true)
 |-- lenght_avg_scaled: double (nullable = true)
 |-- day_of_week_1_scaled: double (nullable = true)
 |-- day_of_week_2_scaled: double (nullable = true)
 |-- day_of_week_3_scaled: double (nullable = true)
 |-- day_of_week_4_scaled: double (nullable = true)
 |-- day_of_week_5_scaled: double (nullable = true)
 |-- day_of_week_6_scaled: double (nullable = true)
 |-- day_of_week_7_scaled: double (nullable = true)
 |-- songs_by_session_scaled: double (nullable = true)
 |-- session_duration_scaled: double (nullable = true)
 |-- event_count_by_session_scaled: double (nullable = true)
 |-- total_sessions_scaled: double (nullable = true)
 |-- browser_chrome: integer (nullable = true)
 |-- browser_firefox: integer (nullable = true)
 |-- browser_ie: integer (nullable = true)
 |-- browser_mobile_safari: integer (nullable = true)
 |-- browser_safari: integer (nullable = true)
 |-- OS_linux: integer (nullable = true)
 |-- OS_mac_os_x: integer (nullable = true)
 |-- OS_ubuntu: integer (nullable = true)
 |-- OS_windows: integer (nullable = true)
 |-- OS_ios: integer (nullable = true)
 |-- page_about_scaled: double (nullable = true)
 |-- page_add_friend_scaled: double (nullable = true)
 |-- page_add_to_playlist_scaled: double (nullable = true)
 |-- page_downgrade_scaled: double (nullable = true)
 |-- page_error_scaled: double (nullable = true)
 |-- page_help_scaled: double (nullable = true)
 |-- page_home_scaled: double (nullable = true)
 |-- page_logout_scaled: double (nullable = true)
 |-- page_nextsong_scaled: double (nullable = true)
 |-- page_roll_advert_scaled: double (nullable = true)
 |-- page_save_settings_scaled: double (nullable = true)
 |-- page_settings_scaled: double (nullable = true)
 |-- page_submit_downgrade_scaled: double (nullable = true)
 |-- page_submit_upgrade_scaled: double (nullable = true)
 |-- page_thumbs_down_scaled: double (nullable = true)
 |-- page_thumbs_up_scaled: double (nullable = true)
 |-- page_upgrade_scaled: double (nullable = true)
```
**Model building**

The data has been split into training and testing datasets assigning 70% and 30%, respectively. The classifications used for this analysis are the following:
- DecisionTreeClassifier
- GBTClassifier
- RandomForestClassifier
- LinearSVC

**Model evaluation**

We will test the trained models’ performances and select the one that has the best performance; it is based on f1-score
![F1score](/images/f1.png)

*obtained from https://en.wikipedia.org/wiki/F-score*

The reason to use this evaluation is due to an imbalance in class distribution that is present in the dataset. There is a small portion of users that have churn, and the purpose of this analysis is to identify the users that can churn Sparkify’s service correctly.

Results:
![results](/images/results.png)

*result of the models*

### Conclusions
The model Machine learning and Spark allow processing large amounts of data; this is helpful for scalable analysis and can keep track of the users. Once we have identified the users with possible churn behavior, a good business strategy to reduce the churn will be AB/Test to evaluate and get more engagement from the users, and this will start with research about the current performance, then observe and formulate a hypothesis and define variations, for two groups (new incentive and a control group with the existing ones) and evaluate the new behavior.

F1 score was the metric to optimize, the better result for this was Linear Support Vector Classification(LinearSVC) with a score of 0.8061; despite the time of training, once it has been trained the prediction time will not be significant for new users. The model can be evaluated once a week to identify new users, and the tuning of the model and features to extract can be evaluated once a week.

For future work, evaluation on other windows of time, features extraction can be useful to improve the performance, checking that we are not getting overfitting in the model. With more data, it will be useful to evaluate: training, validation, and test sets.

