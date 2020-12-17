# Understanding Customer Churning using Spark for Big Data
The demand for audio streaming services has been increased, the offer for these services has increased too, and the competence is fierce. Customer loyalty to a specific app or service is crucial for this model business.
Sparkify is a fictional popular digital media service created by Udacity, similar to Spotify or Pandora; many users use their services every day. It also has two modalities: using a free tier or a premium subscription model on which a user is free to upgrade, downgrade or cancel the service as they will like. It is important that the user likes the service to be loyal to Sparkify.


We track each event; if a user plays a son, visits a specific page, or gets an error, it generates a lot of data that will easily scale up to GB of data. Fortunately, for large amounts of data, we can use big data technologies like Spark, and we can analyze and predict customer churn to avoid that they leave.

*Medium:* https://erickramireztebalan.medium.com/understanding-customer-churning-using-spark-for-big-data-30708fefa226

### Prerequisites
The environment needed for this project:
1. [Python 3.7](https://www.python.org/downloads/release/python-370/)
2. [NumPy](https://numpy.org/)
3. [pandas](https://pandas.pydata.org/)
4. [nltk -Natural Language Toolkit](https://www.nltk.org/)
5. [SQLAlchemy](https://www.sqlalchemy.org/)
6. [scikit-learn](https://scikit-learn.org/stable/)
7. [pickle — Python object serialization](https://docs.python.org/3/library/pickle.html#module-pickle)
8. [Flask](https://flask.palletsprojects.com/en/1.1.x/)
9. [Plotly](https://plotly.com/python/)
10. [wordcloud](https://pypi.org/project/wordcloud/)

## What is customer churn?
Is when a user stops using our company’s product, in this case, the Sparkify service. It is a metric that helps if the business is growing or not, and it is important because normally it costs more to acquire new customers than it does to retain the existing ones. To reduce customer churn, Sparkify has decided to analyze the data and offer incentives to keep the customers.

Spark MLlib implementation to build machine learning models with large datasets for predicting churn rates

