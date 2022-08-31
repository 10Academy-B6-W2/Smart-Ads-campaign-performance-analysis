# Smart-Ads-campaign-performance-analysis

Ad campaign performance evaluation using AB Testing

Case Overview: SmartAd is a mobile first advertiser agency. It is running an online ad for a client with the intention of increasing brand awareness. The company provides an additional service called Brand Impact Optimizer (BIO), a lightweight questionnaire, served with every campaign to determine the impact of the ad they design.

Objectives: The task at hand is to design a reliable hypothesis testing algorithm for the BIO service and determine whether the recent advertising campaign resulted in significant lift in brand awareness.

Methods

    Sequential A/B testing.
    Classic A/B testing.
    A/B testing with Machine Learning.
    
    
# Installation
### Step 1: Downloading source code
```
git clone https://github.com/10Academy-B6-W2/Smart-Ads-campaign-performance-analysis.git
```
### Step 2: Installation of dependencies
```
pip install -r requirements.txt
```

# Data
The BIO data for this project is a “Yes” and “No” response of online users to the following question
Q: Do you know the brand LUX?

	1. Yes
	2. No
# Dataset Column description
* auction_id: the unique id of the online user who has been presented the BIO. In standard terminologies this is called an impression id. The user may see the BIO questionnaire but choose not to respond. In that case both the yes and no columns are zero.

* experiment: which group the user belongs to - control or exposed.

* date: the date in YYYY-MM-DD format

* hour: the hour of the day in HH format.

* device_make: the name of the type of device the user has e.g. Samsung

* platform_os: the id of the OS the user has.

* browser: the name of the browser the user uses to see the BIO questionnaire.

* yes: 1 if the user chooses the “Yes” radio button for the BIO questionnaire.

* no: 1 if the user chooses the “No” radio button for the BIO questionnaire.

# A/B Hypothesis Testing
A/B testing, also known as split testing, refers to a randomized experimentation process wherein two or more versions of a variable (web page, page element, etc.) are shown to different segments of website visitors at the same time to determine which version leaves the maximum impact and drive business metrics.

## Sequential A/B testing
A common issue with classical A/B-tests, especially when you want to be able to detect small differences, is that the sample size needed can be prohibitively large. In many cases it can take several weeks, months or even years to collect enough data to conclude a test. 

*   The lower number of errors we require, the larger sample size we need.
* The smaller the difference we want to detect, the larger sample size is required.

Sequential sampling works in a very non-traditional way; instead of a fixed sample size, you choose one item (or a few) at a time, and then test your hypothesis. 
