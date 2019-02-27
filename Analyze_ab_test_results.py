#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page.  Either way assure that your code passes the project [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).  **Please save regularly
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# For this project, you will be working to understand the results of an A/B test run by an e-commerce website.  Your goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# **As you work through this notebook, follow along in the classroom and answer the corresponding quiz questions associated with each question.** The labels for each classroom concept are provided for each question.  This will assure you are on the right track as you work through the project, and you can feel more confident in your final submission meeting the criteria.  As a final check, assure you meet all the criteria on the [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[23]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# In[ ]:





# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[24]:


df = pd.read_csv('ab_data.csv')
df.head()


# b. Use the below cell to find the number of rows in the dataset.

# In[4]:


df.info()


# c. The number of unique users in the dataset.

# In[5]:


# df.query("converted == '1'").groupby('user_id').count()
df['user_id'].nunique()


# d. The proportion of users converted.

# In[6]:


df.query("converted == '1'").count()[0] /df.query("converted").count()[0]*100


# e. The number of times the `new_page` and `treatment` don't line up.

# In[7]:


df.query("group == 'treatment' and landing_page !='new_page'").count()[0]


# f. Do any of the rows have missing values?

# In[8]:


df.info()


# `2.` For the rows where **treatment** is not aligned with **new_page** or **control** is not aligned with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to provide how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[9]:



df.drop(df.query("(group == 'treatment' and landing_page != 'new_page') or (group == 'control' and landing_page != 'old_page')").index, inplace=True)
df2 = df
df2.info()


# In[10]:


# # Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[11]:


df2.nunique()['user_id']


# In[12]:


# df2['user_id'].duplicated().count()
duplicated_id = df2['user_id'].duplicated()
sum(duplicated_id)


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[13]:


df2[df2.duplicated(['user_id'])]['user_id']


# c. What is the row information for the repeat **user_id**? 

# In[14]:


df2.query("user_id == '773192'")


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[15]:


df2 = df2[df2.timestamp != '2017-01-14 02:55:59.590927']


# `4.` Use **df2** in the below cells to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[16]:


df2['converted'].mean()*100


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[17]:


df2.query("group == 'control'")['converted'].mean()*100


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[18]:


df2.query("group == 'treatment'")['converted'].mean()*100


# d. What is the probability that an individual received the new page?

# In[19]:


all_users = df2.count()[0]
new_users = df2.query("group == 'treatment'").count()[0]

new_users/all_users*100


# e. Use the results in the previous two portions of this question to suggest if you think there is evidence that one page leads to more conversions?  Write your response below.

# the better converted users are from the old page 
# i htink we should maintain the old page 

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# null : 𝑝𝑜𝑙𝑑 >=𝑝𝑛𝑒𝑤
# alternative :  𝑝𝑜𝑙𝑑 < 𝑝𝑛𝑒𝑤

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **convert rate** for $p_{new}$ under the null? 

# In[25]:


pnew =df2['converted'].mean()
pnew


# b. What is the **convert rate** for $p_{old}$ under the null? <br><br>

# In[26]:


pold = df2['converted'].mean()
pold


# c. What is $n_{new}$?

# In[29]:


n_new = df2.query("group == 'treatment'").count()[0]


# d. What is $n_{old}$?

# In[30]:


n_old = df2.query("group == 'control'").count()[0]
n_new ,n_old


# e. Simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[31]:


new_page_converted = np.random.choice([1, 0], size=n_new, p=[pnew, 1-pnew])
    
    


# f. Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[32]:


old_page_converted = np.random.choice([1, 0], size=n_old, p=[pold, 1-pold])


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[33]:


new_page_converted.mean()-old_page_converted.mean()


# h. Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated in parts **a. through g.** above.  Store all 10,000 values in **p_diffs**.

# In[34]:


p_diffs =[]
for i in range(10000):
    new_page_converted = np.random.choice([1, 0], size=n_new, p=[pnew, 1-pnew])
    old_page_converted = np.random.choice([1, 0], size=n_old, p=[pold, 1-pold])
    p_diffs.append(new_page_converted.mean()-old_page_converted.mean())


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[35]:



plt.hist(p_diffs)


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[36]:


diff = df2.query("group == 'treatment'")['converted'].mean() - df2.query("group == 'control'")['converted'].mean()
p_diffs=np.array(p_diffs)
(diff < p_diffs).mean()


# k. In words, explain what you just computed in part **j.**.  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# this is the p-value 
# we could not reject the nul hypothises 

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[37]:


import statsmodels.api as sm

convert_old = sum(df2.query("group == 'control'")['converted'])
convert_new =  sum(df2.query("group == 'treatment'")['converted'])
n_old =  df2.query("group == 'control'").count()[0]
n_new =  df2.query("group == 'treatment'").count()[0]


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](http://knowledgetack.com/python/statsmodels/proportions_ztest/) is a helpful link on using the built in.

# In[38]:


z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new])
z_score, p_value


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# yes that agree an ensure that we could not reject the null 

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you acheived in the previous A/B test can also be acheived by performing regression.<br><br>
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# Logistic Regression
# 
# 

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives.  However, you first need to create a colun for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[40]:


df2['intercept']=1
df2[['control', 'treatment']] = pd.get_dummies(df2['group'])


# c. Use **statsmodels** to import your regression model.  Instantiate the model, and fit the model using the two columns you created in part **b.** to predict whether or not an individual converts.

# In[41]:


logit = sm.Logit(df2['converted'],df2[['intercept','treatment']])


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[42]:


results = logit.fit()
results.summary()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in the **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in the **Part II**?

# Our hypothes
# $H_{0}$ : $p_{new}$ - $p_{old}$ = 0
# $H_{1}$ : $p_{new}$ - $p_{old}$ != 0
# 
# the p-value associated with ab_page is 	0.190
# i think because this is two sideds test and the value you found in the Part II was one sided test.

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# 

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives. You will need to read in the **countries.csv** dataset and merge together your datasets on the approporiate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy varaibles.** Provide the statistical output as well as a written response to answer this question.

# In[43]:


df3 = pd.read_csv('./countries.csv')
df4 = df3.set_index('user_id').join(df2.set_index('user_id'), how='inner')
df4[['CA', 'US']] = pd.get_dummies(df4['country'])[['CA','US']]

df4['country'].astype(str).value_counts()


# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[44]:


df['intercept'] = 1


log = sm.Logit(df4['converted'], df4[['CA', 'US']])
r = log.fit()
r.summary()


# <a id='conclusions'></a>
# ## Finishing Up
# 
# > Congratulations!  You have reached the end of the A/B Test Results project!  This is the final project in Term 1.  You should be very proud of all you have accomplished!
# 
# > **Tip**: Once you are satisfied with your work here, check over your report to make sure that it is satisfies all the areas of the rubric (found on the project submission page at the end of the lesson). You should also probably remove all of the "Tips" like this one so that the presentation is as polished as possible.
# 
# 
# ## Directions to Submit
# 
# > Before you submit your project, you need to create a .html or .pdf version of this notebook in the workspace here. To do that, run the code cell below. If it worked correctly, you should get a return code of 0, and you should see the generated .html file in the workspace directory (click on the orange Jupyter icon in the upper left).
# 
# > Alternatively, you can download this report as .html via the **File** > **Download as** submenu, and then manually upload it into the workspace directory by clicking on the orange Jupyter icon in the upper left, then using the Upload button.
# 
# > Once you've done this, you can submit your project by clicking on the "Submit Project" button in the lower right here. This will create and submit a zip file with this .ipynb doc and the .html or .pdf version you created. Congratulations!

# In[ ]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])

