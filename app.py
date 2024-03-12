from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

df = pd.read_csv('a.csv')

# df.head(5)

# df.describe()

# df.columns

# df.info()

# df.sample(n=10,random_state=32)

# df.shape

"""## Pre Processing"""

# df.isnull().sum()

# changing the experience_level into full form
# changing the employment_type into full formm

df['experience_level']=df['experience_level'].str.replace('SE','Senior-level')
df['experience_level']=df['experience_level'].str.replace('MI','Mid-level')
df['experience_level']=df['experience_level'].str.replace('EN','Entry-level')
# df

df['employment_type']=df['employment_type'].str.replace('FT','Full-time')
df['employment_type']=df['employment_type'].str.replace('CT','Contract')
df['employment_type']=df['employment_type'].str.replace('PT','Part-level')
df['employment_type']=df['employment_type'].str.replace('FL','Freelance')
# df.head(10)

# df.tail(10)

fig, axes = plt.subplots(1, 2, figsize=(15, 7))


sns.countplot(ax=axes[0], x='experience_level', data=df)
axes[0].set_title('Experience level countplot')


df_salvslevel=df.iloc[:,1:5:3]
df_salvslevel


sns.boxplot(df_salvslevel,x='experience_level',y='salary')
axes[1].set_title('Experience level countplot')

plt.savefig('static/1.png')

fig,axes = plt.subplots(1, 3, figsize=(20, 7))

# Subplot 1: Countplot of experience_level
sns.countplot(ax=axes[0], x='experience_level', data=df)
axes[0].set_title('Experience level countplot')

# Subplot 2: Boxplot of salary vs experience_level
sns.boxplot(ax=axes[1], x='experience_level', y='salary', data=df)
axes[1].set_title('Salary vs Experience level boxplot')

# Calculate mean salary for each experience level
mean_salary_by_experience = df.groupby('experience_level')['salary'].mean().reset_index()

# Subplot 3: Barplot of mean salary vs experience_level
sns.barplot(ax=axes[2], x='experience_level', y='salary', data=mean_salary_by_experience, palette='viridis')
axes[2].set_title('Mean Salary by Experience Level')

plt.savefig('static/2.png')

job_counts = df['job_title'].value_counts()


top_10_jobs = job_counts.head(10)
other_jobs = job_counts[10:].sum()


job_labels = top_10_jobs.index.tolist()
job_labels.append('other')
job_sizes = top_10_jobs.tolist()
job_sizes.append(other_jobs)


colors = plt.cm.tab20c.colors


plt.figure(figsize=(8, 6))
plt.pie(job_sizes, labels=job_labels, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Top 10 Job Titles Distribution')
plt.axis('equal')
plt.savefig('static/3.png')

# top_10_jobs = df['job_title'].value_counts().head(10).index.tolist()
# df_top_10_jobs = df[df['job_title'].isin(top_10_jobs)]

df = df.assign(salary_in_inr=(df["salary_in_usd"] * 82.91))

# # Create subplots
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

# # Box Plot for Salary in INR
# axes[0].boxplot([df_top_10_jobs[df_top_10_jobs['job_title'] == job]['salary_in_inr'] for job in top_10_jobs], labels=top_10_jobs)
# axes[0].set_title('Box Plot of Top 10 Job Titles and Salary Distribution (INR)')
# axes[0].set_xlabel('Job Title')
# axes[0].set_ylabel('Salary in INR')
# axes[0].tick_params(axis='x', rotation=45)
# axes[0].yaxis.set_major_formatter(ScalarFormatter())

# # Box Plot for Salary in USD
# axes[1].boxplot([df_top_10_jobs[df_top_10_jobs['job_title'] == job]['salary_in_usd'] for job in top_10_jobs], labels=top_10_jobs)
# axes[1].set_title('Box Plot of Top 10 Job Titles and Salary Distribution (USD)')
# axes[1].set_xlabel('Job Title')
# axes[1].set_ylabel('Salary in USD')
# axes[1].tick_params(axis='x', rotation=45)
# axes[1].yaxis.set_major_formatter(ScalarFormatter())

# plt.tight_layout()
# plt.show()

bins = [0, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000]
labels = ['0-50k', '50k-100k', '100k-150k', '150k-200k', '200k-250k', '250k-300k', '300k-350k', '350k-400k', '400k-450k']
df_new = df[df['company_location'].isin(['US', 'GB', 'CA', 'ES', 'IN'])].copy()
df_new.loc[:, 'salary_range'] = pd.cut(df_new['salary_in_usd'], bins=bins, labels=labels)


df_new = df_new.sort_values(by=['company_location', 'salary_range'])



heatmap_data = df_new.groupby(['company_location', 'salary_range']).size().unstack(fill_value=0)


plt.figure(figsize=(10, 6))

sns.heatmap(heatmap_data.T[::-1], cmap='YlGnBu', annot=True, fmt='d', xticklabels=df_new['company_location'].unique(), yticklabels=labels[::-1])
plt.title('Salary Distribution by Company Location and Range')
plt.xlabel('Company Location')
plt.ylabel('Salary Range (USD)')
plt.savefig('static/4.png')

grouped_df = df.groupby('company_size').agg({
    'salary_in_inr': 'mean',
    'employment_type': pd.Series.mode,
    'experience_level': pd.Series.mode
}).reset_index()


plt.figure(figsize=(10, 6))
sns.barplot(x='company_size', y='salary_in_inr', data=grouped_df)
plt.xlabel('Company Size')
plt.ylabel('Salary (INR)')
plt.title('Average Salary by Company Size')
plt.savefig('static/5.png')

emp_india_earn_usd=df.query(" salary_currency=='USD' and employee_residence=='IN' ")
emp_india_earn_inr=df.query(" salary_currency=='INR' and employee_residence=='IN' ")

# print(df['salary_in_inr'])

print(len(emp_india_earn_inr),len(emp_india_earn_usd))

mean_salries=[emp_india_earn_usd['salary_in_inr'].mean(),emp_india_earn_inr['salary_in_inr'].mean()]
country=["Earned in USD","Earned in INR"]

print(mean_salries)
plt.bar(country,mean_salries)
plt.ylabel("In 10 lakhs")
plt.savefig('static/6.png')

# print(emp_india_earn_usd['salary_in_inr'].sum())
# print(emp_india_earn_inr['salary_in_inr'].sum())
average_salary = df.groupby(["experience_level", "employment_type"])["salary_in_usd"].mean().reset_index().sort_values(by='salary_in_usd', ascending=False)
average_salary

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.bar(range(len(average_salary)), average_salary['salary_in_usd'], color='skyblue')
plt.xticks(range(len(average_salary)), average_salary['experience_level'] + " - " + average_salary['employment_type'], rotation=45, ha="right")
plt.title('Average Salary by Experience Level and Employment Type')
plt.xlabel('Experience Level - Employment Type')
plt.ylabel('Average Salary (USD)')
plt.savefig('static/7.png')

# Median salary by years for entries, mids, seniors and executives
# What are the top 5 company locations to start as a data scientist?
# What are 5 top company locations to work as a senior/executive specialist?
# What are the salary trends?

"""# What are the salary trends?

"""

df_filtered_usd = df[['work_year', 'experience_level', 'salary_in_usd', 'salary_currency']]
df_filtered_temp_usd = df_filtered_usd.query("salary_currency=='USD'")
df_filtered_usd = df_filtered_temp_usd.drop(columns=['salary_currency'])

# Group by year and experience level, then calculate median salary
df_result_usd = df_filtered_usd.groupby(['work_year', 'experience_level']).median().reset_index()

# Pivot the table for better visualization
df_pivot_usd = df_result_usd.pivot(index='work_year', columns='experience_level', values='salary_in_usd')

# Median salary by years for entries, mids, seniors, and executives for India
df_filtered_inr = df[['work_year', 'experience_level', 'salary_in_inr', 'salary_currency']]
df_filtered_temp_inr = df_filtered_inr.query("salary_currency=='INR'")
df_filtered_inr = df_filtered_temp_inr.drop(columns=['salary_currency'])

# Group by year and experience level, then calculate median salary
df_result_inr = df_filtered_inr.groupby(['work_year', 'experience_level']).median().reset_index()

# Pivot the table for better visualization
df_pivot_inr = df_result_inr.pivot(index='work_year', columns='experience_level', values='salary_in_inr')

# Create subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

# Plot for USA
df_pivot_usd.plot(kind='line', marker='o', ax=axes[0])
axes[0].set_title("Median Salary Trends (USA)")
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Median Salary, USD")
axes[0].legend(title='Experience Level')

# Plot for India
df_pivot_inr.plot(kind='line', marker='o', ax=axes[1])
axes[1].set_title("Median Salary Trends (India)")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Median Salary, INR")
axes[1].legend(title='Experience Level')

plt.tight_layout()
plt.savefig('static/8.png')
#-----------------------------------------------------------------------------------------
# Feature Engineering
df = df.assign(salary_in_inr=(df["salary_in_usd"] * 82.91))
df_model = df.copy()
df_model = df_model.drop(columns=['job_title', 'salary_currency', 'salary_in_inr', 'salary', 'company_location', 'employee_residence'])

le = LabelEncoder()
df_model['employment_type'] = le.fit_transform(df_model['employment_type'])
print(df_model['employment_type'])
df_model['experience_level'] = le.fit_transform(df_model['experience_level'])
df_model['company_size'] = le.fit_transform(df_model['company_size'])

X = df_model.drop(columns=['salary_in_usd'])
y = df_model['salary_in_usd']

# Train the model
rf = RandomForestRegressor(n_estimators=100, random_state=32)
rf.fit(X, y)



app = Flask(__name__)


@app.route('/')
def index():
    return render_template('add.html')

@app.route('/obser')
def add():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        year = int(request.form['year'])
        experience_level = int(request.form['experience_level'])
        employment_type = int(request.form['employment_type'])
        company_size = int(request.form['company_size'])
        remote_ratio = int(request.form['remote_ratio'])
        print(year, experience_level, employment_type, company_size, remote_ratio)

        # remote_ratio
        # 0 
        # 50
        # 100

        # encoded experience_level
        # 1 - Entry-level
        # 2 - Mid-level
        # 3- Senior-level
        # 4- Executive-level

        # encoded employment_type
        # 0 - Contract
        # 1 - Freelance
        # 2 - Full-time
        # 3 - Part-time

        # encoded company_size
        # 0 - L
        # 1 - M
        # 2 - S


        test_data = [[year, experience_level, employment_type, company_size, remote_ratio]]
        predicted_salary = rf.predict(test_data)[0]
        predicted_salary_formatted = format(predicted_salary, '.2f')

        if(experience_level == 1):
            experience_level = 'Entry-level'
        elif(experience_level == 2):
            experience_level = 'Mid-level'
        elif(experience_level == 3):
            experience_level = 'Senior-level'
        elif(experience_level == 4):
            experience_level = 'Executive-level'

        if(employment_type == 0):
            employment_type = 'Contract'
        elif(employment_type == 1):
            employment_type = 'Freelance'
        elif(employment_type == 2):
            employment_type = 'Full-time'
        elif(employment_type == 3):
            employment_type = 'Part-time'

        if(company_size == 0):
            company_size = 'Large'
        elif(company_size == 1):
            company_size = 'Medium'
        elif(company_size == 2):
            company_size = 'Small'
    

        return render_template('result.html', year=year, experience_level=experience_level, employment_type=employment_type,company_size=company_size, remote_ratio=remote_ratio, predicted_salary=predicted_salary_formatted)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
