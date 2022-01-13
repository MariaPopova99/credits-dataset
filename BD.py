import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import precision_score,recall_score, auc, accuracy_score, roc_auc_score, f1_score, classification_report, roc_curve
import warnings
warnings.filterwarnings('ignore')



data = pd.read_csv('credit.dataset.csv')
# print(data.describe().round(2).to_string()) #cols and string
# print(data.duplicated().sum()) #duplicate
# print(data.isnull().sum()) #checking nulls
# print(data.title.to_string()) #full column
# print(data['title'][5]) #returns the value of column title string 5
# print(data.loc[12]) #returns the string with index+1
# print(data.head(7).to_string()) #first 7 strings
# print(data.tail(3).to_string()) #last 3 strings
# print(data.nunique()) #unique value
# print(data.salary.unique()) #unique in salary
# print(data.loc[data.title == 'R', ['id','title', 'age']].to_string()) #filter data by title and returns colums id title and age only
'''print person who has car and with age <=20
    and return age, transport, salary, nationality'''
# age_transport=(data.age <= 20) & (data.transport == 'Car')
# print(data.loc[age_transport,['age', 'transport','salary', 'nationality']].to_string())
'''rename the column'''

# rename=data.rename(columns={'age' : 'AgE'})
# print(rename.head().to_string())

# print(data.filter(like='ag')) #filter for columns containing 'ag'
# print(data.filter(like = '27', axis=0)) #filter for strings

# print(data.groupby('title').mean().to_string()) #group by title and mean for other columns

'''mean for income and salary group by region and rename the columns'''
# print(data.groupby('region').aggregate({'income': 'mean', 'salary':'mean'}).rename(columns={'income': 'mean_income', 'salary' : 'mean_salary'}))

# print(data.salary.unique()) #unique values in salary column
# print(data.sort_values(['title','salary'])) #sorting values

'''creating new columns'''
# data['incomeANDsalary']=data.income + data.salary
# print(data.head().to_string())

'''delete columns (for them axis =1, for strings axis =0'''
# print(data.drop(['title'], axis =1))

#check unique data in dataset
# print(data.apply(lambda x:len(x.unique())))

#check for categorical attributes
categ = []
for x in data.dtypes.index:
    if data.dtypes[x] == 'object':
        categ.append(x)
# print(categ)
# #check the unique values
# for col in categ:
#     print(data[col].value_counts())


'''VISUALIZATION (graphs)'''
# data.salary.hist()
# plt.show()
'''
data.plot.scatter(x='id', y='amount of children')
data.plot.scatter(x='id', y='time living the same address')
data.plot.scatter(x='id', y='time in job')
data.plot.scatter(x='id', y='telephone')
data.plot.scatter(x='id', y='num of bank loans')
data.plot.scatter(x='id', y='num finished loans')
data.plot.scatter(x='id', y='EC-card holders')
data.plot.scatter(x='id', y='income')
data.plot.scatter(x='id', y='salary')
data.plot.scatter(x='id', y='salary+ec-cards')
data.plot.scatter(x='id', y='credit risk')
data.plot.scatter(x='id', y='running loans')
data.plot.scatter(x='id', y='region')
data.plot.scatter(x='id', y='requested cash')
data.plot.scatter(x='id', y='nationality')
data.plot.scatter(x='id', y='good/bad')
plt.show()

'''
# print(data.corr().to_string()) #corr matrix

# heat map regression
cols=[i for i in data if i not in categ]
hm = sns.heatmap(data[cols].corr(), cbar = True, annot = True)
plt.show()


# 14.12
'''NEW column as salary = salary = income + salary+ec-cards'''

data=data.drop(np.where(data['amount of children'] == 6)[0])
data['new_income'] = data.salary + data.income
# + data['salary+ec-cards']
del data['salary']
del data['income']
del data['salary+ec-cards']
# print (data.head().to_string())

data_to_check_rich = data.loc[data['requested cash'] == 100000, ['requested cash','new_income','good/bad','age','running loans', 'credit risk',
                                                  'num finished loans', 'num of bank loans', 'time in job','time living the same address']]
print(data_to_check_rich.to_string())

# new heat map
# cols=[i for i in data if i not in categ]
# hm = sns.heatmap(data[cols].corr(), cbar = True, annot = True)
# print(hm)
# plt.show()

# ploting new dataset
'''
ax = plt.gcf()
x=[i for i in range(1,12)]

# data_to_check_rich.plot(x=x, y='good/bad')
plt.plot(x, data_to_check_rich['good/bad'], 'ro')
plt.plot(x, data_to_check_rich['time living the same address'], 'ro')


plt.show()
'''
# replace data to nums
set_nums={'title':{'H':1,'R':2},
          'type of business':{'Radio, TV, Hifi':1,'Furniture,Carpet':2,
                              'Dept. Store,Mail': 3, 'Cars':4,'Leisure':5,'Others':6},
          'resident type':{'Owner':1, 'Lease':2},
          'nationality':{'German':1,'Turkish':2,'Others':3,'Greek':4,
                         'Other European':5,'Yugoslav':6, 'Italian': 7, 'Spanish/Portugue':8},
          'profession':{'Others':1, 'Civil Service, M':2,'Food,Building,Ca':3,
                        'Pensioner':4,'Self-employed pe':5,'Military Service':6,
                        'State,Steel Ind,':7, 'Chemical Industr':8,'Sea Vojage, Gast':9},
          'transport': {'Car':1, 'Without Vehicle':2,'Car and Motor bi':3},
          'credit cards':{'no credit cards':1,'Cheque card':2, 'Mastercard/Euroc':3,
                          'Other credit car':4, 'VISA mybank':5, 'VISA Others':6, 'American Express':7}
          }
'''dataset without categor'''
data = data.replace(set_nums)
print(data.head(7).to_string())

# # check categ again
# categ = []
# for x in data.dtypes.index:
#     if data.dtypes[x] == 'object':
#         categ.append(x)
# print(categ)
# data.to_csv('new_dataset')
'''
# trying to do linearregression

plt.scatter(data['requested cash'], data['new_income'],alpha=0.5)
plt.xlabel('request')
plt.ylabel('income')
plt.ylim(0)
plt.xlim(0)
plt.show()
model = LinearRegression()
x=pd.DataFrame(data['requested cash'])
y=pd.DataFrame(data['new_income'])
model.fit(x,y)
print(model.coef_)
print(model.intercept_)

plt.scatter(data['requested cash'], data['new_income'],
            alpha=0.5,color='blue',label ='+')
plt.scatter(data['age'],data['amount of children'],alpha=0.5,
            color= 'red', label = '-')
plt.plot(x,model.predict(x), color='red',linewidth =2)
plt.show()
# check the model
model.score(x,y)
model.predict([[50000]])
'''
plt.figure(figsize=(10,6))
plt.hist(data['age'], bins=50, ec = 'black')
plt.xlabel('age')
plt.ylabel('nb of people')
plt.show()

plt.hist(data['requested cash'], bins=50, ec = 'darkblue')
plt.xlabel('requested cash')
plt.show()

# for multiple histogram
# x = data.age<30
# y = data['num finished loans']
# data.plot.bar(x,y)
# plt.show()

# age for good/bad
sns.displot(
    {
        0: data[data['good/bad']==0].age,
        1:data[data['good/bad']==1].age
    },
    kind = 'kde',
    common_norm= False
)
plt.title('age')
plt.xlabel('age',fontsize = 13)
plt.ylabel('density',fontsize = 13)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()
print(data.groupby('good/bad')['age'].median())
print(data.groupby('good/bad')['age'].mean())

# salary for good/bad borrower
sns.displot(
    data,x='new_income', hue = 'good/bad', kind='kde', common_norm=False)
plt.show()

# check mean income with considering time in job
mean_inc_job = data.groupby('time in job')['new_income'].median().to_dict()
data['mean_inc_job'] = data['time in job'].map(mean_inc_job)
print(data.head().to_string())