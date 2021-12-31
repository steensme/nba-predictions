# https://medium.com/analytics-vidhya/how-to-web-scrape-tables-online-using-python-and-beautifulsoup-36d5bafeb982

import requests
from bs4 import BeautifulSoup
import pandas as pd

url = 'https://www.espn.com/nba/stats/team/_/view/team/table/offensive/sort/fieldGoalPct/dir/desc"'
requests.get(url)
page = requests.get(url)

soup = BeautifulSoup(page.text, 'lxml')
# print(soup)

table_data = soup.find('table', class_ = 'Table Table--align-right')

headers = []
for i in table_data.find_all('th'):
    title = i.text
    headers.append(title)

team_names = []

team = soup.find('tbody')
for teamlogo in team.find_all('div', class_='flex items-start mr7'):
  teamname = teamlogo.find('img', class_='Image Logo Logo__sm')['title']
  team_names.append(teamname)

stats = pd.DataFrame(columns = headers)

for j in table_data.find_all('tr')[1:]:
        row_data = j.find_all('td')
        row = [tr.text for tr in row_data]
        length = len(stats)
        stats.loc[length] = row

stats.insert(0, 'Team', team_names)

# read in the scores data
scores = pd.read_csv('nov_dec_scores.csv')

scores['result_v'] = scores.iloc[:,3] - scores.iloc[:,5]

# make positive result a win for the visitor and a loss otherwise
scores.loc[scores['result_v'] > 0, 'visitor'] = 'W'
scores.loc[scores['result_v'] < 0, 'visitor'] = "L"

# make negative result v a win for the home and a loss otherwise
scores.loc[scores['result_v'] > 0, 'home'] = "W"
scores.loc[scores['result_v'] < 0, 'home'] = "L"

# Change Los Angeles Clippers to LA Clippers
scores["Visitor/Neutral"] = scores["Visitor/Neutral"].replace("Los Angeles Clippers", "LA Clippers", regex=True)
scores["Home/Neutral"] = scores["Home/Neutral"].replace("Los Angeles Clippers", "LA Clippers", regex=True)

# Make a table with only the teams, whether they were @ home and the win or loss
result1 = pd.DataFrame({'Team':scores['Visitor/Neutral'],'Site': 'V','result':scores['visitor']})
result2 = pd.DataFrame({'Team':scores['Home/Neutral'],'Site': 'H','result':scores['home']})

# combine the tables to get a probability of win based on stats
results = pd.concat([result1,result2])

# add the statistical data to the teams
merged = pd.merge(results, stats, on='Team', how='left')

# Select only the attributes that it makes sense to study
import numpy as np
results_stats = pd.DataFrame({'Site':merged['site'].astype("category"), 'PTS':pd.to_numeric(merged['PTS']), 'FG%':pd.to_numeric(merged['FG%']), 
                              '3P%':pd.to_numeric(merged['3P%']), 'FT%':pd.to_numeric(merged['FT%']), 'OR':pd.to_numeric(merged['OR']), 
                              'DR':pd.to_numeric(merged['DR']), 'AST':pd.to_numeric(merged['AST']), 'STL':pd.to_numeric(merged['STL']), 
                              'BLK':pd.to_numeric(merged['BLK']), 'TO':pd.to_numeric(merged['TO']), 'PF':pd.to_numeric(merged['PF']), 
                              'Result':merged['result'].astype("category")})
results_stats.Site = results_stats.Site.map({'V': 0, 'H':1})
results_stats.Result = results_stats.Result.map({'L': 0, 'W':1})

# Method from https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python

# cols=['Site', 'PTS', 'FG%', '3P%', 'FT%', 'OR', 'DR', 'AST', 'STL', 'BLK', 'TO', 'PF']
cols=['Site', 'FG%', '3P%', 'OR', 'AST', 'TO']

X = results_stats[cols]
y = results_stats.Result

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

# import the class
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression(max_iter=400) # needed to add this for the model to converge

# fit the model with data
logreg.fit(X_train,y_train)

# Make predictions on the test set
y_pred=logreg.predict(X_test)

# Scrape the current days games
url = 'https://www.espn.com/nba/schedule'
requests.get(url)
page = requests.get(url)

soup = BeautifulSoup(page.text, 'lxml')
# print(soup)

table_data = soup.find('table', class_ = 'schedule has-team-logos align-left')

team_names = []

schedule = soup.find('tbody')

for team in schedule.find_all("abbr"):
  team_name = team['title']
  team_names.append(team_name)

# print(team_names)

# Break the list into several lists
games = pd.DataFrame([team_names[i:i + 2] for i in range(0, len(team_names), 2)]) # help from https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks

visitor = pd.DataFrame({'Team': games.iloc[:,0], 'Site': 'V'})
visitor_stats = pd.merge(visitor, stats, on='Team', how='left')
home = pd.DataFrame({'Team': games.iloc[:,1], 'Site': 'H'})
home_stats = pd.merge(home, stats, on='Team', how='left')

visitor_trans = pd.DataFrame({'Site':visitor_stats['Site'].astype("category"), 'PTS':pd.to_numeric(visitor_stats['PTS']), 'FG%':pd.to_numeric(visitor_stats['FG%']), 
                              '3P%':pd.to_numeric(visitor_stats['3P%']), 'FT%':pd.to_numeric(visitor_stats['FT%']), 'OR':pd.to_numeric(visitor_stats['OR']), 
                              'DR':pd.to_numeric(visitor_stats['DR']), 'AST':pd.to_numeric(visitor_stats['AST']), 'STL':pd.to_numeric(visitor_stats['STL']), 
                              'BLK':pd.to_numeric(visitor_stats['BLK']), 'TO':pd.to_numeric(visitor_stats['TO']), 'PF':pd.to_numeric(visitor_stats['PF'])})
visitor_trans.Site = visitor_trans.Site.map({'V': 0, 'H':1})
home_trans = pd.DataFrame({'Site':home_stats['Site'].astype("category"), 'PTS':pd.to_numeric(home_stats['PTS']), 'FG%':pd.to_numeric(home_stats['FG%']), 
                              '3P%':pd.to_numeric(home_stats['3P%']), 'FT%':pd.to_numeric(home_stats['FT%']), 'OR':pd.to_numeric(home_stats['OR']), 
                              'DR':pd.to_numeric(home_stats['DR']), 'AST':pd.to_numeric(home_stats['AST']), 'STL':pd.to_numeric(home_stats['STL']), 
                              'BLK':pd.to_numeric(home_stats['BLK']), 'TO':pd.to_numeric(home_stats['TO']), 'PF':pd.to_numeric(home_stats['PF'])})
home_trans.Site = home_trans.Site.map({'V': 0, 'H':1})

v_prob = logreg.predict_proba(visitor_trans[cols])
h_prob = logreg.predict_proba(home_trans[cols])

games_prob = pd.DataFrame({'Visitor': games.iloc[:,0], 'Prob_V': v_prob[:,0], 'Home': games.iloc[:,1], 'Prob_H':h_prob[:,0]})

prediction = []
margin = []
for row in games_prob.iterrows():
  if row[1].Prob_V > row[1].Prob_H:
    prediction.append(row[1].Visitor)
    margin.append(row[1].Prob_V - row[1].Prob_H)
  else:
    prediction.append(row[1].Home)
    margin.append(row[1].Prob_H - row[1].Prob_V)
	
games_pred = pd.DataFrame({'Visitor': games.iloc[:,0], 'Home': games.iloc[:,1], 'Prediction': prediction, 'Margin': margin})

games_pred
