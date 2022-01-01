# Required Packages:
# install requests
# install bs4
# install pandas
# install lxml
# install -U scikit-learn

# run program: python nba_predictions.py

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

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

# Scrape the games from November
url = 'https://www.basketball-reference.com/leagues/NBA_2022_games-november.html'
requests.get(url)
page = requests.get(url)

soup = BeautifulSoup(page.text, 'lxml')
# print(soup)

table_data = soup.find('table', id = 'schedule') #class_ = 'suppress_glossary sortable stats_table now_sortable')

headers = []
for i in table_data.thead.find_all('th'):
    title = i.text
    headers.append(title)

scores_nov = pd.DataFrame(columns = headers[1:]) # Leave out 'Date' column becaues this is hard to scrape ('th' tag, not 'td')

for j in table_data.tbody.find_all('tr'):
  row_data = j.find_all('td')
  row = [tr.text for tr in row_data]
  length = len(scores_nov)
  scores_nov.loc[length] = row
  
 # Scrape the games from December
url = 'https://www.basketball-reference.com/leagues/NBA_2022_games-december.html'
requests.get(url)
page = requests.get(url)

soup = BeautifulSoup(page.text, 'lxml')
# print(soup)

table_data = soup.find('table', id = 'schedule') #class_ = 'suppress_glossary sortable stats_table now_sortable')

headers = []
for i in table_data.thead.find_all('th'):
    title = i.text
    headers.append(title)

scores_dec = pd.DataFrame(columns = headers[1:]) # Leave out 'Date' column becaues this is hard to scrape ('th' tag, not 'td')

for j in table_data.tbody.find_all('tr'):
  row_data = j.find_all('td')
  row = [tr.text for tr in row_data]
  length = len(scores_dec)
  scores_dec.loc[length] = row
  
# Join the scores data
scores = pd.concat([scores_nov,scores_dec])

# Drop Games that do not have scores yet
scores['PTS'] = scores['PTS'].replace('', np.nan, inplace=False)
scores = scores.dropna(subset=['PTS'], inplace=False)

# Convert 'PTS' columns into integers and subtract them to see the score margin
scores['result_v'] = scores.iloc[:,2].astype("int64") - scores.iloc[:,4].astype("int64")

# make positive result a win for the visitor and a loss otherwise
scores.loc[scores['result_v'] > 0, 'visitor'] = 'W'
scores.loc[scores['result_v'] < 0, 'visitor'] = "L"

# make negative result v a win for the home and a loss otherwise
scores.loc[scores['result_v'] < 0, 'home'] = "W"
scores.loc[scores['result_v'] > 0, 'home'] = "L"

# Change Los Angeles Clippers to LA Clippers
scores["Visitor/Neutral"] = scores["Visitor/Neutral"].replace("Los Angeles Clippers", "LA Clippers", regex=True)
scores["Home/Neutral"] = scores["Home/Neutral"].replace("Los Angeles Clippers", "LA Clippers", regex=True)

# Make a table with only the teams, whether they were @ home and the win or loss
result1 = pd.DataFrame({'Team':scores['Visitor/Neutral'],'Site': 'V','result':scores['visitor']})
result2 = pd.DataFrame({'Team':scores['Home/Neutral'],'Site': 'H','result':scores['home']})

# combine the tables to get a probability of win based on stats
results = pd.concat([result1,result2])

# add the statistical data to the teams
teams_stats = pd.merge(results, stats, on='Team', how='left')

# Select only the attributes that it makes sense to study
results_stats = pd.DataFrame({'Site':teams_stats['Site'].astype("category"), 'PTS':pd.to_numeric(teams_stats['PTS']), 
                                'FG%':pd.to_numeric(teams_stats['FG%']), '3P%':pd.to_numeric(teams_stats['3P%']), 
                                'FT%':pd.to_numeric(teams_stats['FT%']), 'OR':pd.to_numeric(teams_stats['OR']), 
                                'DR':pd.to_numeric(teams_stats['DR']), 'AST':pd.to_numeric(teams_stats['AST']), 
                                'STL':pd.to_numeric(teams_stats['STL']), 'BLK':pd.to_numeric(teams_stats['BLK']), 
                                'TO':pd.to_numeric(teams_stats['TO']), 'PF':pd.to_numeric(teams_stats['PF']), 
                                'Result':teams_stats['result'].astype("category")})
results_stats.Site = results_stats.Site.map({'V': 0, 'H':1})
results_stats.Result = results_stats.Result.map({'L': 0, 'W':1})

# Check for null values in the data set
is_NaN = teams_stats.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = teams_stats[row_has_NaN]
print('If the index is not empty there are missing values in the table')
print(rows_with_NaN)
print()

# Note: depending on the accuracy of the testing of the model, the columns included can be adjusted
cols=['Site', 'PTS', 'FG%', '3P%', 'FT%', 'OR', 'DR', 'AST', 'STL', 'BLK', 'TO', 'PF']
# cols=['Site', 'FG%', '3P%', 'OR', 'AST', 'TO']

X = results_stats[cols]
y = results_stats.Result

# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

# Set up the model
from sklearn.linear_model import LogisticRegression

# instantiate the model (using the default parameters)
logreg = LogisticRegression(max_iter=500) # needed to add this for the model to converge

# fit the model with data
logreg.fit(X_train,y_train)

# Make predictions on the test set
y_pred=logreg.predict(X_test)

# import the metrics class to test the model accuracy
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
print()
print('accuracy of model for teams on regardless of opponent on test data: ')
print((cnf_matrix[0,0] + cnf_matrix[1,1]) / (cnf_matrix[0,0] + cnf_matrix[0,1] + cnf_matrix[1,0] + cnf_matrix[1,1]))
print()

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

games_prob = pd.DataFrame({'Visitor': games.iloc[:,0], 'Prob_V': v_prob[:,1], 'Home': games.iloc[:,1], 'Prob_H':h_prob[:,1]})

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

# Print out the prediction and the margin of difference in probabilities
print('Matchup, prediction and predicted margin in probability of winning:')
print(games_pred)
