get_ipython().system('pip install nba_api')

get_ipython().run_line_magic('matplotlib', 'inline')
import re
import time
import math
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nba_api.stats.endpoints import PlayerGameLogs, TeamGameLogs

def gamedatapull():
    player = pd.DataFrame()  # Initialize an empty dataframe to store the combined data
    team = pd.DataFrame()  # Initialize an empty dataframe to store the combined data

    for i in range(2):#24):
        season = f"{2000 + i}-{str(2000 + i + 1)[2:]}" # Right Season

        # Get player game logs data
        player_game_logs = PlayerGameLogs(
            season_nullable=season, # change year(s) if needed
            season_type_nullable='Regular Season') # Regular Season, Playoffs, Pre Season
        df_player_game_logs = player_game_logs.get_data_frames()[0]

        # Get team game logs data
        team_game_logs = TeamGameLogs(
            league_id_nullable='00',  # nba 00, g_league 20, wnba 10
            team_id_nullable='',  # You can specify a specific team_id if needed
            season_nullable=season, # change year(s) if needed
            season_type_nullable='Regular Season') # Regular Season, Playoffs, Pre Season
        df_team_game_logs = team_game_logs.get_data_frames()[0]

        # Concatenate the merged dataframe to the combined dataframe
        player = pd.concat([df_player_game_logs, player], ignore_index=True)
        team =  pd.concat([df_team_game_logs, team], ignore_index=True)

    return player, team
    
# Call the function to retrieve the combined dataframe
player, T3aM = gamedatapull()

# PFTkn
def player_dict(combined_dataframe):
  combined_dataframe['Opposing_Team'] = combined_dataframe['MATCHUP'].str[-3:]
  combined_dataframe['PPP'] = combined_dataframe['PTS'] / (combined_dataframe['FGA'] + 0.44 * combined_dataframe['FTA'] + combined_dataframe['TOV'])
  combined_dataframe['Ast/TO'] = combined_dataframe['AST'].astype(str) + "/" + combined_dataframe['TOV'].astype(str)
  combined_dataframe['BlkFGA'] = combined_dataframe['BLK'].astype(str) + "/" + combined_dataframe['FGA'].astype(str)
  combined_dataframe['FGm'] = combined_dataframe['FGA'] - combined_dataframe['FGM']
  combined_dataframe['2FGA'] =  combined_dataframe['FGA'] - combined_dataframe['FG3A']
  combined_dataframe['2FGM'] = combined_dataframe['FGM'] - combined_dataframe['FG3M']
  combined_dataframe['2FGm'] = combined_dataframe['2FGA'] - combined_dataframe['2FGM']
  combined_dataframe['FG3m'] = combined_dataframe['FG3A'] - combined_dataframe['FG3M']
  combined_dataframe['FTm'] = combined_dataframe['FTA'] - combined_dataframe['FTM']
  combined_dataframe['SST'] = combined_dataframe['STL'].astype(str) + "/" + combined_dataframe['FGA'].astype(str)
  combined_dataframe['Pos'] = (combined_dataframe['FGA'] + 0.44 * combined_dataframe['FTA'] + combined_dataframe['TOV'])
  combined_dataframe['StlPos'] =  (combined_dataframe['STL'] * 100).astype(str) + "/" + combined_dataframe['Pos'].astype(str)
  combined_dataframe['PFCom'] = combined_dataframe['PF']
  combined_dataframe.drop(['Pos', 'AVAILABLE_FLAG'], axis=1, inplace=True)
  return combined_dataframe

player_stats = player_dict(player)

def split_game_matchup(game):
    if '@' in game:
        away, home = game.split('@')
    elif 'vs.' in game:
        home, away = game.split('vs.')
    return home.strip(), away.strip()

def convert_to_winner(row):
    if row['WL'] == 'W':
        return row['TEAM_ABBREVIATION']
    else:
        return row['Opposing_Team']

def count_sequences(string):
    # Find all sequences of consecutive 'W's and 'L's
    sequences = re.findall(r'(W+|L+)', string)

    # Count the number of occurrences and sum the lengths for each sequence type
    counts = {'W': 0, 'L': 0}
    for seq in sequences:
        counts[seq[0]] += len(seq)
    tup = str(counts['W']), " - ", str(counts['L'])
    y = ''.join(tup)
    return y

def next_func(df):
  df = df.sort_values('GAME_DATE')
  df['Opposing_Team'] = df['MATCHUP'].str[-3:]

  # Apply the function to create 'Winner' column
  df['Winner'] = df.apply(convert_to_winner, axis=1)

  # Apply the function to split the Game column into Home and Away teams
  df[['H_Team', 'A_Team']] = df['MATCHUP'].apply(lambda x: pd.Series(split_game_matchup(x)))

  df['Record'] = df.groupby(["SEASON_YEAR", "TEAM_NAME"], group_keys=False)['WL'].apply(lambda x: x.cumsum()).apply(count_sequences)
  df = df.sort_values('GAME_ID').reset_index()

  opp = list()
  home = list()
  away = list()
  homep = list()
  awayp = list()
  for i in range(len(df)):
    if i % 2 == 0:
      opp.append(df['Record'][i + 1])
    else:
      opp.append(df['Record'][i - 1])
  df['Opp_Record'] = opp

  for i in range(len(df)):
    if df['H_Team'][i] == df['TEAM_ABBREVIATION'][i]:
      home.append(df['Record'][i])
      away.append(df['Opp_Record'][i])
    else:
      home.append(df['Opp_Record'][i])
      away.append(df['Record'][i])

  df['H_Team_Record'] = home
  df['A_Team_Record'] = away

  return df
def countw(string):
    # Find all sequences of consecutive 'W's and 'L's
    sequences = re.findall(r'(W+|L+)', string)

    # Count the number of occurrences and sum the lengths for each sequence type
    counts = {'W': 0, 'L': 0}
    for seq in sequences:
        counts[seq[0]] += len(seq)
    tup = str(counts['W'])
    y = ''.join(tup)
    return y

def countl(string):
    # Find all sequences of consecutive 'W's and 'L's
    sequences = re.findall(r'(W+|L+)', string)

    # Count the number of occurrences and sum the lengths for each sequence type
    counts = {'W': 0, 'L': 0}
    for seq in sequences:
        counts[seq[0]] += len(seq)
    tup = str(counts['L'])
    y = ''.join(tup)
    return y


def split(df, home, away, target, Home_Col, Away_Col):

  df[Home_Col] = home[target]
  df[Away_Col] = away[target]

  home_list = list()
  away_list = list()

  for i in range(len(df) - 1):
    if pd.isna(df[Home_Col].iloc[i]):
      home_list.append(df[Home_Col][i + 1])

    if pd.isna(df[Away_Col].iloc[i]):
      away_list.append(df[Away_Col][i + 1])

    if not pd.isna(df[Home_Col].iloc[i]):
      home_list.append(df[Home_Col][i])

    if not pd.isna(df[Away_Col].iloc[i]):
      away_list.append(df[Away_Col][i])

  df[Home_Col] = pd.Series(home_list)
  df[Away_Col] = pd.Series(away_list)

  df[Home_Col] = df[Home_Col].fillna(method='ffill', inplace=False)
  df[Away_Col] = df[Away_Col].fillna(method='ffill', inplace=False)

  return df

def record_func(df):
  home = df[(df['TEAM_ABBREVIATION'] == df['H_Team'])]
  df['Home_Wins'] = home[(home['TEAM_ABBREVIATION'] == home['Winner'])].groupby(["SEASON_YEAR", "TEAM_NAME"], group_keys=False)['WL'].apply(lambda x: x.cumsum()).apply(countw)
  df['Home_Loss'] = home[(home['TEAM_ABBREVIATION'] != home['Winner'])].groupby(["SEASON_YEAR", "TEAM_NAME"], group_keys=False)['WL'].apply(lambda x: x.cumsum()).apply(countl)

  away = df[(df['TEAM_ABBREVIATION'] == df['A_Team'])]
  df['Away_Wins'] = away[(away['TEAM_ABBREVIATION'] == away['Winner'])].groupby(["SEASON_YEAR", "TEAM_NAME"], group_keys=False)['WL'].apply(lambda x: x.cumsum()).apply(countw)
  df['Away_Loss'] = away[(away['TEAM_ABBREVIATION'] != away['Winner'])].groupby(["SEASON_YEAR", "TEAM_NAME"], group_keys=False)['WL'].apply(lambda x: x.cumsum()).apply(countl)

  df['Home_Wins'] = df['Home_Wins'].fillna(method='ffill', inplace=False).fillna(0)
  df['Home_Loss'] = df['Home_Loss'].fillna(method='ffill', inplace=False).fillna(0)

  df['Away_Wins'] = df['Away_Wins'].fillna(method='ffill', inplace=False).fillna(0)
  df['Away_Loss'] = df['Away_Loss'].fillna(method='ffill', inplace=False).fillna(0)

  df['Home_Record'] = df['Home_Wins'].astype(str) + ' - ' + df['Home_Loss'].astype(str)
  df['Away_Record'] = df['Away_Wins'].astype(str) + ' - ' + df['Away_Loss'].astype(str)

  # Drop unnecessary columns
  df.drop(['Home_Wins', 'Home_Loss', 'Away_Wins', 'Away_Loss', 'AVAILABLE_FLAG'], axis=1, inplace=True)

  df = split(df, home, away, "PTS", "H_Points", "A_Points")
  df = split(df, home, away, "REB", "H_Total_Rebounds", "A_Total_Rebounds")
  df = split(df, home, away, "OREB", "H_Offensive_Rebounds", "A_Offensive_Rebounds")
  df = split(df, home, away, "DREB", "H_Defensive_Rebounds", "A_Defensive_Rebounds")
  df = split(df, home, away, "AST", "H_Assists", "A_Assists")
  df = split(df, home, away, "TOV", "H_Turnovers", "A_Turnovers")
  df = split(df, home, away, "STL", "H_Steals", "A_Steals")
  df = split(df, home, away, "BLK", "H_Blocks", "A_Blocks")
  df = split(df, home, away, "FTA", "H_All_Free_Throws", "A_All_Free_Throws")
  df = split(df, home, away, "FGA", "H_FG_Attempts", "A_FG_Attempts")
  df = split(df, home, away, "FGM", "H_FG_Made", "A_FG_Made")

  df['FGm'] = df['FGA'] - df['FGM']
  df['FG3m'] = df['FG3A'] - df['FG3M']
  df['2FG_Attempts'] = df['FGA'] - df['FG3A']
  df['2FG_Made'] = df['FGM'] - df['FG3M']
  df['2FG_Missed'] = df['FGm'] - df['FG3m']
  return df
def drop(df):
  home = df[(df['TEAM_ABBREVIATION'] == df['H_Team'])]
  away = df[(df['TEAM_ABBREVIATION'] == df['A_Team'])]
  df = split(df, home, away, "FGm", "H_FG_Missed", "A_FG_Missed")
  df = split(df, home, away, "2FG_Attempts", "H_2FG_Attempts", "A_2FG_Attempts")
  df = split(df, home, away, "2FG_Made", "H_2FG_Made", "A_2FG_Made")
  df = split(df, home, away, "2FG_Missed", "H_2FG_Missed", "A_2FG_Missed")
  df = split(df, home, away, "FG3A", "H_3FG_Attempts", "A_3FG_Attempts")
  df = split(df, home, away, "FG3M", "H_3FG_Made", "A_3FG_Made")
  df = split(df, home, away, "FG3m", "H_3FG_Missed", "A_3FG_Missed")
  df = split(df, home, away, "FG3_PCT", "H_FG3_PCT", "A_FG3_PCT")
  df = split(df, home, away, "FT_PCT", "H_FT_PCT", "A_FT_PCT")
  df = split(df, home, away, "BLKA", "H_BLKA", "A_BLKA")
  df = split(df, home, away, "PLUS_MINUS", "H_PLUS_MINUS", "A_PLUS_MINUS")
  df = split(df, home, away, "FG_PCT", "H_FG_PCT", "A_FG_PCT")
  df = split(df, home, away, "FTM", "H_FTM", "A_FTM")
  df = split(df, home, away, "PF", "H_PF", "A_PF")
  df = split(df, home, away, "PFD", "H_PFD", "A_PFD")

  df.rename(columns={'GAME_DATE': 'Date',  'GAME_ID' : 'Game_ID', 'SEASON_YEAR': 'Season'}, inplace=True)
  df.drop(['TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME', 'MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB',
           'AST', 'TOV', 'STL', 'BLK', 'PF', 'PFD', 'GP_RANK', 'MIN_RANK', 'FG3_PCT', 'FT_PCT', 'BLKA', 'PFD', 'PLUS_MINUS', 'W_RANK', 'L_RANK',
           'W_PCT_RANK', 'FG_PCT', 'FGM_RANK', 'FGA_RANK', 'FG_PCT_RANK', 'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK', 'FTM_RANK', 'FTA_RANK',
           'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK', 'REB_RANK', 'AST_RANK', 'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK', 'PF_RANK',
           'PFD_RANK', 'PTS_RANK', 'PLUS_MINUS_RANK', 'PTS', 'WL', 'Opposing_Team', 'Record', 'Opp_Record', 'FGm', 'FG3m', '2FG_Attempts',
           '2FG_Made', '2FG_Missed', 'MATCHUP', 'Home_Record', 'Away_Record'], axis=1, inplace=True)

  even_df = df.iloc[::2]
  odd_df = df.iloc[1::2]
  df = pd.merge(odd_df, even_df, on=['Game_ID', 'Date', 'Season', 'H_Team', 'A_Team', 'H_Team_Record', 'A_Team_Record', 
                                     'H_Points', 'A_Points', 'Winner','H_Total_Rebounds', 'A_Total_Rebounds', 
                                     'H_Offensive_Rebounds', 'A_Offensive_Rebounds', 'H_Defensive_Rebounds', 
                                     'A_Defensive_Rebounds', 'H_Assists', 'A_Assists', 'H_Turnovers', 'A_Turnovers', 
                                     'H_Steals', 'A_Steals', 'H_Blocks', 'A_Blocks', 'H_All_Free_Throws', 'A_All_Free_Throws', 'H_FG_Attempts', 'A_FG_Attempts', 'H_FG_Made', 'A_FG_Made',
                                     'H_FG_Missed', 'A_FG_Missed', 'H_2FG_Attempts', 'A_2FG_Attempts', 'H_2FG_Made', 'A_2FG_Made', 'H_2FG_Missed',
                                     'A_2FG_Missed', 'H_3FG_Attempts', 'A_3FG_Attempts', 'H_3FG_Made', 'A_3FG_Made', 'H_3FG_Missed', 'A_3FG_Missed', 'H_FG3_PCT', 'A_FG3_PCT',
                                     'H_FT_PCT', 'A_FT_PCT', 'H_BLKA', 'A_BLKA', 'H_PLUS_MINUS', 'A_PLUS_MINUS', 'H_FG_PCT', 'A_FG_PCT',
                                     "H_FTM", "A_FTM", "H_PF", "A_PF", "H_PFD", "A_PFD"], suffixes=('_o', '_e'), how='inner', validate='one_to_one')

  df.drop(['index_o', 'index_e', 'Winner'], axis=1, inplace=True)
  df['Date'] = pd.to_datetime(df['Date'])
  return df
# HHTR AND AATR
# Double check software

team_stats = drop(record_func(next_func(T3aM)))

#Home and road team win probabilities implied by Elo ratings and home court adjustment
def win_probs(home_elo, away_elo, home_court_advantage) :
  h = math.pow(10, home_elo/400)
  r = math.pow(10, away_elo/400)
  a = math.pow(10, home_court_advantage/400)

  denom = r + a*h
  home_prob = a*h / denom
  away_prob = r / denom

  return home_prob, away_prob

  #odds the home team will win based on elo ratings and home court advantage

def home_odds_on(home_elo, away_elo, home_court_advantage) :
  h = math.pow(10, home_elo/400)
  r = math.pow(10, away_elo/400)
  a = math.pow(10, home_court_advantage/400)
  return a*h/r

#this function determines the constant used in the elo rating, based on margin of victory and difference in elo ratings
def elo_k(MOV, elo_diff):
  k = 20
  if MOV>0:
      multiplier=(MOV+3)**(0.8)/(7.5+0.006*(elo_diff))
  else:
      multiplier=(-MOV+3)**(0.8)/(7.5+0.006*(-elo_diff))
  return k*multiplier


#updates the home and away teams elo ratings after a game

def update_elo(home_score, away_score, home_elo, away_elo, home_court_advantage) :
  home_prob, away_prob = win_probs(home_elo, away_elo, home_court_advantage)

  if (home_score - away_score > 0) :
    home_win = 1
    away_win = 0
  else :
    home_win = 0
    away_win = 1

  k = elo_k(home_score - away_score, home_elo - away_elo)

  updated_home_elo = home_elo + k * (home_win - home_prob)
  updated_away_elo = away_elo + k * (away_win - away_prob)

  return updated_home_elo, updated_away_elo


#takes into account prev season elo
def get_prev_elo(team, game_date, season, team_stats, elo_df) :
  prev_game = team_stats[team_stats['Date'] < game_date][(team_stats['H_Team'] == team) | (team_stats['A_Team'] == team)].sort_values(by = 'Date').tail(1).iloc[0]

  if team == prev_game['H_Team'] :
    elo_rating = elo_df[elo_df['Game_ID'] == prev_game['Game_ID']]['H_Team_Elo_After'].values[0]
  else :
    elo_rating = elo_df[elo_df['Game_ID'] == prev_game['Game_ID']]['A_Team_Elo_After'].values[0]

  if prev_game['Season'] != season :
    return (0.75 * elo_rating) + (0.25 * 1505)
  else :
    return elo_rating

def elo_df_I_teams_elo_df(team_stats):
    team_stats.sort_values(by = 'Date', inplace = True)
    team_stats.reset_index(inplace=True, drop = True)
    elo_df = pd.DataFrame(columns=['Game_ID', 'H_Team', 'A_Team', 'H_Team_Elo_Before', 'A_Team_Elo_Before', 'H_Team_Elo_After', 'A_Team_Elo_After'])
    teams_elo_df = pd.DataFrame(columns=['Game_ID','Team', 'Elo', 'Date', 'Where_Played', 'Season'])

    for index, row in team_stats.iterrows():
      game_id = row['Game_ID']
      game_date = row['Date']
      season = row['Season']
      h_team, a_team = row['H_Team'], row['A_Team']
      h_score, a_score = row['H_Points'], row['A_Points']

      if (h_team not in elo_df['H_Team'].values and h_team not in elo_df['A_Team'].values) :
        h_team_elo_before = 1500
      else :
        h_team_elo_before = get_prev_elo(h_team, game_date, season, team_stats, elo_df)

      if (a_team not in elo_df['H_Team'].values and a_team not in elo_df['A_Team'].values) :
        a_team_elo_before = 1500
      else :
        a_team_elo_before = get_prev_elo(a_team, game_date, season, team_stats, elo_df)

      h_team_elo_after, a_team_elo_after = update_elo(h_score, a_score, h_team_elo_before, a_team_elo_before, 69)

      new_row = {'Game_ID': game_id, 'H_Team': h_team, 'A_Team': a_team, 'H_Team_Elo_Before': h_team_elo_before, 'A_Team_Elo_Before': a_team_elo_before, \
                                                                            'H_Team_Elo_After' : h_team_elo_after, 'A_Team_Elo_After': a_team_elo_after}
      teams_row_one = {'Game_ID': game_id,'Team': h_team, 'Elo': h_team_elo_before, 'Date': game_date, 'Where_Played': 'Home', 'Season': season}
      teams_row_two = {'Game_ID': game_id,'Team': a_team, 'Elo': a_team_elo_before, 'Date': game_date, 'Where_Played': 'Away', 'Season': season}

      elo_df = elo_df.append(new_row, ignore_index = True)
      teams_elo_df = teams_elo_df.append(teams_row_one, ignore_index=True)
      teams_elo_df = teams_elo_df.append(teams_row_two, ignore_index=True)
    return elo_df, teams_elo_df, team_stats

warnings.filterwarnings("ignore")
elo_df, teams_elo_df, team_stats = elo_df_I_teams_elo_df(team_stats)

def T_teams_elo_df(teams_elo_df, team_stats):
    dates = list(set([d.strftime("%m-%d-%Y") for d in teams_elo_df["Date"]]))
    dates = sorted(dates, key=lambda x: time.strptime(x, '%m-%d-%Y'))
    teams = team_stats["A_Team"]
    dataset = pd.DataFrame(columns=dates)
    dataset["Team"] = teams.drop_duplicates()
    dataset = dataset.set_index("Team")
    for index, row in teams_elo_df.iterrows():
      date = row["Date"].strftime("%m-%d-%Y")
      team = row["Team"]
      elo = row["Elo"]
      dataset[date][team] = elo

    teams_elo_df['Elo'] = teams_elo_df['Elo'].astype(float)
    return teams_elo_df, team_stats

teams_elo_df, team_stats = T_teams_elo_df(teams_elo_df, team_stats)

#given a team and a date, this method will return that teams average stats over the previous n games
def get_avg_stats_last_n_games(team, game_date, season_team_stats, n) :
  prev_game_df = season_team_stats[season_team_stats['Date'] < game_date][(season_team_stats['H_Team'] == team) | (season_team_stats['A_Team'] == team)].sort_values(by = 'Date').tail(n)

  h_df = prev_game_df.iloc[:, range(3, 43, 2)]
  h_df.columns = [x[2:] for x in h_df.columns]
  a_df = prev_game_df.iloc[:, range(4, 44, 2)]
  a_df.columns = [x[2:] for x in a_df.columns]

  df = pd.concat([h_df, a_df])
  df = df[df['Team'] == team]
  df.drop(columns = ['Team'], inplace=True)

  return df.mean()

def Recent_performance_df(team_stats):
    recent_performance_df = pd.DataFrame()

    for season in team_stats['Season'].unique() :
      l = ['Date', 'Game_ID', 'Season', 'H_Team', 'A_Team']
      other = list(team_stats.columns[7:60])
      cols = l + other

      season_team_stats = team_stats[team_stats['Season'] == season].sort_values(by = 'Date')[cols].reset_index(drop = True)

      season_recent_performance_df = pd.DataFrame()

      for index, row in season_team_stats.iterrows() :
        game_id = row['Game_ID']
        game_date = row['Date']
        h_team = row['H_Team']
        a_team = row['A_Team']

        h_team_recent_performance = get_avg_stats_last_n_games(h_team, game_date, season_team_stats, 10)
        h_team_recent_performance.index = ['H_Last_10_Avg_' + x for x in h_team_recent_performance.index]

        a_team_recent_performance = get_avg_stats_last_n_games(a_team, game_date, season_team_stats, 10)
        a_team_recent_performance.index = ['A_Last_10_Avg_' + x for x in a_team_recent_performance.index]

        new_row = pd.concat([h_team_recent_performance, a_team_recent_performance], sort=False)
        new_row['Game_ID'] = game_id

        season_recent_performance_df = season_recent_performance_df.append(new_row, ignore_index=True)
        season_recent_performance_df = season_recent_performance_df[new_row.index]


      recent_performance_df = pd.concat([recent_performance_df, season_recent_performance_df])
    return recent_performance_df, team_stats
warnings.filterwarnings("ignore")
recent_performance_df, team_stats = Recent_performance_df(team_stats)
recent_performance_df.dropna()

def Final_team_stats(team_stats, elo_df, recent_performance_df):
    final_team_stats = team_stats.iloc[0:, [0,1,2,3,4,7,8]].merge(elo_df.drop(columns=['H_Team', 'A_Team']), on = 'Game_ID') \
                                         .merge(recent_performance_df, on = 'Game_ID')

    final_team_stats = final_team_stats.dropna()
    return final_team_stats

final_team_stats = Final_team_stats(team_stats, elo_df, recent_performance_df)

def Team_performances_df(final_team_stats, recent_performance_df, teams_elo_df):
    home_cols = final_team_stats.columns[final_team_stats.columns.str.startswith('H_')]

    team_df = final_team_stats.iloc[0:, [0,1,2,3,4,5,6]].drop(columns=['H_Team', 'A_Team'])
    team_df_home = team_df.drop(columns=team_df.columns[team_df.columns.str.startswith('A_')])
    team_df_away = team_df.drop(columns=team_df.columns[team_df.columns.str.startswith('H_')])

    recent_performance_home = recent_performance_df.drop(columns=recent_performance_df.columns \
                                                         [recent_performance_df.columns.str.startswith('A_')])
    recent_performance_away = recent_performance_df.drop(columns=recent_performance_df.columns \
                                                         [recent_performance_df.columns.str.startswith('H_')])

    team_by_team_home = team_df_home.merge(teams_elo_df[teams_elo_df.Where_Played == "Home"], on = 'Game_ID') \
                                         .merge(recent_performance_home, on = 'Game_ID')
    team_by_team_away = team_df_away.merge(teams_elo_df[teams_elo_df.Where_Played == "Away"], on = 'Game_ID') \
                                         .merge(recent_performance_away, on = 'Game_ID')

    team_by_team_home.columns = team_by_team_home.columns.str.replace("H_", "")
    team_by_team_away.columns = team_by_team_away.columns.str.replace("A_", "")
    team_by_team_home.columns = team_by_team_home.columns.str.replace("_y", "")
    team_by_team_home.columns = team_by_team_home.columns.str.replace("_x", "")
    team_by_team_home = team_by_team_home.loc[:,~team_by_team_home.columns.duplicated()]

    team_by_team_away.columns = team_by_team_away.columns.str.replace("_y", "")
    team_by_team_away.columns = team_by_team_away.columns.str.replace("_x", "")
    team_by_team_away = team_by_team_away.loc[:,~team_by_team_away.columns.duplicated()]

    team_performances_df = pd.concat([team_by_team_home, team_by_team_away]).sort_index(axis=0).reset_index().drop(columns=['index'])
    return team_performances_df, final_team_stats 

team_performances_df, final_team_stats = Team_performances_df(final_team_stats, recent_performance_df, teams_elo_df)

def Final_Player_Stats1(final_team_stats):
    final_team_stats['Label'] = [1 if x > 0 else 0 for x in final_team_stats['H_Points'] - final_team_stats['A_Points']]
    final_team_stats.drop(columns=['H_Points', 'A_Points'], inplace=True)
    final_team_stats = final_team_stats.dropna()

    combined_copy = final_team_stats
    return final_team_stats, combined_copy

final_team_stats, combined_copy = Final_Player_Stats1(final_team_stats)
#final_team_stats.to_csv('Final_Team_Stats.csv')

def Player(player_stats):
    player_stats.rename(columns={'GAME_DATE': 'Date', 'MIN' : 'Min', 'SEASON_YEAR' : 'Season', 'PLAYER_NAME' : 'PlayerName', 'GAME_ID' : 'GameID', 'STL' : 'Stl', 'FG3M' : '3FGM',
                                 'BLK' : 'Blk', 'OREB' : 'OffReb', 'AST' : 'Ast', 'DREB' : 'DefReb', 'TOV' : 'T/O', 'PTS':'Pts'}, inplace=True)
    player_stats = player_stats.sort_values(by = 'Date').reset_index(drop = True)
    player_stats['PPP'] = player_stats['PPP'].apply(lambda x: '0' if x == '-' else x)
    player_stats['PPP'] = player_stats['PPP'].astype(float)
    return player_stats
player_stats = Player(player_stats)

def rolling_average_last_n_games(n) :
  player_stats_recent_performance_df = pd.DataFrame()

  for season in player_stats['Season'].unique() :
    season_player_stats = player_stats[player_stats['Season'] == season]

    for player in season_player_stats['PlayerName'].unique() :
      player_recent_performance = season_player_stats[season_player_stats['PlayerName'] == player].rolling(n, min_periods=1).mean().shift(1)

      player_stats_recent_performance_df = player_stats_recent_performance_df.append(player_recent_performance, ignore_index=False)

  player_stats_recent_performance_df.columns = ['Last_' + str(n) + '_Avg_' + x for x in player_stats_recent_performance_df.columns]
  player_stats_recent_performance_df = player_stats[['PlayerName', 'GameID']].merge(player_stats_recent_performance_df.drop(columns='Last_' + str(n) + '_Avg_' + 'GameID'), left_index=True, right_index=True)

  return player_stats_recent_performance_df
    
warnings.filterwarnings("ignore")
player_recent_performance_df = rolling_average_last_n_games(10)

def Avg_season_stats(player_stats, player_recent_performance_df):
    avg_season_stats = pd.DataFrame()

    for season in player_stats['Season'].unique() :
      season_player_stats = player_stats[player_stats['Season'] == season]
      for player in season_player_stats['PlayerName'].unique() :
        player_avg_season_stats = season_player_stats[season_player_stats['PlayerName'] == player].expanding().mean().shift(1)

        avg_season_stats = avg_season_stats.append(player_avg_season_stats, ignore_index=False)


    avg_season_stats.drop(columns= 'GameID', inplace = True)
    avg_season_stats.columns = ['Avg_Season_' + x for x in avg_season_stats.columns]
    avg_season_stats = player_stats.iloc[:, 0:6].merge(avg_season_stats, left_index = True, right_index = True)
    final_player_stats = avg_season_stats.merge(player_recent_performance_df.drop(columns = ['GameID', 'PlayerName']), left_index=True, right_index=True)
    final_player_stats = final_player_stats.dropna()
    return avg_season_stats, final_player_stats

warnings.filterwarnings("ignore")
avg_season_stats, final_player_stats = Avg_season_stats(player_stats, player_recent_performance_df)
#final_player_stats.to_csv('Final_Player_Stats.csv')

def Team_performances_DF(team_performances_df, combined_copy):
    labels = []

    for i, row in combined_copy.iterrows():
      game_id = row['Game_ID']
      win = row['Label']
      tp_row = team_performances_df.loc[team_performances_df['Game_ID'] == game_id]
      for j, r in tp_row.iterrows():
        #print(r['Where_Played'])
        if r['Where_Played'] == 'Home' and win == 0:
          labels.append(0)
        elif r['Where_Played'] == 'Away' and win == 1:
          labels.append(0)
        else:
          labels.append(1)

    team_performances_df['Win'] = labels

    team_performances_df['Date'] = pd.to_datetime(team_performances_df['Date'])
    team_performances_df.sort_values(by=['Date'], inplace=True, ascending=True)
    return team_performances_df
team_performances_df = Team_performances_DF(team_performances_df, combined_copy)

def PLayer_stats(player_stats):
    player_stats['PER_stub'] = player_stats['FGM'] * 85.910 + player_stats['Stl'] * 53.897 + player_stats['3FGM'] * 51.757 + player_stats['FTM'] * 46.845 \
                               + player_stats['Blk'] * 39.190 + player_stats['OffReb'] * 39.190+ player_stats['Ast'] * 34.677+ player_stats['DefReb'] * 14.707 \
                               - player_stats['PFCom'] * 17.17 - player_stats['FTm'] * 20.091 - player_stats['FGm'] * 39.190 - player_stats['T/O'] * 53.897


    # raw PER intermediate value
    player_stats['PER_stub'] = player_stats['FGM']*85.910+player_stats['Stl']*53.897+player_stats['3FGM']*51.757+player_stats['FTM']*46.845+player_stats['Blk']\
                              *39.190+player_stats['OffReb']*39.19+player_stats['Ast']*34.677+player_stats['DefReb']*14.707-player_stats['PFCom']*17.174-player_stats['FTm']\
                              *20.091-player_stats['FGm']*39.190-player_stats['T/O']*53.897


    # calculate PER per player, per game
    player_stats['PER'] = player_stats['PER_stub'] * (1 / player_stats['Min'])
    return player_stats
player_stats = PLayer_stats(player_stats)

def FTS(final_team_stats):
    fts = final_team_stats.copy()
    fts['H_Team_Elo_Before'] = fts['H_Team_Elo_Before'].astype(float)
    fts['A_Team_Elo_Before'] = fts['A_Team_Elo_Before'].astype(float)
    fts['SEASON'] = fts['Season'].str[:4]
    fts['Temp'] = fts['Date']
    # 0 = Away, 1 = Home
    fts.reset_index(inplace=True, drop=True)
    for x in range(len(fts)):
        if fts['Label'][x] == 0:
            fts['Label'][x] = fts['A_Team'][x]
        else:
            fts['Label'][x] = fts['H_Team'][x]
    fts.drop(['Season', 'Game_ID', 'H_Team', 'A_Team'],axis=1, inplace=True)
    fts['Date'] = fts['Date'] - pd.to_datetime('10-31-' + fts['SEASON'])
    fts['SEASON'] = fts['Temp'].dt.year
    fts.drop('Temp', axis=1, inplace=True)
    fts['Date'] = fts['Date'].dt.days.astype(int)
    return fts
fts = FTS(final_team_stats)

features = fts.drop(columns = 'Label')
label = fts['Label']
