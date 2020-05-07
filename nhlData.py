import pandas as pd 
from collections import defaultdict
import pickle
import json
import numpy as np
import random
import matplotlib.pyplot as plt
import pprint
import requests
import numbers
import time
from datetime import datetime
from geopy.distance import geodesic
from scipy.stats import percentileofscore as percentile

pp = pprint.PrettyPrinter(indent=4)


#grabs the time zone of each arena by its lat-lon coordinates
with open("arena_tzones.json","r") as f:
	ARENA_ZONES = json.load(f)

with open("./arenas.json",'r') as f:
	ARENAS = json.load(f)

#grabs games and separates out the playoffs, as they negate lots of the travel issues
#by being exclusively between two teams at a time
GAMES = pd.read_csv("./game.csv")
PLAYOFFS = GAMES[GAMES.type=="P"]
GAMES = GAMES[GAMES.type=="R"]


#grabs team info to create a mapping from team_id to team name and abbreviation
def getTeams(filename):
	teams_df = pd.read_csv(filename)
	teams = defaultdict(lambda: {})
	for id_,city,name,abv in zip(teams_df['team_id'],teams_df["shortName"],teams_df['teamName'],teams_df["abbreviation"]):
		teams[id_] = abv
	return teams


TEAMS = getTeams("./team_info.csv")
DST_MAX = 2782.8694688882847 #max distance travelled for normalization
DAY_MAX = 10 #max days of rest to account for
TEAMSTATS = pd.read_csv("./game_teams_stats.csv")



# nhl_seasons = defaultdict(lambda: [])
# for game_id, season, date, home_id, away_id, home_goals, away_goals in zip(GAMES["game_id"],GAMES["season"],GAMES["date_time"],GAMES["home_team_id"],GAMES["away_team_id"],GAMES["home_goals"],GAMES["away_goals"]):
# 	game = {}
# 	game["home"] = int(home_id)
# 	game["away"] = int(away_id)
# 	game["date"] = str(date)
# 	game["id"] = int(game_id)
# 	game["stats"] = defaultdict(lambda: {})
# 	nhl_seasons[str(season)].append(game)


#gets the counting stats for each team's games for every season
def getGameStats(for_df,against_df):
	stats = {}
	stats["goals_for"] = int(for_df["goals"].values[0])
	stats["goals_against"] = int(against_df["goals"].values[0])
	stats["shots_for"] = int(for_df["shots"].values[0])
	stats["shots_against"] = int(against_df["shots"].values[0])

	stats["hits_for"] = int(for_df["hits"].values[0])
	stats["hits_against"] = int(against_df["hits"].values[0])
	stats["giveaways"] = int(for_df["giveaways"].values[0])
	stats["takeaways"] = int(for_df["takeaways"].values[0])

	stats["pim_for"] = int(for_df["pim"].values[0])
	stats["pim_against"] = int(against_df["pim"].values[0])
	stats["power_plays"] = int(for_df["powerPlayOpportunities"].values[0])
	stats["power_play_goals"] = int(for_df["powerPlayGoals"].values[0])
	stats["penalty_kills"] = int(against_df["powerPlayOpportunities"].values[0])
	stats["penalty_kill_goals"] = int(against_df["powerPlayGoals"].values[0])

	if stats["power_plays"] > 0:
		stats["power_play_percentage"] = stats["power_play_goals"] / stats["power_plays"]
	else:
		stats["power_play_percentage"] = 0

	if stats["penalty_kills"] > 0:
		stats["penalty_kill_percentage"] = stats["penalty_kill_goals"] / stats["penalty_kills"]
	else:
		stats["penalty_kill_percentage"] = 100

	stats["faceoff_percentage"] = float(for_df["faceOffWinPercentage"].values[0])
	stats["shooting_percentage"] = float(for_df["goals"].values[0] / for_df["shots"].values[0])
	stats["save_percentage"] = float(1 - (against_df["goals"].values[0] / against_df["shots"].values[0]))
	stats["PDO"] = float(stats["shooting_percentage"] + stats["save_percentage"])

	return stats


#gets the cumulative stats for each teams games for every season 
def getCumulative(season):

	cumulative_stats = defaultdict(lambda: 0)
	for idx, game in enumerate(season):
		game_stats = game["stats"]["game"]

		cumulative_stats["games_played"] += 1
		cumulative_stats["goals_for"] += game_stats["goals_for"]
		cumulative_stats["goals_against"] += game_stats["goals_against"]
		cumulative_stats["shots_for"] += game_stats["shots_for"]
		cumulative_stats["shots_against"] += game_stats["shots_against"]

		cumulative_stats["hits_for"] += game_stats["hits_for"]
		cumulative_stats["hits_against"] += game_stats["hits_against"]
		cumulative_stats["giveaways"] += game_stats["giveaways"]
		cumulative_stats["takeaways"] += game_stats["takeaways"]

		cumulative_stats["pim_for"] += game_stats["pim_for"]
		cumulative_stats["pim_against"] += game_stats["pim_against"]
		cumulative_stats["power_plays"] += game_stats["power_plays"]
		cumulative_stats["power_play_goals"] += game_stats["power_play_goals"]
		cumulative_stats["penalty_kills"] += game_stats["penalty_kills"]
		cumulative_stats["penalty_kill_goals"] += game_stats["penalty_kill_goals"]


		cumulative_stats["power_play_percentage"] = cumulative_stats["power_play_goals"] / max(cumulative_stats["power_plays"],1)
		cumulative_stats["penalty_kill_percentage"] = 1 - (cumulative_stats["penalty_kill_goals"] / max(cumulative_stats["penalty_kills"],1))


		cumulative_stats["shooting_percentage"] = cumulative_stats["goals_for"] / cumulative_stats["shots_for"]
		cumulative_stats["save_percentage"] = 1 - (cumulative_stats["goals_against"] / cumulative_stats["shots_against"])
		cumulative_stats["PDO"] = cumulative_stats["shooting_percentage"] + cumulative_stats["save_percentage"]

		game["stats"]["cumulative"] = {i : cumulative_stats[i] for i in cumulative_stats}

	return season


#compares each team's current game performance to their 
#season distribution for each game stat
def getTeamPercentile(season):
	team_percentile = defaultdict(lambda: [])
	for idx, game in enumerate(season):
		game_stats = game["stats"]["game"]

		team_percentile["goals_for"].append(game_stats["goals_for"])
		team_percentile["goals_against"].append(game_stats["goals_against"])
		team_percentile["shots_for"].append(game_stats["shots_for"])
		team_percentile["shots_against"].append(game_stats["shots_against"])
		team_percentile["hits_for"].append(game_stats["hits_for"])
		team_percentile["hits_against"].append(game_stats["hits_against"])
		team_percentile["giveaways"].append(game_stats["giveaways"])
		team_percentile["takeaways"].append(game_stats["takeaways"])
		team_percentile["pim_for"].append(game_stats["pim_for"])
		team_percentile["pim_against"].append(game_stats["pim_against"])
		team_percentile["power_plays"].append(game_stats["power_plays"])
		team_percentile["power_play_goals"].append(game_stats["power_play_goals"])
		team_percentile["penalty_kills"].append(game_stats["penalty_kills"])
		team_percentile["penalty_kill_goals"].append(game_stats["penalty_kill_goals"])
		team_percentile["power_play_percentage"].append(game_stats["power_play_percentage"])
		team_percentile["penalty_kill_percentage"].append(game_stats["penalty_kill_percentage"])
		team_percentile["shooting_percentage"].append(game_stats["shooting_percentage"])
		team_percentile["save_percentage"].append(game_stats["save_percentage"])
		team_percentile["PDO"].append(game_stats["PDO"])

		game["stats"]["team_percentile"] = {i : float(percentile(team_percentile[i],game_stats[i],kind='mean')/100) for i in dict(team_percentile)}
	return season

#quick conversion from "home win OT" and "away win REG"
#to a categorical var
def settledToInt(settled):
	if "OT" in settled:
		return 1
	elif "SO" in settled:
		return 2
	else:
		return 3


#compares each team's cumulative performance to that of all other teams at
#the same point in the season
def getLeagueDistribution(team_seasons, season, idx):

	league_percentile = defaultdict(lambda: [])
	for team_id in team_seasons:
		try:
			game_stats = team_seasons[team_id][season][idx]["stats"]["cumulative"]

			league_percentile["goals_for"].append(game_stats["goals_for"])
			league_percentile["goals_against"].append(game_stats["goals_against"])
			league_percentile["shots_for"].append(game_stats["shots_for"])
			league_percentile["shots_against"].append(game_stats["shots_against"])
			league_percentile["hits_for"].append(game_stats["hits_for"])
			league_percentile["hits_against"].append(game_stats["hits_against"])
			league_percentile["giveaways"].append(game_stats["giveaways"])
			league_percentile["takeaways"].append(game_stats["takeaways"])
			league_percentile["pim_for"].append(game_stats["pim_for"])
			league_percentile["pim_against"].append(game_stats["pim_against"])
			league_percentile["power_plays"].append(game_stats["power_plays"])
			league_percentile["power_play_goals"].append(game_stats["power_play_goals"])
			league_percentile["penalty_kills"].append(game_stats["penalty_kills"])
			league_percentile["penalty_kill_goals"].append(game_stats["penalty_kill_goals"])
			league_percentile["power_play_percentage"].append(game_stats["power_play_percentage"])
			league_percentile["penalty_kill_percentage"].append(game_stats["penalty_kill_percentage"])
			league_percentile["shooting_percentage"].append(game_stats["shooting_percentage"])
			league_percentile["save_percentage"].append(game_stats["save_percentage"])
			league_percentile["PDO"].append(game_stats["PDO"])

		except:
			continue

	return league_percentile



#gets an instance-based set of data points connecting each team involved in 
#a given game with their cumulative statistics as well as who won the game
def getCumulativeInstances(team_seasons):

	instances = defaultdict(lambda: defaultdict(lambda: {}))
	for team_id in team_seasons:
		for season in team_seasons[team_id]:
			for game in team_seasons[team_id][season]:
				if str(game["home"]) == str(team_id):
					instances[game["id"]]["home"] = game["stats"]["league_percentile"]
					if game["won"]: 
						instances[game["id"]]["winner"] = [1,0]
					else:
						instances[game["id"]]["winner"] = [0,1]
				elif str(game["away"]) == str(team_id):
					instances[game["id"]]["away"] = game["stats"]["league_percentile"]
					if game["won"]: 
						instances[game["id"]]["winner"] = [0,1]
					else:
						instances[game["id"]]["winner"] = [1,0]
				else:
					print("something went wrong: ", game["home"], game["away"])



	return instances







#wraps all of the data getting functions into one that puts the teams into one dictionary
def getTeamGameStats(nhl_seasons):
	nhl_seasons = {i:sorted(nhl_seasons[i],key=lambda x: x.get("date")) for i in nhl_seasons}
	team_seasons = defaultdict(lambda: defaultdict(lambda: []))
	for team_id in TEAMS:
		print(TEAMS[team_id])
		for season in nhl_seasons:
			team_season_games = sorted([game for game in nhl_seasons[season] if game["home"] == team_id or game["away"] == team_id], key=lambda x: x.get("date"))
			for idx, game in enumerate(team_season_games):


				game_df = TEAMSTATS[TEAMSTATS.game_id == game["id"]]
				for_df = game_df[game_df.team_id == team_id]
				against_df = game_df[game_df.team_id != team_id]

				game["won"] = int(for_df["won"].values[0])
				game["stats"]["travel"] = defaultdict(lambda: 0)
				game["stats"]["travel"]["home"] = int("home" == for_df["HoA"].values[0])
				game["stats"]["travel"]["game_day"] = datetime.strptime(game["date"], "%Y-%m-%d").weekday()
				game["stats"]["travel"]["game_reg"] = int(str(for_df["settled_in"]) == "REG") 
				game["stats"]["travel"]["game_ot"] = int(str(for_df["settled_in"]) == "OT")
				game["stats"]["travel"]["game_so"] = int(str(for_df["settled_in"]) == "SO")

				if idx == 0:
					game["stats"]["travel"]["rest_days"] = 1.0

					if game["stats"]["travel"]["home"]:
						game["stats"]["travel"]["timezone"] = 0
						game["stats"]["travel"]["distance"] = 0

					else:
						prev_tz = ARENA_ZONES[TEAMS[team_id]]
						curr_tz = ARENA_ZONES[TEAMS[game["home"]]]
						game["stats"]["travel"]["timezone"] = abs(curr_tz - prev_tz)

						prev_loc = ARENAS[TEAMS[team_id]]
						curr_loc = ARENAS[TEAMS[game["home"]]]
						game["stats"]["travel"]["distance"] = (geodesic(prev_loc, curr_loc).miles / DST_MAX)

				else:
					prev_game = team_season_games[idx-1]

					prev_date = prev_game["date"]
					curr_date = game["date"]
					d1 = datetime.strptime(prev_date, "%Y-%m-%d")
					d2 = datetime.strptime(curr_date, "%Y-%m-%d")				
					game["stats"]["travel"]["rest_days"] = min(abs((d1 - d2).days),10) / DAY_MAX

					prev_tz = ARENA_ZONES[TEAMS[prev_game["home"]]]
					curr_tz = ARENA_ZONES[TEAMS[game["home"]]]
					game["stats"]["travel"]["timezone"] = abs(curr_tz - prev_tz)

					prev_loc = ARENAS[TEAMS[prev_game["home"]]]
					curr_loc = ARENAS[TEAMS[game["home"]]]
					game["stats"]["travel"]["distance"] = geodesic(curr_loc,prev_loc).miles / DST_MAX

				game["stats"]["game"] = getGameStats(for_df,against_df)


			team_season_games = getCumulative(team_season_games)
			team_season_games = getTeamPercentile(team_season_games)
			team_seasons[team_id][season] = team_season_games



	for team_id in team_seasons:
		for season in team_seasons[team_id]:
			for idx, game in enumerate(team_seasons[team_id][season]):
				league_percentile = getLeagueDistribution(team_seasons,season,idx)
				game["stats"]["league_percentile"] = {i: float(percentile(league_percentile[i],game["stats"]["cumulative"][i])/100) for i in dict(league_percentile)}

	return team_seasons



#grabs the specified sets of features for the provided
#sublist of games
def getFeatures(sublist,feature_sets=["league_percentile","team_percentile"]):

	X = None
	flag = False
	feature_names = []
	for idx,game in enumerate(sublist):
		x = []
		features = []
		for sets in feature_sets:
			for key in game["stats"][sets]:
				if not flag:
					feature_name = sets + "_" + key
					features.append(feature_name)
				x.append(game["stats"][sets][key])


		if not flag:
			X = np.array(x)
			feature_names = [k for k in features]
			flag = True
		else:
			X = np.vstack((X,x))

	return X, feature_names



#generates sequences data from the list of all team seasons
#gets length N sequences
#gets the requested feature sets
def generateSequences(team_seasons, feature_sets=["league_percentile"], N=5):

	X_data = None
	y_data = None
	flag = False
	feature_names = None
	game_ids = []

	for tdx, team_id in enumerate(team_seasons):
		print("team: %s | %2d / %2d" % (TEAMS[team_id],tdx+1,len(team_seasons)))
		for sdx, season in enumerate(team_seasons[team_id]):
			print("\tseason: %s | %2d / %2d" % (season, sdx + 1, len(team_seasons[team_id])))
			for i in range(0,len(team_seasons[team_id][season])-N-1):

				sublist = team_seasons[team_id][season][i:i+N] # N previous games (not including season[i+1])
				time_series, feature_names = getFeatures(sublist,feature_sets=feature_sets)
				outcome = team_seasons[team_id][season][i+N+1]["won"]
				game_ids.append(team_seasons[team_id][season][i+N+1]) #game being predicted


				if not flag:
					X_data = np.array(time_series)

					y_data = [outcome]

					flag = True
				else:
					X_data = np.dstack((X_data,time_series))
					y_data.append(outcome)


	return X_data, np.array(y_data), feature_names, game_ids




if __name__ == "__main__":
	pass







