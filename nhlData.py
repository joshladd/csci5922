import pandas as pd 
from collections import defaultdict
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
from datetime import datetime
import pprint
from geopy.distance import geodesic
import requests
import numbers
import time
from scipy.stats import percentileofscore as percentile

pp = pprint.PrettyPrinter(indent=4)



with open("arena_tzones.json","r") as f:
	ARENA_ZONES = json.load(f)

with open("./arenas.json",'r') as f:
	ARENAS = json.load(f)

GAMES = pd.read_csv("./game.csv")
PLAYOFFS = GAMES[GAMES.type=="P"]
GAMES = GAMES[GAMES.type=="R"]

def getTeams(filename):

	teams_df = pd.read_csv(filename)
	teams = defaultdict(lambda: {})
	for id_,city,name,abv in zip(teams_df['team_id'],teams_df["shortName"],teams_df['teamName'],teams_df["abbreviation"]):

		teams[id_] = abv

	return teams


TEAMS = getTeams("./team_info.csv")


DAYS = []
OFFSETS = []
DISTANCES = []

DST_MAX = 2782.8694688882847
DAY_MAX = 10


TEAMSTATS = pd.read_csv("./game_teams_stats.csv")


nhl_seasons = defaultdict(lambda: [])

for game_id, season, date, home_id, away_id, home_goals, away_goals in zip(GAMES["game_id"],GAMES["season"],GAMES["date_time"],GAMES["home_team_id"],GAMES["away_team_id"],GAMES["home_goals"],GAMES["away_goals"]):
	game = {}
	game["home"] = int(home_id)
	game["away"] = int(away_id)
	game["date"] = str(date)
	game["id"] = int(game_id)
	game["stats"] = defaultdict(lambda: {})
	nhl_seasons[str(season)].append(game)


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

def settledToInt(settled):
	if "OT" in settled:
		return 1
	elif "SO" in settled:
		return 2
	else:
		return 3


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


# def getCumulativeLastN(season, idx, N):

def getCumulativeInstances(team_seasons):

	instances = defaultdict(lambda: defaultdict(lambda: {}))
	for team_id in team_seasons:
		# print(team_id)
		for season in team_seasons[team_id]:
			for game in team_seasons[team_id][season]:
				if str(game["home"]) == str(team_id):
					# print("y1")
					instances[game["id"]]["home"] = game["stats"]["league_percentile"]
					if game["won"]: 
						instances[game["id"]]["winner"] = [1,0]
					else:
						instances[game["id"]]["winner"] = [0,1]
				elif str(game["away"]) == str(team_id):
					# print("y2")

					instances[game["id"]]["away"] = game["stats"]["league_percentile"]
					if game["won"]: 
						instances[game["id"]]["winner"] = [0,1]
					else:
						instances[game["id"]]["winner"] = [1,0]
				else:
					print("something went wrong: ", game["home"], game["away"])



	return instances





nhl_seasons = {i:sorted(nhl_seasons[i],key=lambda x: x.get("date")) for i in nhl_seasons}
team_seasons = defaultdict(lambda: defaultdict(lambda: []))



# for team_id in TEAMS:

# 	print(TEAMS[team_id])
# 	for season in nhl_seasons:
# 		team_season_games = sorted([game for game in nhl_seasons[season] if game["home"] == team_id or game["away"] == team_id], key=lambda x: x.get("date"))
# 		for idx, game in enumerate(team_season_games):

# 			# if game["home"] == team_id:


# 			game_df = TEAMSTATS[TEAMSTATS.game_id == game["id"]]
# 			for_df = game_df[game_df.team_id == team_id]
# 			against_df = game_df[game_df.team_id != team_id]

# 			# print(for_df["won"])
# 			# print(against_df["won"])
# 			# print(for_df["won"].values[0] == against_df["won"].values[0])
# 			# print(game)
# 			# quit()

# 			game["won"] = int(for_df["won"].values[0])
# 			game["stats"]["travel"] = defaultdict(lambda: 0)
# 			game["stats"]["travel"]["home"] = int("home" == for_df["HoA"].values[0])
# 			game["stats"]["travel"]["game_day"] = datetime.strptime(game["date"], "%Y-%m-%d").weekday()
# 			game["stats"]["travel"]["game_reg"] = int(str(for_df["settled_in"]) == "REG") 
# 			game["stats"]["travel"]["game_ot"] = int(str(for_df["settled_in"]) == "OT")
# 			game["stats"]["travel"]["game_so"] = int(str(for_df["settled_in"]) == "SO")

# 			if idx == 0:
# 				game["stats"]["travel"]["rest_days"] = 1.0

# 				if game["stats"]["travel"]["home"]:
# 					game["stats"]["travel"]["timezone"] = 0
# 					game["stats"]["travel"]["distance"] = 0

# 				else:
# 					prev_tz = ARENA_ZONES[TEAMS[team_id]]
# 					curr_tz = ARENA_ZONES[TEAMS[game["home"]]]
# 					game["stats"]["travel"]["timezone"] = abs(curr_tz - prev_tz)

# 					prev_loc = ARENAS[TEAMS[team_id]]
# 					curr_loc = ARENAS[TEAMS[game["home"]]]
# 					game["stats"]["travel"]["distance"] = (geodesic(prev_loc, curr_loc).miles / DST_MAX)

# 			else:
# 				prev_game = team_season_games[idx-1]

# 				prev_date = prev_game["date"]
# 				curr_date = game["date"]
# 				d1 = datetime.strptime(prev_date, "%Y-%m-%d")
# 				d2 = datetime.strptime(curr_date, "%Y-%m-%d")				
# 				game["stats"]["travel"]["rest_days"] = min(abs((d1 - d2).days),10) / DAY_MAX

# 				prev_tz = ARENA_ZONES[TEAMS[prev_game["home"]]]
# 				curr_tz = ARENA_ZONES[TEAMS[game["home"]]]
# 				game["stats"]["travel"]["timezone"] = abs(curr_tz - prev_tz)

# 				prev_loc = ARENAS[TEAMS[prev_game["home"]]]
# 				curr_loc = ARENAS[TEAMS[game["home"]]]
# 				game["stats"]["travel"]["distance"] = geodesic(curr_loc,prev_loc).miles / DST_MAX

# 			game["stats"]["game"] = getGameStats(for_df,against_df)


# 		team_season_games = getCumulative(team_season_games)
# 		team_season_games = getTeamPercentile(team_season_games)
# 		team_seasons[team_id][season] = team_season_games



# for team_id in team_seasons:
# 	for season in team_seasons[team_id]:
# 		for idx, game in enumerate(team_seasons[team_id][season]):
# 			league_percentile = getLeagueDistribution(team_seasons,season,idx)
# 			game["stats"]["league_percentile"] = {i: float(percentile(league_percentile[i],game["stats"]["cumulative"][i])/100) for i in dict(league_percentile)}


def getNumpyData(team_seasons, feature_sets=["league_percentile"]):

	X = None
	y = None
	flag = False
	feature_names = None
	for team_id in team_seasons:
		for season in team_seasons[team_id]:
			for game in team_seasons[team_id][season]:
				x = []
				features = []
				for sets in feature_sets:
					for key in game["stats"][sets]:
						feature_name = sets + "_" + key
						features.append(feature_name)

						x.append(game["stats"][sets][key])

				if not flag:
					X = np.array(x)
					y = [game["won"]]
					feature_names = [k for k in features]
					flag = True
				else:
					X = np.vstack((X,x))
					y.append(game["won"])



	print(np.sum(y),len(list(y)))
	return X, np.array(y), feature_names


def getN_1Data(team_seasons, feature_sets=["league_percentile"]):

	X = None
	y = None
	flag = False
	feature_names = None
	game_ids = []
	for team_id in team_seasons:
		for season in team_seasons[team_id]:
			for idx, game in enumerate(team_seasons[team_id][season]):
				game_ids.append(game["id"])
				# prev_game = team_seasons[team_id][season][idx]
				x = [game["travel"]["home"]]
				features = []
				for sets in feature_sets:
					for key in game["stats"][sets]:
						feature_name = sets + "_" + key
						features.append(feature_name)

						x.append(game["stats"][sets][key])

				if not flag:
					X = np.array(x)
					y = [game["won"]]
					feature_names = [k for k in features]
					flag = True
				else:
					X = np.vstack((X,x))
					y.append(game["won"])



	return X, np.array(y), ["home"] + feature_names, game_ids


def getFeatures(sublist,feature_sets=["league_percentile","team_percentile"],forget=["goals_for","goals_against"]):

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




print("reading team data")
with open("team_data_1.json","r") as f:
	team_seasons = json.load(f)


# instances = getCumulativeInstances(team_seasons)
# with open("instances.json","w") as f:
# 	json.dump(instances,f,indent=3)

# labels = None
# data = None
# flag = False
# for game_id in instances:
# 	game = instances[game_id]
# 	label = game["winner"]
# 	if isinstance(label,dict):
# 		continue
# 	else:
# 		label = np.array(game["winner"])
# 	l = game["home"]
# 	r = game["away"]
# 	instant = np.zeros(shape=(len(l.keys()),2))
# 	for idx, key in enumerate(l):
# 		instant[idx,0] = l[key]
# 		instant[idx,1] = r[key]

# 	if not flag:
# 		labels = [label]
# 		data = instant
# 		flag = True
# 	else:
# 		labels.append(label)
# 		data = np.dstack((data,instant))

# labels = np.array(labels)
# print(labels.shape)
# print(data.shape)


# with open("instances.pkl","wb") as f:
# 	pickle.dump(np.swapaxes(data,0,2),f)

# with open("instance_labels.pkl","wb") as f:
# 	pickle.dump(labels,f)


# def generateNonSequences(team_seasons, feature_sets = ["travel"], N=10):









print(team_seasons.keys())
# quit()

for sets in [["travel"],["league_percentile"],["cumulative"], ["game"], ["team_percentile"],["travel", "league_percentile"],["travel", "cumulative"], ["travel", "game"], ["travel", "team_percentile"]]:
	if sets == ["travel"]:
		continue
	print("generating sequences")
	X , y , features , games = generateSequences(team_seasons,feature_sets=sets,N=5)

	# for trainx, trainy, game in zip(X,y,games):
	# 	print(game)


	X = np.swapaxes(X,0,2)
	X = np.swapaxes(X,1,2)

	# print(X[0])
	# print(X.shape)
	print(features)
	# # quit()
	title = '-'.join(sets)
	with open("X_"+title+"_N5.pkl","wb") as f:
		pickle.dump(X, f)

	with open("y_"+title+"_N5.pkl","wb") as f:
		pickle.dump(y, f)

	with open("features_"+title+"_N5.pkl", "wb") as f:
		pickle.dump(features, f)

	with open("games_"+title+"_N5.pkl","wb") as f:
		pickle.dump(games, f)


	print("-----------")
	print(len(games))
	print(y.shape)
	print(X.shape)
	print("-----------")


	idx = 0
	collisions = defaultdict(lambda: defaultdict(lambda: []))
	for row, target, game in zip (X,y,games):


		if len(collisions[game["id"]]["data"]) == 0:
			collisions[game["id"]]["data"].append(row)
			collisions[game["id"]]["target"].append(target)
			collisions[game["id"]]["count"] = 1

		else:
			collisions[game["id"]]["data"].append(collisions[game["id"]]["data"][0])
			collisions[game["id"]]["target"].append(target)
			collisions[game["id"]]["count"] += 1



	x = None
	y = None
	flag = False
	games = []
	for idx, game in enumerate(collisions):
		games.append(game)
		print("\r%5d / %5d" % (idx + 1, len(collisions)),end='')
		if collisions[game]["count"] <= 1:
			continue

		point = np.concatenate((collisions[game]["data"][0],collisions[game]["data"][1]), axis = 1)
		# print('1p',point.shape)
		target = np.array(collisions[game]["target"])
		# print('1t',np.array(target).shape)


		if not flag:
			x = point   #np.hstack((point[0],point[1]))
			y = target
			flag = True
		else:

			x = np.dstack((x,point))
			y = np.vstack((y,target))




	print(x.shape)
	print(y.shape)










	with open("X_"+title+"_N5.pkl","wb") as f:
		pickle.dump(x, f)

	with open("y_"+title+"_N5.pkl","wb") as f:
		pickle.dump(y, f)

	with open("features_"+title+"_N5.pkl", "wb") as f:
		pickle.dump(features, f)

	with open("games_"+title+"_N5.pkl","wb") as f:
		pickle.dump(games, f)

	# quit()





print("writing collisions")

with open("collisions.json","w") as f:
	json.dump(collisions,f,indent=3)


quit()

'''


with open("X_N5.pkl","rb") as f:
	X = pickle.load(f)

with open("y_N5.pkl","rb") as f:
	y = pickle.load(f)

with open("features_N5.pkl", "rb") as f:
	FEATURES = pickle.load(f)




	# print("generating numpy matrices")
	# X, y, feature_names, game_ids = getN_1Data(team_seasons,feature_sets=["cumulative"])


	# X_new = defaultdict(lambda: [])
	# y_new = defaultdict(lambda: 0)

	# print("converting to non-duplicates")

	# # print(len(list(set(game_ids))),len(game_ids))

	# # singletons = [i for i in game_ids if game_ids.count(i)==1]
	# # print(singletons)
	# # quit()
	# for game_id, row , score in zip(game_ids, X, y):

	# 	if len(X_new[game_id]) == 0:
	# 		X_new[game_id] = row
	# 		y_new[game_id] = score
	# 	else:
	# 		if X_new[game_id][0] == 1:
	# 			X_new[game_id] = np.hstack((X_new[game_id],row))
	# 		else:
	# 			X_new[game_id] = np.hstack((row,X_new[game_id]))
	# 			y_new[game_id] = score


	# X = None
	# y = None
	# flag = False
	# g = []
	# print("re numpy-fying")
	# for game_id in set(game_ids):
	# 	if X_new[game_id].shape[0] == 21:
	# 		continue
	# 	g.append(game_id)

	# 	# print("shape: ", X_new[game_id].shape)
	# 	# print("shape: ", X_new[game_id].reshape(-1,1).shape)
	# 	# print("game_id: ",game_id," shows up ",game_ids.count(game_id)," times...")
	# 	if not flag:
	# 		X = X_new[game_id]
			
	# 		y = [y_new[game_id]]
	# 		# print()
	# 		flag = True

	# 	else:
	# 		X = np.vstack((X,X_new[game_id]))
	# 		y.append(y_new[game_id])



	# y = np.array(y)




			




	# with open("team_percentile_X.pkl","rb") as f:
	# 	X = pickle.load(f)

	# with open("team_percentile_y.pkl","rb") as f:
	# 	y = pickle.load(f)

	# with open("team_percentile_features.pkl", "rb") as f:
	# 	feature_names = pickle.load(f)


	# with open("team_percentile_X.pkl","wb") as f:
	# 	pickle.dump(X, f)

	# with open("team_percentile_y.pkl","wb") as f:
	# 	pickle.dump(y, f)

	# with open("team_percentile_features.pkl", "wb") as f:
	# 	pickle.dump(feature_names, f)


	# print(X.shape)
	# print(y.shape)
	# quit()

	# print(np.mean(y))

	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=True,stratify=y)


	X_t1 = X_train[:,:21]
	X_t2 = X_train[:,21:]

	X_train = np.vstack((X_t1,X_t2))
	y_t1 = y_train
	y_t2 = 1-y_train
	y_train = np.hstack((y_t1,y_t2))

	print(X_train.shape)
	print(y_train.shape)
	# quit()

	clf = RandomForestClassifier(n_estimators=100,verbose=1)
	clf.fit(X_train , y_train)
	feature_weights = clf.feature_importances_

	features = [(i,j) for i,j in zip(feature_names,feature_weights)]
	features = sorted(features,key=lambda x: x[1],reverse=True)

	for i, j in features:

		print(i,j)





	X_t1 = X_test[:,:21]
	X_t2 = X_test[:,21:]

	X_test = np.vstack((X_t1,X_t2))
	y_t1 = y_test
	y_t2 = 1-y_test
	y_test = np.hstack((y_t1,y_t2))

	pred = clf.predict(X_test)
	total = 0
	for i, j in zip(pred,y_test):
		total += i == j

	print("acc: ", total / len(y_test))











		# team_games_played = [i for i in range(len(team_season_games))]
		# home_game_stats = getHomeGameStats(team_season_games)









NHL [ season ]
		[ game1 , ... , game1200 ]


TEAM [ season ]
		[ game1 , ... , game82 ]
			[ "id" ] -> game_id
			[ "home" ] -> home_id
			[ "away" ] -> away_id
			[ "date" ] - > date






'''












