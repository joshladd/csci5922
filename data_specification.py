




percentile = ordinal?


NHL [ season ] -> []

		[ game_id ] -> {}
			[ home_id ] -> categorical
			[ away_id ] -> categorical
			[ date ] -> datetime
			[ game_type ] -> categorical




TEAMSTATS [ season ] -> []

			[ game_id ] -> {}
				[ game_date ] -> datetime
				[ game_win ] -> categorical

				[ travel ] -> {}
					[ timezone ] -> numerical
					[ game_day ] -> categorical
					[ start_time ] -> numerical
					[ distance ] -> numerical
					[ rest_days ] -> numerical
					[ home ] -> categorical
					[ end ] -> categorical

				[ stats ] -> {}

					[ game ] -> {}
						[ goals_for ] -> numerical
						[ goals_against ] -> numerical
						[ shots_for ] -> numerical
						[ shots_against ] -> numerical
						[ PDO ] -> numerical
						[ corsi_for_percentage ] -> numerical
						[ fenwick_for_percentage ] -> numerical
						[ hits ] -> numerical
						[ blocks ] -> numerical
						[ giveaways ] -> numerical
						[ takeaways ] -> numerical
						[ penalty_minutes ] -> numerical
						[ power_plays ] -> numerical
						[ power_play_goals_for ] -> numerical
						[ power_play_goals_against ] -> numerical

					[ cumulative ] -> {}
						[ goals_for ] -> numerical
						[ goals_against ] -> numerical
						[ goal_differential ] -> numerical
						[ shots_for ] -> numerical
						[ shots_against ] -> numerical
						[ wins ] -> numerical
						[ points ] -> numerical
						[ power_play_percentage ] -> numerical
						[ penalty_kill_percentage ] -> numerical
						[ corsi_for_percentage ] -> numerical
						[ fenwick_for_percentage ] -> numerical
						[ penalty_minutes ] -> numerical
						[ blocks ] -> numerical
						[ hits ] -> numerical
						[ PDO ] -> numerical

					[ team_percentile ] -> {}
						[ goals_for ] -> percentile
						[ goals_against ] -> percentile
						[ shots_for ] -> percentile
						[ shots_against ] -> percentile
						[ PDO ] -> percentile
						[ corsi_for_percentage ] -> percentile
						[ fenwick_for_percentage ] -> percentile
						[ hits ] -> percentile
						[ blocks ] -> percentile
						[ power_play_percentage ] -> percentile
						[ penalty_kill_percentage ] -> percentile
						[ penalty_minutes ] -> percentile

					[ league_percentile ] -> {}
						[ goals_for ] -> percentile
						[ goals_against ] -> percentile
						[ goal_differential ] -> percentile
						[ shot_percentage ] -> percentile
						[ save_percentage ] -> percentile
						[ PDO ] -> percentile
						[ wins ] -> percentile
						[ points ] -> percentile
						[ corsi_for_percentage ] -> percentile
						[ fenwick_for_percentage ] -> percentile
						[ penalty_minutes ] -> percentile
						[ power_play_percentage ] -> percentile
						[ penalty_kill_percentage ] -> percentile





'''
PREPROCESSING

1) Normalize appropriate numerical values 0-1
2) Learn categorical variable embeddings
3) Game-instance dataset for feature selection
4) Generate N-length sequences of games for N = [1,5,10,GAMES_PLAYED] (3 or 5 or both?)

*) Standardising? Scaling?
*) Higher-level feature embeddings?
*) PCA? t-SNE? LSA?
*) Feature relationships? confusion matrix? pearson correlation?
*) 

'''


'''
 MODELS
--------

1) Coin flip [ stats , travel ]
	1.1) Weighted flip by Points
	1.2) Weighted flip by Goals
	1.3) Weighted flip by Wins
	1.4) Weighted flip by Distance / Rest Days

2) Multi-coin flip [ stats , travel ]
	2.1) Weighted flip for each feature in 
		[ goals , points , wins , distance , rest_days , home_win_pct vs away_win_pct ]
	2.2) Majority of coin flips determines game

3) Last-N Instance NN [ stats ]
	3.1) team_percentile_average
	3.2) league_percentile

4) Cumulative Instance NN [ stats ]
	4.1) team_cumulative
	4.2) league_percentile
	4.3) travel_cumulative

5) Travel RNN 
	5.1) Last-N [ travel , winner , cumulative_points , last_N_points ]
	5.2) Last-N [ travel , winner , league_percentile , team_percentile , cumulative ]

6) Stats RNN
	6.1) Last-N [ winner , cumulative , league_percentile , team_percentile]


'''




'''
FEATURE SELECTION
'''

feature_sets = [ travel , game , cumulative , team_percentile , league_percentile ]

#eyeball results from 1 & 2 to determine usefulness
#potentially do a combination of discovered features?

#1
for feature_set in feature_sets :

	
	X_train , X_test = split( feature_set )
	y_train , y_test = split( game_win )

	clf = RandomForestClassifier
	clf.fit( X_train , y_train , params = {} )
	clf.predict( X_test , y_test )

	feature_ranking = clf.important_features( feature_names )


#2
for feature_set in feature_sets :

	max_accuracy = 0
	new_feature_set = []

	while new_feature_set.length < feature_set.length :
		max_feature = None

		for feature in feature_set :

			if new_feature_set is empty :
				features = [ feature ]
			elif feature in new_feature_set :
				continue
			else:
				features = new_feature_set + [ feature ]

			X_train , X_test = split( features )
			y_train , y_test = split( game_win )

			clf = RandomForestClassifier
			clf.fit( X_train , y_train , params = {} )
			feature_accuracy = clf.predict( X_test, y_test )

			if feature_accuracy > max_accuracy :
				max_accuracy = feature_accuracy
				max_feature = feature

		if not max_feature :
			break
		else :
			new_feature_set.append( max_feature )















