import random
import pickle
import os
import numpy as np
# Team SVM Libraries
import itertools
from sklearn import preprocessing
import pandas as pd
#import time


class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number  # index for this agent
        self.opponent_number = 1 - agent_number  # index for opponent
        self.n_items = params["n_items"]

        # Unpickle the trained model
        # Complications: pickle should work with any machine learning models
        # However, this does not work with custom defined classes, due to the way pickle operates
        # TODO you can replace this with your own model
        self.filename1 = 'machine_learning_model/trained_model'
        self.filename2 = 'machine_learning_model/trained_model'
        
        # self.filename1 = 'machine_learning_model/final_model_covs_and_embedded'
        # self.filename2 = 'machine_learning_model/final_model_covs_only'
        self.trained_model_covs_and_noisy = pickle.load(open(self.filename1, 'rb'))
        self.trained_model_covs_only = pickle.load(open(self.filename2, 'rb'))
        
        # Training Mean and Standard Deviation for Normalization
        self.train_mean = 0
        self.train_std = 1
        
        # Item Embeddings
        self.item0embedding = 'data/item0embedding'
        self.item1embedding = 'data/item1embedding'

    def _process_last_sale(self, last_sale, profit_each_team):
        # print("last_sale: ", last_sale)
        # print("profit_each_team: ", profit_each_team)
        my_current_profit = profit_each_team[self.this_agent_number]
        opponent_current_profit = profit_each_team[self.opponent_number]

        my_last_prices = last_sale[2][self.this_agent_number]
        opponent_last_prices = last_sale[2][self.opponent_number]

        did_customer_buy_from_me = last_sale[1] == self.this_agent_number
        did_customer_buy_from_opponent = last_sale[1] == self.opponent_number

        which_item_customer_bought = last_sale[0]

        print("My current profit: ", my_current_profit)
        print("Opponent current profit: ", opponent_current_profit)
        print("My last prices: ", my_last_prices)
        print("Opponent last prices: ", opponent_last_prices)
        print("Did customer buy from me: ", did_customer_buy_from_me)
        print("Did customer buy from opponent: ", did_customer_buy_from_opponent)
        print("Which item customer bought: ", which_item_customer_bought)

        # TODO - add your code here to potentially update your pricing strategy based on what happened in the last round
        pass

    # Given an observation which is #info for new buyer, information for last iteration, and current profit from each time
    # Covariates of the current buyer, and potentially embedding. Embedding may be None
    # Data from last iteration (which item customer purchased, who purchased from, prices for each agent for each item (2x2, where rows are agents and columns are items)))
    # Returns an action: a list of length n_items=2, indicating prices this agent is posting for each item.
    def action(self, obs):
        new_buyer_covariates, new_buyer_embedding, last_sale, profit_each_team = obs
        self._process_last_sale(last_sale, profit_each_team)
        
        # TEAM SVM CODE STARTS HERE
        covs= self.normalize(new_buyer_covariates)
        print(new_buyer_embedding)
        if new_buyer_embedding.size != 0:
            vector = self.get_user_item_vectors(new_buyer_embedding)
            full_covs = np.concatenate((covs, vector))
            p, r = self.find_optimal_revenue_fast(self.trained_model_covs_and_noisy, full_covs)
        else:
            p, r = self.find_optimal_revenue_fast(self.trained_model_covs_only, covs)
        return p
        # TODO Currently this output is just a deterministic 2-d array, but the students are expected to use the buyer covariates to make a better prediction
        # and to use the history of prices from each team in order to create prices for each item.
    
    def normalize(self, covariate):
        # z = (x - u) / s
        return (covariate - self.train_mean) / self.train_std
    
    def get_user_item_vectors(self, user_vectors):
        items0 = np.dot(user_vectors, np.array(self.item0_embedding).T)
        items1 = np.dot(user_vectors, np.array(self.item1_embedding).T)
        return np.column_stack((items0, items1))[0]
    
    def get_demand_predict(self, model, prices, covariates):
        variables = np.concatenate((list(prices),covariates))
        # remove the prediction for no products purchased
        return model.predict_proba([variables])[0][1:]
    
    def revenue_maximization(self, prices, demands):
        r = [p1*d1 + p2*d2 for p1,p2,d1,d2 in [list(itertools.chain(*i)) for i in zip(prices, demands)]]
        mr = max(r)
        ind = r.index(mr)
        return prices[ind], mr
    
    def find_optimal_revenue_fast(self, model, user_covariates, eps = 1e-10):
        pr_1 = np.linspace(0, 2.3, 10)
        pr_2 = np.linspace(0, 4, 10)
        diff = 1
        r_old = 0
        while diff > eps:
            prices = list(itertools.product(pr_1, pr_2))
            demands = [self.get_demand_predict(model, x, user_covariates) for x in prices]
            opt_prices, r_new = self.revenue_maximization(prices, demands)
            diff = r_new - r_old
            r_old = r_new
            ind1 = np.where(pr_1 == opt_prices[0])[0][0]
            ind2 = np.where(pr_2 == opt_prices[1])[0][0]
            ind1_before = ind1 - 1 if ind1 > 0 else 0
            ind1_after = ind1 + 1 if ind1 < 9 else 9
            ind2_before = ind2 - 1 if ind2 > 0 else 0
            ind2_after = ind2 + 1 if ind2 < 9 else 9
            pr_1 = np.linspace(pr_1[ind1_before], pr_1[ind1_after], 10)
            pr_2 = np.linspace(pr_2[ind2_before], pr_2[ind2_after], 10)
        max_revenue = r_new
        return opt_prices, max_revenue