import random
import pickle
import os
import numpy as np

# Team SVM Libraries
import itertools
import time


class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number  # index for this agent
        self.opponent_number = 1 - agent_number  # index for opponent
        self.n_items = params["n_items"]

        # Unpickle the trained model
        # Complications: pickle should work with any machine learning models
        # However, this does not work with custom defined classes, due to the way pickle operates
        # TODO you can replace this with your own model

        self.filename1 = "machine_learning_model/final_model_covs_and_noisy"
        self.filename2 = "machine_learning_model/final_model_covs_only"
        self.trained_model_covs_and_noisy = pickle.load(open(self.filename1, "rb"))
        self.trained_model_covs_only = pickle.load(open(self.filename2, "rb"))

        # Training Mean and Standard Deviation for Normalization
        self.train_means = np.array([0.00534622, 0.00412864, 0.00322634])
        self.train_stds = np.array(
            [0.9274842, 0.86229847, 0.72909165]
        )  # variance = [0.86022694, 0.74355865, 0.53157464]

        # Item Embeddings
        self.item0_embedding = pickle.load(open("data/item0embedding", "rb"))
        self.item1_embedding = pickle.load(open("data/item1embedding", "rb"))

        # time variable for tracking how fast our program runs
        self.time = 0
        self.iter = 0

        # competitor pricing strategy
        self.alpha = 1

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

        # print("My current profit: ", my_current_profit)
        # print("Opponent current profit: ", opponent_current_profit)
        # print("My last prices: ", my_last_prices)
        # print("Opponent last prices: ", opponent_last_prices)
        # print("Did customer buy from me: ", did_customer_buy_from_me)
        # print("Did customer buy from opponent: ", did_customer_buy_from_opponent)
        # print("Which item customer bought: ", which_item_customer_bought)
        # print("Time to run last iteration: ", self.time)

        if did_customer_buy_from_me:  # can increase prices
            self.alpha *= 1.1
        else:  # should decrease prices
            self.alpha *= 0.9

    # Given an observation which is #info for new buyer, information for last iteration, and current profit from each time
    # Covariates of the current buyer, and potentially embedding. Embedding may be None
    # Data from last iteration (which item customer purchased, who purchased from, prices for each agent for each item (2x2, where rows are agents and columns are items)))
    # Returns an action: a list of length n_items=2, indicating prices this agent is posting for each item.
    def action(self, obs):
        new_buyer_covariates, new_buyer_embedding, last_sale, profit_each_team = obs
        if self.iter > 0:
            self._process_last_sale(last_sale, profit_each_team)

        self.time = time.time()  # start timer
        covs = self.normalize_covs(new_buyer_covariates)
        if new_buyer_embedding is not None:
            vector = self.get_user_item_vectors(new_buyer_embedding)
            full_covs = np.concatenate((covs, vector))
            prices, rev = self.find_optimal_revenue_fast(
                self.trained_model_covs_and_noisy, full_covs
            )
        else:
            self.last_full_covs = None
            prices, rev = self.find_optimal_revenue_fast(
                self.trained_model_covs_only, covs
            )

        prices = [self.alpha * p for p in prices]

        self.time = time.time() - self.time  # end timer
        self.iter += 1

        return prices

    def normalize_covs(self, covariate):
        # z = (x - u) / s
        return (covariate - self.train_means) / self.train_stds

    def get_user_item_vectors(self, user_vectors):
        items0 = np.dot(
            user_vectors, np.array(self.item0_embedding, dtype=np.float64).T
        )
        items1 = np.dot(
            user_vectors, np.array(self.item1_embedding, dtype=np.float64).T
        )
        return np.column_stack((items0, items1))[0]

    def get_demand_predict(self, model, prices, covariates):
        variables = np.concatenate((list(prices), covariates))
        # remove the prediction for no products purchased
        return model.predict_proba([variables])[0][1:]

    def revenue_maximization(self, prices, demands):
        r = [
            p1 * d1 + p2 * d2
            for p1, p2, d1, d2 in [
                list(itertools.chain(*i)) for i in zip(prices, demands)
            ]
        ]
        mr = max(r)
        ind = r.index(mr)
        return prices[ind], mr

    def find_optimal_revenue_fast(self, model, user_covariates, eps=1e-10):
        pr_1 = np.linspace(0, 2.3, 10)
        pr_2 = np.linspace(0, 4, 10)
        diff = 1
        r_old = 0
        while diff > eps:
            prices = list(itertools.product(pr_1, pr_2))
            demands = [
                self.get_demand_predict(model, x, user_covariates) for x in prices
            ]
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
