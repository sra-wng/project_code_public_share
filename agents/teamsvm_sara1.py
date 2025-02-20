import random
import pickle
import os
import numpy as np

# Team SVM Libraries
import itertools
import time
from sklearn.linear_model import Ridge


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
        self.epsilon = 1e-6
        self.alpha = 1
        # self.opponent_alpha = 1
        self.winning_streak = 0
        self.losing_streak = 0
        self.positive_weights = [
            1.2,
            1.15,
            1.1,
            1.09,
            1.08,
            1.07,
            1.06,
            1.05,
            1.05,
            1.03,
        ]
        self.penalty_weights = [
            0.8,
            0.85,
            0.9,
            0.91,
            0.92,
            0.93,
            0.94,
            0.95,
            0.96,
            0.97,
        ]
        self.my_prices = []
        self.my_ideal_prices = []
        self.opponent_prices = []
        self.agent_winner = []
        self.item_purchased = []
        self.all_covs = []
        self.lose_on_purpose = False
        self.opponent_last_alpha = 1
        self.lose_on_purpose_list = []
        # self.new_models = False

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

        # TEAM SVM CODE STARTS HERE
        self.my_prices.append(my_last_prices)
        self.opponent_prices.append(opponent_last_prices)
        self.agent_winner.append(last_sale[1])
        self.item_purchased.append(which_item_customer_bought)

        i = which_item_customer_bought

        # update alpha in first round
        if self.iter == 1 and did_customer_buy_from_opponent:
            self.alpha = opponent_last_prices[i] / self.my_ideal_prices[0][i]

        # check if opponent increased their alpha this round if we lost on purpose in the previous round
        elif self.iter > 1 and self.lose_on_purpose_list[-2]:
            self.opponent_last_alpha = (
                opponent_last_prices[i] / self.opponent_prices[-2][i]
            )
            if self.opponent_last_alpha > 1:
                print("opponent took the bait and increased alpha")
            elif self.opponent_last_alpha < 1:
                print("opponent decreased alpha")
            else:
                print("opponent alpha is the same")

        elif not self.lose_on_purpose:
            self.winning_streak = (
                self.winning_streak + 1 if did_customer_buy_from_me else 0
            )
            self.winning_streak = (
                len(self.positive_weights) - 1
                if self.winning_streak > len(self.positive_weights) - 1
                else self.winning_streak
            )
            self.losing_streak = (
                self.losing_streak + 1 if did_customer_buy_from_me else 0
            )
            self.losing_streak = (
                len(self.penalty_weights) - 1
                if self.losing_streak > len(self.penalty_weights) - 1
                else self.losing_streak
            )
            self.alpha *= (
                self.positive_weights[self.winning_streak]
                if did_customer_buy_from_me
                else self.penalty_weights[self.losing_streak]
            )
        else:
            self.alpha *= 1.05

        self.alpha = 1 if self.alpha > 1 else self.alpha

        # add forgiveness if the alpha goes too low
        self.alpha = (
            0.75 if (self.alpha < 0.35 and random.uniform(0, 1) < 0.08) else self.alpha
        )

        # Learn my customer's prices
        # if self.iter % 100 == 0:
        #     self.new_models = True
        #     X = self.all_covs
        #     y_price0 = [p[0] for p in self.opponent_prices]
        #     y_price1 = [p[1] for p in self.opponent_prices]
        #     self.model_price0 = Ridge(max_iter=500).fit(X,y_price0)
        #     self.model_price1 = Ridge(max_iter=500).fit(X,y_price1)

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
        # self.all_covs.append(covs)
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

        prices = list(prices)
        self.my_ideal_prices.append(prices)
        # Fixed Pricing Defense
        fixed = False
        if len(self.opponent_prices) > 5:
            if all(
                x[0] == self.opponent_prices[-1][0] for x in self.opponent_prices[-3:]
            ):
                fixed = True
                if prices[0] > self.opponent_prices[-1][0]:
                    prices[0] = self.opponent_prices[-1][0] - self.epsilon
            if all(
                x[1] == self.opponent_prices[-1][1] for x in self.opponent_prices[-3:]
            ):
                fixed = True
                if prices[1] > self.opponent_prices[-1][1]:
                    prices[1] = self.opponent_prices[-1][1] - self.epsilon
        if not fixed:
            prices = [self.alpha * p for p in prices]
            # Purposely lose low revenue items to improve alpha to our benefit
            if (rev < 1.05) and self.alpha > 0.2:
                self.lose_on_purpose = True
                prices = [1000000000, 1000000000]
            else:
                self.lose_on_purpose = False
                if rev > 1.95:  # 90% discount to make sure we capture these customers
                    prices = [0.9 * p for p in prices]
            self.lose_on_purpose_list.append(self.lose_on_purpose)

        # Guard against negative prices
        prices = [self.epsilon if p <= 0 else p for p in prices]

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
