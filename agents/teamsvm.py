import random
import pickle
import os
import numpy as np

# Team SVM Libraries
import itertools
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

        self.iter = 0

        # competitor pricing strategy
        self.epsilon = 1e-6
        self.rev_sacrifice = 1.05
        self.alpha = 1
        self.alpha_list = []
        self.opponent_alpha = 1
        self.opponent_alpha_list = []
        self.opponent_logic_list = []
        self.opp_price_mean = [0, 0]
        self.my_prices = []
        self.my_ideal_prices = []
        self.opponent_prices = []
        self.agent_winner = []
        self.item_purchased = []
        self.lose_on_purpose = False
        self.illogical = False

    def _process_last_sale(self, last_sale, profit_each_team):
        my_current_profit = profit_each_team[self.this_agent_number]
        opponent_current_profit = profit_each_team[self.opponent_number]

        my_last_prices = last_sale[2][self.this_agent_number]
        opponent_last_prices = last_sale[2][self.opponent_number]

        did_customer_buy_from_me = last_sale[1] == self.this_agent_number
        did_customer_buy_from_opponent = last_sale[1] == self.opponent_number

        which_item_customer_bought = last_sale[0]

        # TEAM SVM CODE STARTS HERE
        self.my_prices.append(my_last_prices)
        self.opponent_prices.append(opponent_last_prices)
        self.agent_winner.append(last_sale[1])
        self.item_purchased.append(which_item_customer_bought)

        # determine opponents estimated alpha from last 7 turns
        opp_prices_no_outliers = []
        for p in self.opponent_prices:
            if (0 <= p[0] <= 5) and (0 <= p[1] <= 5):
                opp_prices_no_outliers.append(p)
        if len(opp_prices_no_outliers) > 7:
            self.opp_price_mean = np.mean(opp_prices_no_outliers[-7:], axis=0)
            my_ideal_price_mean = np.mean(self.my_ideal_prices[-7:], axis=0)
            self.opponent_alpha = np.mean(
                self.opp_price_mean / my_ideal_price_mean, axis=0
            )
            self.opponent_alpha_list.append(self.opponent_alpha)

            # confirm opponent has a logical alpha
            if len(self.opponent_alpha_list) > 3:
                if self.agent_winner[-2] == self.opponent_number:
                    if self.opponent_alpha_list[-1] >= self.opponent_alpha_list[-2]:
                        self.opponent_logic_list.append(True)
                    else:
                        self.opponent_logic_list.append(False)
                else:
                    if self.opponent_alpha_list[-1] <= self.opponent_alpha_list[-2]:
                        self.opponent_logic_list.append(True)
                    else:
                        self.opponent_logic_list.append(False)
                if len(self.opponent_logic_list) > 30:
                    self.illogical = (
                        True
                        if sum(self.opponent_logic_list[-30:])
                        / len(self.opponent_logic_list[-30:])
                        < 0.55
                        else False
                    )

            # confirm opponent increase their alpha after lose on purpose move
            if self.lose_on_purpose:
                if self.opponent_alpha_list[-1] > self.opponent_alpha_list[-2]:
                    self.rev_sacrifice += 0.01 if self.rev_sacrifice < 1.25 else 0
                else:
                    self.rev_sacrifice -= 0.02 if self.rev_sacrifice > 0.02 else 0

            # set base alpha as benevolent opponent alpha
            self.alpha = (
                0.94 * self.alpha
                if self.alpha > self.opponent_alpha
                else 0.94 * self.opponent_alpha
            )

        if self.iter == 1 and did_customer_buy_from_opponent:
            i = which_item_customer_bought
            self.alpha = opponent_last_prices[i] / self.my_ideal_prices[0][i]

        elif not self.lose_on_purpose:
            self.alpha *= 1.15 if did_customer_buy_from_me else 0.95
        else:
            self.alpha *= 1.05

        self.alpha = 1 if self.alpha > 1 else self.alpha
        self.alpha = self.epsilon if self.alpha < 0 else self.alpha

        # add forgiveness if the alpha goes too low
        self.alpha = (
            0.75 if (self.alpha < 0.35 and random.uniform(0, 1) < 0.08) else self.alpha
        )
        self.alpha_list.append(self.alpha)

    # Given an observation which is #info for new buyer, information for last iteration, and current profit from each time
    # Covariates of the current buyer, and potentially embedding. Embedding may be None
    # Data from last iteration (which item customer purchased, who purchased from, prices for each agent for each item (2x2, where rows are agents and columns are items)))
    # Returns an action: a list of length n_items=2, indicating prices this agent is posting for each item.
    def action(self, obs):
        new_buyer_covariates, new_buyer_embedding, last_sale, profit_each_team = obs
        if self.iter > 0:
            self._process_last_sale(last_sale, profit_each_team)

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
        if self.illogical and (self.opp_price_mean.all() != 0):
            prices = [
                random.uniform(0.38 * self.opp_price_mean[0], 0.42 * self.opp_price_mean[0]),
                random.uniform(0.38 * self.opp_price_mean[1], 0.42 * self.opp_price_mean[1]),
            ]
        elif not fixed:
            prices = [self.alpha * p for p in prices]
            # Purposely lose low revenue items to improve alpha to our benefit
            if rev < self.rev_sacrifice and self.alpha < 0.8:
                self.lose_on_purpose = True
                prices = [float("inf"), float("inf")]
            else:
                self.lose_on_purpose = False
                if rev > 1.95:  # 80% discount to make sure we capture these customers
                    prices = [0.8 * p for p in prices]

        # Guard against negative prices
        prices = [self.epsilon if p <= 0 else p for p in prices]

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
