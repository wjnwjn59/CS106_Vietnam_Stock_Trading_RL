import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing

matplotlib.use("Agg")
import datetime

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.marketdata.vnquantdownloader import vnquantDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent
from finrl.model.models import DRLEnsembleAgent
from finrl.trade.backtest import backtest_stats, backtest_plot, get_daily_return, get_baseline

import itertools



def train_one():
    """
    train an agent
    """
    print("==============Start Fetching Data===========")
    df = YahooDownloader(
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        ticker_list=config.DOW_30_TICKER,
    ).fetch_data()
    ## Fetch data form VN30
    #df = vnquantDownloader(
    #   start_date=config.START_DATE,
    #   end_date=config.END_DATE,
    #   ticker_list=config.VN_30_TICKER,
    #).fetch_data()
    print("==============Start Feature Engineering===========")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=True,
        user_defined_feature=False,
    )

    processed = fe.preprocess_data(df)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))

    processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date','tic'])

    processed_full = processed_full.fillna(0)

    # Training & Trading data split
    train = data_split(processed_full, config.START_DATE, config.START_TRADE_DATE)
    trade = data_split(processed_full, config.START_TRADE_DATE, config.END_DATE)

    # calculate state action space
    stock_dimension = len(train.tic.unique())
    state_space = (
        1
        + 2 * stock_dimension
        + len(config.TECHNICAL_INDICATORS_LIST) * stock_dimension
    )

    env_kwargs = {
        "hmax": 100, 
        "initial_amount": 1000000, 
        "buy_cost_pct": 0.001, 
        "sell_cost_pct": 0.001, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
        "action_space": stock_dimension, 
        "reward_scaling": 1e-4
        }

    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    agent = DRLAgent(env=env_train)

    print("==============Model Training===========")
    now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")

    model_sac = agent.get_model("sac")
    trained_sac = agent.train_model(
        model=model_sac, tb_log_name="sac", total_timesteps=80000
    )

    print("==============Start Trading===========")
    e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=250, **env_kwargs)

    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_sac, environment = e_trade_gym
    )
    df_account_value.to_csv(
        "./" + config.RESULTS_DIR + "/df_account_value_" + now + ".csv"
    )
    df_actions.to_csv("./" + config.RESULTS_DIR + "/df_actions_" + now + ".csv")

    print("==============Get Backtest Results===========")
    perf_stats_all = backtest_stats(df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/perf_stats_all_" + now + ".csv")


# Define train_ensemble_agent()
def train_ensemble_agent():
    print("==============Start Fetching Data===========")
    df = YahooDownloader(
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        ticker_list=config.DOW_30_TICKER,
    ).fetch_data()
    print("==============Start Feature Engineering===========")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=True,
        user_defined_feature=False,
    )

    processed = fe.preprocess_data(df)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))

    processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date','tic'])

    processed_full = processed_full.fillna(0)

    # Training & Trading data split
    train = data_split(processed_full, config.START_DATE, config.START_TRADE_DATE)
    trade = data_split(processed_full, config.START_TRADE_DATE, config.END_DATE)

    # calculate state action space
    stock_dimension = len(train.tic.unique())
    state_space = (
        1
        + 2 * stock_dimension
        + len(config.TECHNICAL_INDICATORS_LIST) * stock_dimension
    )

    train_period = ['2009-01-01', '2015-10-01']
    val_test_period = ['2015-10-01', '2016-01-01']

    env_kwargs = {
        "df": df,
        "train_period" : train_period,
        "val_test_period" : val_test_period,
        "rebalance_window" : 63,
        "validation_window" : 63,
        "stock_dim" : stock_dimension,
        "hmax" : 100 ,
        "initial_amount" : 1000000,
        "buy_cost_pct" : 0.001,
        "sell_cost_pct" : 0.001,
        "reward_scaling" : 1e-4,
        "state_space" : state_space,
        "action_space" : stock_dimension,
        "tech_indicator_list" : config.TECHNICAL_INDICATORS_LIST,
        "print_verbosity" : 10
        }

    timesteps_dict = {
        'a2c' : 100000,
        'ppo' : 100000,
        'ddpg' : 50000
    }

    # e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    # env_train, _ = e_train_gym.get_sb_env()
    ensembleAgent = DRLEnsembleAgent(**env_kwargs)

    print("==============Model Training===========")
    now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")
    ensemble_model = ensembleAgent.run_ensemble_strategy(config.A2C_PARAMS, config.PPO_PARAMS, config.DDPG_PARAMS, timesteps_dict=timesteps_dict)

    training_ensemble_agent = ensembleAgent.run_ensemble_strategy(
                a2c_model, ppo_model, ddpg_model, timesteps_dict=100000)
