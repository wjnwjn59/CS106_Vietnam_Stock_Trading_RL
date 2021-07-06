import json
import logging
import os
import time
from argparse import ArgumentParser
import datetime

from finrl.config import config


def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        dest="mode",
        help="start mode, train, download_data" " backtest",
        metavar="MODE",
        default="train",
    )
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    if not os.path.exists("./" + config.DATA_SAVE_DIR):
        os.makedirs("./" + config.DATA_SAVE_DIR)
    if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
        os.makedirs("./" + config.TRAINED_MODEL_DIR)
    if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
        os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
    if not os.path.exists("./" + config.RESULTS_DIR):
        os.makedirs("./" + config.RESULTS_DIR)

    if options.mode == "train":
        import finrl.autotrain.training

        finrl.autotrain.training.train_one()
        #finrl.autotrain.training.train_ensemble_agent()

    elif options.mode == "download_data":
        from finrl.marketdata.yahoodownloader import YahooDownloader

        df = YahooDownloader(start_date=config.START_DATE,
                             end_date=config.END_DATE,
                             ticker_list=config.DOW_30_TICKER).fetch_data()
        now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")
        df.to_csv("./" + config.DATA_SAVE_DIR + "/" + now + ".csv")

        # Download command with vietnamese dataset
        from finrl.marketdata.vnquantdownloader import vnquantDownloader
        from finrl.preprocessing.preprocessors import FeatureEngineer

        # df = vnquantDownloader(start_date=config.START_DATE,
        #                        end_date=config.END_DATE,
        #                        ticker_list=config.VN_30_TICKER, ).fetch_data()

        fe = FeatureEngineer(
                        use_technical_indicator=True,
                        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
                        use_turbulence=True,
                        user_defined_feature=False,
        )

        processed = fe.preprocess_data(df)


        now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")
        processed.to_csv("./" + config.DATA_SAVE_DIR + "/" + now + ".csv")


        
        
if __name__ == "__main__":
    main()
