from argparse import ArgumentParser, Namespace
from src.client.signfedavg import SignFedAvgClient
from src.server.fedavg import FedAvgServer

class SignFedAvgServer(FedAvgServer):
    algorithm_name: str = "SignFedAvg"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = True  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = SignFedAvgClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--lr_server", type=float, default=1, help="SignFedAvg's parameter")
        return parser.parse_args(args_list)