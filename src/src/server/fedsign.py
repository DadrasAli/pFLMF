from argparse import ArgumentParser, Namespace
from src.client.fedsign import FedSignClient
from src.server.fedavg import FedAvgServer

class FedSignServer(FedAvgServer):
    algorithm_name: str = "FedSign"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates that clients return `diff = W_global - W_local` as parameter update; `False` for `W_local` only.
    client_cls = FedSignClient

    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--p", type=float, default=0.5, help="FedSign's parameter")
        return parser.parse_args(args_list)