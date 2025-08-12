from argparse import ArgumentParser, Namespace
from copy import deepcopy

from src.server.fedavg import FedAvgServer     #get_fedavg_argparser
from src.client.pflmf import pFLMFClient



class pFLMFServer(FedAvgServer):
    algorithm_name = str = "pFLMF"
    all_model_params_personalized = False
    return_diff = False  # Sending parameter differences
    client_cls = pFLMFClient


    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--num_steps_v", type=float, default=1, help="pFLMF's number of v steps")
        parser.add_argument("--num_steps_u", type=float, default=1, help="pFLMF's number of u steps")
        parser.add_argument("--lr_v", type=float, default=0.0001, help="pFLMF's learning rate of v")
        parser.add_argument("--lr_u", type=float, default=0.001, help="pFLMF's learning rate of u")     

        return parser.parse_args(args_list)

