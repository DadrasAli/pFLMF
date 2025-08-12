import functools
import inspect
import json
import os
import pickle
import random
import shutil
import time
import traceback
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy as np
import ray
import torch
import csv  # <-- NEW: Import csv
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.pretty import pprint as rich_pprint
from rich.progress import track
from torchvision import transforms

from data.utils.datasets import DATASETS, BaseDataset
from src.client.fedavg import FedAvgClient
from src.utils.constants import (
    DATA_MEAN,
    DATA_STD,
    FLBENCH_ROOT,
    LR_SCHEDULERS,
    MODE,
    OPTIMIZERS,
)
from src.utils.functional import (
    evaluate_model,
    fix_random_seed,
    get_optimal_cuda_device,
    initialize_data_loaders,
)
from src.utils.logger import Logger
from src.utils.metrics import Metrics
from src.utils.models import MODELS, DecoupledModel
from src.utils.trainer import FLbenchTrainer


class FedAvgServer:
    algorithm_name = "FedAvg"
    all_model_params_personalized = False  # `True` indicates that clients have their own fullset of personalized model parameters.
    return_diff = False  # `True` indicates clients return `diff = W_global - W_local`; `False` => sending W_local.
    client_cls = FedAvgClient

    def __init__(
        self,
        args: DictConfig,
        init_trainer=True,
        init_model=True,
    ):
        """
        Args:
            args: A DictConfig of all arguments.
            init_trainer: True => trainer is created now.
            init_model: True => model is initialized now.
        """
        self.args = args

        self.device = get_optimal_cuda_device(self.args.common.use_cuda)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.set_device(self.device)

        fix_random_seed(self.args.common.seed, use_cuda=self.device.type == "cuda")

        self.output_dir = Path(HydraConfig.get().runtime.output_dir)
        with open(
            FLBENCH_ROOT / "data" / self.args.dataset.name / "args.json", "r"
        ) as f:
            self.args.dataset.update(DictConfig(json.load(f)))

        # get client partition info
        try:
            partition_path = (
                FLBENCH_ROOT / "data" / self.args.dataset.name / "partition.pkl"
            )
            with open(partition_path, "rb") as f:
                self.data_partition = pickle.load(f)
        except:
            raise FileNotFoundError(f"Please partition {self.args.dataset.name} first.")
        self.train_clients: List[int] = self.data_partition["separation"]["train"]
        self.test_clients: List[int] = self.data_partition["separation"]["test"]
        self.val_clients: List[int] = self.data_partition["separation"]["val"]
        self.client_num: int = self.data_partition["separation"]["total"]

        # init model parameters
        if init_model:
            self.init_model()

        self.client_optimizer_states = {i: {} for i in range(self.client_num)}
        self.client_lr_scheduler_states = {i: {} for i in range(self.client_num)}

        self.client_local_epoches: List[int] = [
            self.args.common.local_epoch
        ] * self.client_num

        # system heterogeneity (straggler) setting
        if (
            self.args.common.straggler_ratio > 0
            and self.args.common.local_epoch
            > self.args.common.straggler_min_local_epoch
        ):
            straggler_num = int(self.client_num * self.args.common.straggler_ratio)
            normal_num = self.client_num - straggler_num
            self.client_local_epoches = [self.args.common.local_epoch] * normal_num + \
                random.choices(
                    range(
                        self.args.common.straggler_min_local_epoch,
                        self.args.common.local_epoch,
                    ),
                    k=straggler_num,
                )
            random.shuffle(self.client_local_epoches)

        # pre-generate which clients are selected each round
        self.client_sample_stream = [
            random.sample(
                self.train_clients,
                max(1, int(self.client_num * self.args.common.join_ratio)),
            )
            for _ in range(self.args.common.global_epoch)
        ]
        self.selected_clients: List[int] = []
        self.current_epoch = 0

        # for controlling special behaviors while testing
        self.testing = False

        # ensure output directory
        if not os.path.isdir(self.output_dir) and (
            self.args.common.save_log
            or self.args.common.save_learning_curve_plot
            or self.args.common.save_metrics
        ):
            os.makedirs(self.output_dir, exist_ok=True)

        # track client metrics
        self.client_metrics = {i: {} for i in self.train_clients}
        self.aggregated_client_metrics = {
            "before": {"train": [], "val": [], "test": []},
            "after": {"train": [], "val": [], "test": []},
        }

        self.verbose = False
        stdout = Console(log_path=False, log_time=False, soft_wrap=True, tab_size=4)
        self.logger = Logger(
            stdout=stdout,
            enable_log=self.args.common.save_log,
            logfile_path=self.output_dir / "main.log",
        )
        self.test_results: Dict[int, Dict[str, Dict[str, Metrics]]] = {}

        # progress bar
        self.train_progress_bar = track(
            range(self.args.common.global_epoch),
            "[bold green]Training...",
            console=stdout,
        )

        # monitor tool setup
        if self.args.common.monitor is not None:
            self.monitor_window_name_suffix = (
                self.args.dataset.monitor_window_name_suffix
            )

        if self.args.common.monitor == "visdom":
            from visdom import Visdom
            self.viz = Visdom()
        elif self.args.common.monitor == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard = SummaryWriter(log_dir=self.output_dir)

        # init dataset + data indices
        self.dataset = self.get_dataset()
        self.client_data_indices = self.get_clients_data_indices()

        # create trainer
        self.trainer: FLbenchTrainer = None
        if self.client_cls is None or not issubclass(self.client_cls, FedAvgClient):
            raise ValueError(f"{self.client_cls} is not a subclass of {FedAvgClient}.")
        if init_trainer:
            self.init_trainer()

        # create loaders for centralized evaluation
        if 0 < self.args.common.test.server.interval <= self.args.common.global_epoch:
            if self.all_model_params_personalized:
                self.logger.warn(
                    "Warning: Centralized evaluation is not supported for unique model setting."
                )
            else:
                (
                    self.trainloader,
                    self.testloader,
                    self.valloader,
                    self.trainset,
                    self.testset,
                    self.valset,
                ) = initialize_data_loaders(
                    self.dataset, self.client_data_indices, self.args.common.batch_size
                )

        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

    def init_model(
        self,
        model: Optional[DecoupledModel] = None,
        preprocess_func: Optional[Callable[[DecoupledModel], None]] = None,
        postprocess_func: Optional[Callable[[DecoupledModel], None]] = None,
    ):
        """Initialize global model & possibly client personal params."""
        if model is None:
            self.model: DecoupledModel = MODELS[self.args.model.name](
                dataset=self.args.dataset.name,
                pretrained=self.args.model.use_torchvision_pretrained_weights,
            )
        else:
            self.model = model

        self.model.check_and_preprocess(self.args)

        if preprocess_func is not None:
            preprocess_func(self.model)

        _init_global_params, _init_global_params_name = [], []
        for key, param in self.model.named_parameters():
            _init_global_params.append(param.data.clone())
            _init_global_params_name.append(key)

        self.public_model_param_names = _init_global_params_name
        self.public_model_params: OrderedDict[str, torch.Tensor] = OrderedDict(
            zip(_init_global_params_name, _init_global_params)
        )

        if self.args.model.external_model_weights_path is not None:
            file_path = str(
                (FLBENCH_ROOT / self.args.model.external_model_weights_path).absolute()
            )
            if os.path.isfile(file_path) and file_path.find(".pt") != -1:
                self.public_model_params.update(
                    torch.load(file_path, map_location="cpu")
                )
            elif not os.path.isfile(file_path):
                raise FileNotFoundError(f"{file_path} is not a valid file path.")
            elif file_path.find(".pt") == -1:
                raise TypeError(f"{file_path} is not a valid .pt file.")

        # for local buffer or personal model
        self.clients_personal_model_params = {i: {} for i in range(self.client_num)}

        if self.args.common.buffers == "local":
            _init_buffers = OrderedDict(self.model.named_buffers())
            for i in range(self.client_num):
                self.clients_personal_model_params[i] = deepcopy(_init_buffers)

        if self.all_model_params_personalized:
            for params_dict in self.clients_personal_model_params.values():
                params_dict.update(deepcopy(self.model.state_dict()))

        if postprocess_func is not None:
            postprocess_func(self.model)

    def init_trainer(self, **extras):
        """Initialize FLbenchTrainer for either serial or parallel mode."""
        if self.args.mode == "serial" or self.args.parallel.num_workers < 2:
            self.trainer = FLbenchTrainer(
                server=self,
                client_cls=self.client_cls,
                mode=MODE.SERIAL,
                num_workers=0,
                init_args=dict(
                    model=deepcopy(self.model),
                    optimizer_cls=self.get_client_optimizer_cls(),
                    lr_scheduler_cls=self.get_client_lr_scheduler_cls(),
                    args=self.args,
                    dataset=self.dataset,
                    data_indices=self.client_data_indices,
                    device=self.device,
                    return_diff=self.return_diff,
                    **extras,
                ),
            )
        else:
            # parallel mode
            model_ref = ray.put(self.model.cpu())
            optimzier_cls_ref = ray.put(self.get_client_optimizer_cls())
            lr_scheduler_cls_ref = ray.put(self.get_client_lr_scheduler_cls())
            dataset_ref = ray.put(self.dataset)
            data_indices_ref = ray.put(self.client_data_indices)
            args_ref = ray.put(self.args)
            device_ref = ray.put(None)  # each worker picks its own GPU
            return_diff_ref = ray.put(self.return_diff)

            self.trainer = FLbenchTrainer(
                server=self,
                client_cls=self.client_cls,
                mode=MODE.PARALLEL,
                num_workers=int(self.args.parallel.num_workers),
                init_args=dict(
                    model=model_ref,
                    optimizer_cls=optimzier_cls_ref,
                    lr_scheduler_cls=lr_scheduler_cls_ref,
                    args=args_ref,
                    dataset=dataset_ref,
                    data_indices=data_indices_ref,
                    device=device_ref,
                    return_diff=return_diff_ref,
                    **{key: ray.put(value) for key, value in extras.items()},
                ),
            )

    def get_clients_data_indices(self) -> list[dict[str, list[int]]]:
        """Return each client's train/val/test indices from partition."""
        try:
            partition_path = (
                FLBENCH_ROOT / "data" / self.args.dataset.name / "partition.pkl"
            )
            with open(partition_path, "rb") as f:
                partition = pickle.load(f)
        except:
            raise FileNotFoundError(f"Please partition {self.args.dataset.name} first.")
        return partition["data_indices"]

    def get_dataset(self) -> BaseDataset:
        """Load the dataset according to the config."""
        dataset: BaseDataset = DATASETS[self.args.dataset.name](
            root=FLBENCH_ROOT / "data" / self.args.dataset.name,
            args=self.args.dataset,
            **self.get_dataset_transforms(),
        )
        return dataset

    def get_dataset_transforms(self):
        """Define data preprocessing transforms for train/test splits."""
        test_data_transform = transforms.Compose(
            [
                transforms.Normalize(
                    DATA_MEAN[self.args.dataset.name], DATA_STD[self.args.dataset.name]
                )
            ]
            if self.args.dataset.name in DATA_MEAN
            and self.args.dataset.name in DATA_STD
            else []
        )
        test_target_transform = transforms.Compose([])

        train_data_transform = transforms.Compose(
            [
                transforms.Normalize(
                    DATA_MEAN[self.args.dataset.name], DATA_STD[self.args.dataset.name]
                )
            ]
            if self.args.dataset.name in DATA_MEAN
            and self.args.dataset.name in DATA_STD
            else []
        )
        train_target_transform = transforms.Compose([])

        return dict(
            train_data_transform=train_data_transform,
            train_target_transform=train_target_transform,
            test_data_transform=test_data_transform,
            test_target_transform=test_target_transform,
        )

    def get_client_optimizer_cls(self) -> type[torch.optim.Optimizer]:
        """Partial-init the client optimizer class with the config-provided args."""
        target_optimizer_cls: type[torch.optim.Optimizer] = OPTIMIZERS[
            self.args.optimizer.name
        ]
        keys_required = inspect.getfullargspec(target_optimizer_cls.__init__).args
        args_valid = {}
        for key, value in self.args.optimizer.items():
            if key in keys_required:
                args_valid[key] = value

        optimizer_cls = functools.partial(target_optimizer_cls, **args_valid)
        args_valid["name"] = self.args.optimizer.name
        self.args.optimizer = DictConfig(args_valid)
        return optimizer_cls

    def get_client_lr_scheduler_cls(
        self,
    ) -> Union[type[torch.optim.lr_scheduler.LRScheduler], None]:
        """Partial-init the client LR scheduler, if any."""
        if hasattr(self.args, "lr_scheduler"):
            if self.args.lr_scheduler.name is None:
                del self.args.lr_scheduler
                return None
            lr_scheduler_args = getattr(self.args, "lr_scheduler")
            if lr_scheduler_args.name is not None:
                target_scheduler_cls: type[torch.optim.lr_scheduler.LRScheduler] = (
                    LR_SCHEDULERS[lr_scheduler_args.name]
                )
                keys_required = inspect.getfullargspec(target_scheduler_cls.__init__).args
                args_valid = {}
                for key, value in self.args.lr_scheduler.items():
                    if key in keys_required:
                        args_valid[key] = value

                lr_scheduler_cls = functools.partial(target_scheduler_cls, **args_valid)
                args_valid["name"] = self.args.lr_scheduler.name
                self.args.lr_scheduler = DictConfig(args_valid)
                return lr_scheduler_cls
        return None

    def train(self):
        """Runs the main FL training loop over global rounds."""
        avg_round_time = 0
        for E in self.train_progress_bar:
            self.current_epoch = E
            self.verbose = ((E + 1) % self.args.common.verbose_gap == 0)

            if self.verbose:
                self.logger.log("-" * 28, f"TRAINING EPOCH: {E + 1}", "-" * 28)

            self.selected_clients = self.client_sample_stream[E]
            begin = time.time()
            self.train_one_round()
            end = time.time()
            avg_round_time = (avg_round_time * E + (end - begin)) / (E + 1)

            # server-side evaluation if needed
            if (
                self.args.common.test.server.interval > 0
                and (E + 1) % self.args.common.test.server.interval == 0
            ):
                self.test_global_model()

            # client-side test if needed
            if (
                self.args.common.test.client.interval > 0
                and (E + 1) % self.args.common.test.client.interval == 0
            ):
                self.test_client_models()

            self.display_metrics()

        self.logger.log(
            f"{self.algorithm_name}'s average time per global epoch: "
            f"{int(avg_round_time // 60)} min {(avg_round_time % 60):.2f} sec."
        )

        # Save losses after training finishes
        self.save_losses()

    # (A) HELPER TO LOG EACH CLIENT'S BEFORE/AFTER METRICS
    # def save_client_metrics_to_csv(self, client_packages, csv_path="client_metrics.csv"):
    #     """
    #     Records for each client_id and each split (train/val/test):
    #         [client_id, split, loss_before, accuracy_before, loss_after, accuracy_after].
    #     """
    #     write_header = not os.path.exists(csv_path)
    #     with open(csv_path, "a", newline="") as f:
    #         writer = csv.writer(f)
    #         if write_header:
    #             writer.writerow([
    #                 "client_id",
    #                 "split",
    #                 "loss_before",
    #                 "accuracy_before",
    #                 "loss_after",
    #                 "accuracy_after"
    #             ])

    #         for cid, pack in client_packages.items():
    #             eval_res = pack["eval_results"]  # dictionary with "before"/"after"
    #             for split in ["train", "val", "test"]:
    #                 m_before = eval_res["before"][split]
    #                 m_after  = eval_res["after"][split]
    #                 # Only write if it has any data
    #                 if m_before.size > 0 or m_after.size > 0:
    #                     row = [
    #                         cid,
    #                         split,
    #                         m_before.loss,
    #                         m_before.accuracy,
    #                         m_after.loss,
    #                         m_after.accuracy
    #                     ]
    #                     writer.writerow(row)

    def train_one_round(self):
        """Single communication round: clients train -> aggregator -> log metrics."""
        client_packages = self.trainer.train()
        self.aggregate_client_updates(client_packages)

        # (B) CALL THE FUNCTION THAT WRITES 'BEFORE' & 'AFTER' TO CSV
        #self.save_client_metrics_to_csv(client_packages, "client_metrics.csv")

        # Then track average train loss "after" for local use
        avg_train_loss = sum(
            cp["eval_results"]["after"]["train"].loss
            for cp in client_packages.values()
        ) / len(client_packages)
        self.train_losses.append(avg_train_loss)

    def package(self, client_id: int):
        """Packages the model params/etc. for the selected client."""
        return dict(
            client_id=client_id,
            local_epoch=self.client_local_epoches[client_id],
            **self.get_client_model_params(client_id),
            optimizer_state=self.client_optimizer_states[client_id],
            lr_scheduler_state=self.client_lr_scheduler_states[client_id],
            return_diff=self.return_diff,
        )

    def test_client_models(self):
        """Client-side test after training finishes, if configured."""
        self.testing = True
        clients = list(set(self.val_clients + self.test_clients))
        template = {
            "before": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
            "after": {"train": Metrics(), "val": Metrics(), "test": Metrics()},
        }

        if len(clients) > 0:
            if self.val_clients == self.train_clients == self.test_clients:
                # single group
                results = {"all_clients": template}
                self.trainer.test(clients, results["all_clients"])
                # gather aggregated test/val losses
                if self.args.common.test.client.test:
                    avg_test_loss = results["all_clients"]["after"]["test"].loss
                    self.test_losses.append(avg_test_loss)
                if self.args.common.test.client.val:
                    avg_val_loss = results["all_clients"]["after"]["val"].loss
                    self.val_losses.append(avg_val_loss)
            else:
                # separate out val/test
                results = {
                    "val_clients": deepcopy(template),
                    "test_clients": deepcopy(template),
                }
                if len(self.val_clients) > 0:
                    self.trainer.test(self.val_clients, results["val_clients"])
                if len(self.test_clients) > 0:
                    self.trainer.test(self.test_clients, results["test_clients"])

                if self.args.common.test.client.test and "test_clients" in results:
                    avg_test_loss = results["test_clients"]["after"]["test"].loss
                    self.test_losses.append(avg_test_loss)
                if self.args.common.test.client.val and "val_clients" in results:
                    avg_val_loss = results["val_clients"]["after"]["val"].loss
                    self.val_losses.append(avg_val_loss)

            # store into self.test_results
            if self.current_epoch + 1 not in self.test_results:
                self.test_results[self.current_epoch + 1] = results
            else:
                self.test_results[self.current_epoch + 1].update(results)

        self.testing = False

    def test_global_model(self):
        """Server-side evaluation if not all personal model params."""
        if any(len(params) for params in self.clients_personal_model_params.values()):
            return
        self.testing = True
        metrics = self.evaluate(
            model_in_train_mode=self.args.common.test.server.model_in_train_mode
        )

        if self.current_epoch + 1 not in self.test_results:
            self.test_results[self.current_epoch + 1] = {
                "centralized": {"before": metrics, "after": metrics}
            }
        else:
            self.test_results[self.current_epoch + 1]["centralized"] = {
                "before": metrics,
                "after": metrics,
            }
        self.testing = False

    @torch.no_grad()
    def evaluate(
        self, model: torch.nn.Module = None, model_in_train_mode: bool = True
    ) -> dict[str, Metrics]:
        """Evaluates the server (global) model on train/val/test set (if available)."""
        target_model = self.model if model is None else model
        self.dataset.eval()
        train_metrics = Metrics()
        val_metrics = Metrics()
        test_metrics = Metrics()
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")

        if hasattr(self, "testset") and len(self.testset) > 0 and self.args.common.test.server.test:
            test_metrics = evaluate_model(
                model=target_model,
                dataloader=self.testloader,
                criterion=criterion,
                device=self.device,
                model_in_train_mode=model_in_train_mode,
            )

        if hasattr(self, "valset") and len(self.valset) > 0 and self.args.common.test.server.val:
            val_metrics = evaluate_model(
                model=target_model,
                dataloader=self.valloader,
                criterion=criterion,
                device=self.device,
                model_in_train_mode=model_in_train_mode,
            )

        if hasattr(self, "trainset") and len(self.trainset) > 0 and self.args.common.test.server.train:
            train_metrics = evaluate_model(
                model=target_model,
                dataloader=self.trainloader,
                criterion=criterion,
                device=self.device,
                model_in_train_mode=model_in_train_mode,
            )
        return {"train": train_metrics, "val": val_metrics, "test": test_metrics}

    def get_client_model_params(self, client_id: int) -> OrderedDict[str, torch.Tensor]:
        """Return global + personal model params for the client."""
        regular_params = deepcopy(self.public_model_params)
        personal_params = self.clients_personal_model_params[client_id]
        return dict(
            regular_model_params=regular_params,
            personal_model_params=personal_params
        )

    @torch.no_grad()
    def aggregate_client_updates(
        self, client_packages: OrderedDict[int, Dict[str, Any]]
    ):
        """Aggregate client model params -> global model (FedAvg)."""
        client_weights = [pkg["weight"] for pkg in client_packages.values()]
        weights = torch.tensor(client_weights, dtype=torch.float32)
        weights = weights / weights.sum()

        if self.return_diff:
            # sum up param differences
            for name, global_param in self.public_model_params.items():
                diffs = torch.stack(
                    [
                        pkg["model_params_diff"][name]
                        for pkg in client_packages.values()
                    ],
                    dim=-1,
                )
                aggregated = torch.sum(diffs * weights, dim=-1)
                self.public_model_params[name].data -= aggregated
        else:
            # standard FedAvg
            for name, global_param in self.public_model_params.items():
                client_params = torch.stack(
                    [
                        pkg["regular_model_params"][name]
                        for pkg in client_packages.values()
                    ],
                    dim=-1,
                )
                aggregated = torch.sum(client_params * weights, dim=-1)
                global_param.data = aggregated

        self.model.load_state_dict(self.public_model_params, strict=False)

    def display_metrics(self):
        """
        Display aggregated client and/or server metrics at each round.
        We aggregate from self.client_metrics for selected clients.
        """
        for split, client_side_test_flag, server_side_test_flag in [
            ("train", self.args.common.test.client.train, self.args.common.test.server.train),
            ("val",   self.args.common.test.client.val,   self.args.common.test.server.val),
            ("test",  self.args.common.test.client.test,  self.args.common.test.server.test),
        ]:
            for stage in ["before", "after"]:
                if client_side_test_flag:
                    aggregated = Metrics()
                    for i in self.selected_clients:
                        aggregated.update(
                            self.client_metrics[i][self.current_epoch][stage][split]
                        )
                    self.aggregated_client_metrics[stage][split].append(aggregated)

                    # Optionally plot
                    if self.args.common.monitor == "visdom":
                        self.viz.line(
                            [aggregated.accuracy],
                            [self.current_epoch],
                            win=(f"Accuracy-{self.monitor_window_name_suffix}/"
                                 f"{split}set-{stage}LocalTraining"),
                            update="append",
                            name=self.algorithm_name,
                            opts=dict(
                                title=(f"Accuracy-{self.monitor_window_name_suffix}/"
                                       f"{split}set-{stage}LocalTraining"),
                                xlabel="Communication Rounds",
                                ylabel="Accuracy",
                                legend=[self.algorithm_name],
                            ),
                        )
                    elif self.args.common.monitor == "tensorboard":
                        self.tensorboard.add_scalar(
                            f"Accuracy-{self.monitor_window_name_suffix}/"
                            f"{split}set-{stage}LocalTraining",
                            aggregated.accuracy,
                            self.current_epoch,
                            new_style=True,
                        )

            # log server side if available
            if (
                server_side_test_flag
                and self.current_epoch + 1 in self.test_results
                and "centralized" in self.test_results[self.current_epoch + 1]
            ):
                if self.args.common.monitor == "visdom":
                    self.viz.line(
                        [
                            self.test_results[self.current_epoch + 1]["centralized"][
                                "after"
                            ][split].accuracy
                        ],
                        [self.current_epoch + 1],
                        win=(f"Accuracy-{self.monitor_window_name_suffix}/"
                             f"{split}set-CentralizedEvaluation"),
                        update="append",
                        name=self.algorithm_name,
                        opts=dict(
                            title=(f"Accuracy-{self.monitor_window_name_suffix}/"
                                   f"{split}set-CentralizedEvaluation"),
                            xlabel="Communication Rounds",
                            ylabel="Accuracy",
                            legend=[self.algorithm_name],
                        ),
                    )
                elif self.args.common.monitor == "tensorboard":
                    self.tensorboard.add_scalar(
                        f"Accuracy-{self.monitor_window_name_suffix}/"
                        f"{split}set-CentralizedEvaluation",
                        self.test_results[self.current_epoch + 1]["centralized"][
                            "after"
                        ][split].accuracy,
                        self.current_epoch + 1,
                        new_style=True,
                    )

    def show_max_metrics(self):
        """Displays the maximum metrics (accuracy, etc.) that method obtains."""
        self.logger.log("=" * 20, self.algorithm_name, "Max Accuracy", "=" * 20)
        colors = {
            "before": "blue",
            "after": "red",
            "train": "yellow",
            "val": "green",
            "test": "cyan",
        }

        def _print(groups):
            for group in groups:
                epoches = [
                    E
                    for E, results in self.test_results.items()
                    if group in results.keys()
                ]
                if len(epoches) > 0:
                    self.logger.log(f"{group}:")
                    for stage in ["before", "after"]:
                        for split, flag in [
                            ("train", self.args.common.test.client.train or self.args.common.test.server.train),
                            ("val",   self.args.common.test.client.val   or self.args.common.test.server.val),
                            ("test",  self.args.common.test.client.test  or self.args.common.test.server.test),
                        ]:
                            if flag:
                                # collect that stage & split's accuracies across all relevant epochs
                                metrics_list = [
                                    (epoch, self.test_results[epoch][group][stage][split])
                                    for epoch in epoches
                                ]
                                if metrics_list:
                                    epoch, best_acc = max(
                                        [(ep, mt.accuracy) for (ep, mt) in metrics_list],
                                        key=lambda x: x[1],
                                    )
                                    self.logger.log(
                                        f"[{colors[split]}]({split})[/{colors[split]}] "
                                        f"[{colors[stage]}]{stage}[/{colors[stage]}] "
                                        f"fine-tuning: {best_acc:.2f}% at epoch {epoch}"
                                    )

        if self.train_clients == self.val_clients == self.test_clients:
            _print(["all_clients"])
        else:
            _print(["val_clients", "test_clients"])
        if self.args.common.test.server.interval > 0:
            _print(["centralized"])

    def save_model_weights(self):
        """Optional method to save final global model if not personalized."""
        model_name = f"{self.args.dataset.name}_{self.args.common.global_epoch}_{self.args.model}.pt"
        if not self.all_model_params_personalized:
            torch.save(self.public_model_params, self.output_dir / model_name)
        else:
            self.logger.warn(
                f"{self.algorithm_name}'s all_model_params_personalized=True => skipping save."
            )

    def save_learning_curve_plot(self):
        """Plot and save learning curves from the CSV files if available."""
        import matplotlib.pyplot as plt
        import pandas as pd

        # Define the paths to the CSV files
        avg_loss_csv = self.output_dir / "avg_loss.csv"
        avg_acc_csv = self.output_dir / "avg_accuracy.csv"

        # Check if the CSV files exist
        if not avg_loss_csv.exists() or not avg_acc_csv.exists():
            self.logger.log("CSV files not found. Ensure that 'save_losses' was executed before plotting.")
            return

        # Read the CSV files into DataFrames
        df_loss = pd.read_csv(avg_loss_csv)
        df_acc = pd.read_csv(avg_acc_csv)

        # Figure 1: Average Loss (Before Fine-tuning)
        plt.figure()
        plt.plot(df_loss["epoch"], df_loss["train before"], label="train loss")
        plt.plot(df_loss["epoch"], df_loss["test before"], label="test loss")
        plt.plot(df_loss["epoch"], df_loss["val before"], label="val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Average Loss Before")
        plt.legend()
        plt.savefig(self.output_dir / "avg_loss_before.png", bbox_inches="tight")
        plt.close()

        # Figure 2: Average Accuracy (Before Fine-tuning)
        plt.figure()
        plt.plot(df_acc["epoch"], df_acc["train before"], label="train acc")
        plt.plot(df_acc["epoch"], df_acc["test before"], label="test acc")
        plt.plot(df_acc["epoch"], df_acc["val before"], label="val ACC")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Average Accuracy Before")
        plt.legend()
        plt.savefig(self.output_dir / "avg_acc_before.png", bbox_inches="tight")
        plt.close()

        # Figure 3: Average Loss (After Fine-tuning)
        plt.figure()
        plt.plot(df_loss["epoch"], df_loss["train after"], label="train loss")
        plt.plot(df_loss["epoch"], df_loss["test after"], label="test loss")
        plt.plot(df_loss["epoch"], df_loss["val after"], label="val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Average Loss After")
        plt.legend()
        plt.savefig(self.output_dir / "avg_loss_after.png", bbox_inches="tight")
        plt.close()

        # Figure 4: Average Accuracy (After Fine-tuning)
        plt.figure()
        plt.plot(df_acc["epoch"], df_acc["train after"], label="train acc")
        plt.plot(df_acc["epoch"], df_acc["test after"], label="test acc")
        plt.plot(df_acc["epoch"], df_acc["val after"], label="val ACC")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Average Accuracy After")
        plt.legend()
        plt.savefig(self.output_dir / "avg_acc_after.png", bbox_inches="tight")
        plt.close()


    def save_metrics_stats(self):
        """Store aggregated metrics to CSV if desired."""
        import pandas as pd

        df = pd.DataFrame()
        for stage in ["before", "after"]:
            for split in ["train", "val", "test"]:
                if len(self.aggregated_client_metrics[stage][split]) > 0:
                    for metric in ["accuracy"]:
                        stats = [
                            getattr(m, metric)
                            for m in self.aggregated_client_metrics[stage][split]
                        ]
                        df.insert(
                            loc=df.shape[1],
                            column=f"{metric}_{split}_{stage}",
                            value=np.array(stats).T,
                        )
        df.to_csv(self.output_dir / "metrics.csv", index=True, index_label="epoch")

    def save_losses(self):
        import pandas as pd

        # We will collect data for "all_clients" at each epoch (assuming you do client-side testing).
        # If you prefer "val_clients", "test_clients", or "centralized", just change group_name accordingly.
        group_name = "all_clients"

        epochs = []
        train_before_losses, train_after_losses = [], []
        val_before_losses,   val_after_losses   = [], []
        test_before_losses,  test_after_losses  = [], []

        train_before_accs,   train_after_accs   = [], []
        val_before_accs,     val_after_accs     = [], []
        test_before_accs,    test_after_accs    = [], []

        # Loop over epochs in self.test_results
        for epoch in sorted(self.test_results.keys()):
            # Skip if "all_clients" not in this epoch's results
            if group_name not in self.test_results[epoch]:
                continue

            # Get the dictionary for "all_clients" at this epoch
            group_data = self.test_results[epoch][group_name]

            # group_data has the structure:
            # {
            #   "before": {"train": Metrics, "val": Metrics, "test": Metrics},
            #   "after":  {"train": Metrics, "val": Metrics, "test": Metrics}
            # }
            before_train = group_data["before"]["train"]
            after_train  = group_data["after"]["train"]
            before_val   = group_data["before"]["val"]
            after_val    = group_data["after"]["val"]
            before_test  = group_data["before"]["test"]
            after_test   = group_data["after"]["test"]

            epochs.append(epoch)

            # ---- Losses ----
            train_before_losses.append(before_train.loss)
            train_after_losses.append(after_train.loss)
            val_before_losses.append(before_val.loss)
            val_after_losses.append(after_val.loss)
            test_before_losses.append(before_test.loss)
            test_after_losses.append(after_test.loss)

            # ---- Accuracies ----
            train_before_accs.append(before_train.accuracy)
            train_after_accs.append(after_train.accuracy)
            val_before_accs.append(before_val.accuracy)
            val_after_accs.append(after_val.accuracy)
            test_before_accs.append(before_test.accuracy)
            test_after_accs.append(after_test.accuracy)

        # Create DataFrame for losses
        df_loss = pd.DataFrame({
            "epoch": epochs,
            "train before": train_before_losses,
            "train after":  train_after_losses,
            "val before":   val_before_losses,
            "val after":    val_after_losses,
            "test before":  test_before_losses,
            "test after":   test_after_losses
        })

        # Create DataFrame for accuracies
        df_acc = pd.DataFrame({
            "epoch": epochs,
            "train before": train_before_accs,
            "train after":  train_after_accs,
            "val before":   val_before_accs,
            "val after":    val_after_accs,
            "test before":  test_before_accs,
            "test after":   test_after_accs
        })

        # Save them as two separate CSV files
        df_loss.to_csv(self.output_dir / "avg_loss.csv", index=False)
        df_acc.to_csv(self.output_dir / "avg_accuracy.csv", index=False)



    def run_experiment(self):
        """Entrypoint for the FL-bench experiment."""
        self.logger.log("=" * 20, self.algorithm_name, "=" * 20)
        self.logger.log("Experiment Arguments:")
        rich_pprint(
            OmegaConf.to_object(self.args), console=self.logger.stdout, expand_all=True
        )
        if self.args.common.save_log:
            rich_pprint(
                OmegaConf.to_object(self.args),
                console=self.logger.logfile_logger,
                expand_all=True,
            )
        if self.args.common.monitor == "tensorboard":
            self.tensorboard.add_text(
                f"ExperimentalArguments-{self.monitor_window_name_suffix}",
                json.dumps(OmegaConf.to_object(self.args), indent=4),
            )

        begin = time.time()
        try:
            self.train()
        except KeyboardInterrupt:
            self.logger.close()
            del self.train_progress_bar
            if self.args.common.delete_useless_run:
                if os.path.isdir(self.output_dir):
                    shutil.rmtree(self.output_dir)
                return
        except Exception as e:
            self.logger.log(traceback.format_exc())
            self.logger.log(f"Exception occurred: {e}")
            self.logger.close()
            del self.train_progress_bar
            raise

        end = time.time()
        total = end - begin
        self.logger.log(
            f"{self.algorithm_name}'s total running time: "
            f"{int(total // 3600)} h {int((total % 3600) // 60)} m {int(total % 60)} s."
        )
        self.logger.log("=" * 20, self.algorithm_name, "Experiment Results:", "=" * 20)
        self.logger.log(
            "[green]Display format: (before local fine-tuning) -> (after local fine-tuning)\n",
            "So if finetune_epoch = 0, x.xx% -> 0.00% is normal.\n",
            "Centralized testing ONLY happens after model aggregation, so the stats between '->' are the same.",
        )
        all_test_results = {
            epoch: {
                group: {
                    split: {
                        "loss": f"[red]{metrics['before'][split].loss:.4f} -> {metrics['after'][split].loss:.4f}[/red]",
                        "accuracy": f"[blue]{metrics['before'][split].accuracy:.2f}% -> {metrics['after'][split].accuracy:.2f}%[/blue]",
                    }
                    for split, flag in [
                        (
                            "train",
                            self.args.common.test.client.train
                            or self.args.common.test.server.train,
                        ),
                        (
                            "val",
                            self.args.common.test.client.val
                            or self.args.common.test.server.val,
                        ),
                        (
                            "test",
                            self.args.common.test.client.test
                            or self.args.common.test.server.test,
                        ),
                    ]
                    if flag
                }
                for group, metrics in results.items()
            }
            for epoch, results in self.test_results.items()
        }
        self.logger.log(json.dumps(all_test_results, indent=4))
        if self.args.common.monitor == "tensorboard":
            for epoch, results in all_test_results.items():
                self.tensorboard.add_text(
                    f"Results-{self.monitor_window_name_suffix}",
                    text_string=f"<pre>{results}</pre>",
                    global_step=epoch,
                )

        self.show_max_metrics()
        self.logger.close()

        # optional: plot the training curves
        if self.args.common.save_learning_curve_plot:
            self.save_learning_curve_plot()

        # optional: save aggregated metrics
        if self.args.common.save_metrics:
            self.save_metrics_stats()

        # optional: save final global model
        if self.args.common.save_model:
            self.save_model_weights()
