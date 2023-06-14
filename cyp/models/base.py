import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, namedtuple
from tqdm import tqdm
from datetime import datetime

from .gp import GaussianProcess
from .loss import l1_l2_loss


class ModelBase:
    """
    Base class for all models
    """

    def __init__(
        self,
        model,
        model_weight,
        model_bias,
        model_type,
        savedir,
        use_gp=True,
        sigma=1,
        r_loc=0.5,
        r_year=1.5,
        sigma_e=0.32,
        sigma_b=0.01,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        hyperparameters_df=None, # [CS4245]
    ):
        self.savedir = savedir / model_type
        self.savedir.mkdir(parents=True, exist_ok=True)

        print(f"Using {device.type}")
        if device.type != "cpu":
            model = model.cuda()
        self.model = model
        self.model_type = model_type
        self.model_weight = model_weight
        self.model_bias = model_bias

        self.device = device
        self.hyperparameters_df = hyperparameters_df # [CS4245]

        # for reproducability
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.gp = None
        if use_gp:
            self.gp = GaussianProcess(sigma, r_loc, r_year, sigma_e, sigma_b)

    def run(
        self,
        path_to_histogram=Path("data/img_output/histogram_all_full.npz"),
        times="all",
        pred_years=None,
        num_runs=2,
        train_steps=25000,
        batch_size=32,
        starter_learning_rate=1e-3,
        weight_decay=0,
        l1_weight=0,
        patience=10,
        with_italy_validation=False # [CS4245]
    ):
        """
        Train the models. Note that multiple models are trained: as per the paper, a model
        is trained for each year, with all preceding years used as training values. In addition,
        for each year, 2 models are trained to account for random initialization.

        Parameters
        ----------
        path_to_histogram: pathlib Path, default=Path('data/img_output/histogram_all_full.npz')
            The location of the training data
        times: {'all', 'realtime'}
            Which time indices to train the model on. If 'all', a full run (32 timesteps) is used.
            If 'realtime', range(10, 31, 4) is used.
        pred_years: int, list or None, default=None
            Which years to build models for. If None, the default values from the paper (range(2009, 2016))
            are used.
        num_runs: int, default=2
            The number of runs to do per year. Default taken from the paper
        train_steps: int, default=25000
            The number of steps for which to train the model. Default taken from the paper.
        batch_size: int, default=32
            Batch size when training. Default taken from the paper
        starter_learning_rate: float, default=1e-3
            Starter learning rate. Note that the learning rate is divided by 10 after 2000 and 4000 training
            steps. Default taken from the paper
        weight_decay: float, default=1
            Weight decay (L2 regularization) on the model weights
        l1_weight: float, default=0
            In addition to MSE, L1 loss is also used (sometimes). This is the weight to assign to this L1 loss.
        patience: int or None, default=10
            The number of epochs to wait without improvement in the validation loss before terminating training.
            Note that the original repository doesn't use early stopping.
        with_italy_validation: bool, default=False
            The boolean indicating whether to include Italy in the validation set.
        """

        with np.load(path_to_histogram) as hist:
            images = hist["output_image"]
            locations = hist["output_locations"]
            yields = hist["output_yield"]
            years = hist["output_year"]
            indices = hist["output_index"]

        # to collect results
        years_list, run_numbers, rmse_list, me_list, times_list = [], [], [], [], []
        if self.gp is not None:
            rmse_gp_list, me_gp_list = [], []

        if pred_years is None:
            pred_years = range(2009, 2016)
        elif type(pred_years) is int:
            pred_years = [pred_years]

        if times == "all":
            times = [32]
        else:
            times = range(10, 31, 4)

        ### [CS4245] ###
        training_times_list = []
        epochs_list = []  
        ################

        for pred_year in pred_years:
            for run_number in range(1, num_runs + 1):
                for time in times:
                    print(
                        f"Training to predict on {pred_year}, Run number {run_number}"
                    )
                    
                    results, training_time, num_epoch_run = self._run_1_year( # [CS4245]
                        images,
                        yields,
                        years,
                        locations,
                        indices,
                        pred_year,
                        time,
                        run_number,
                        train_steps,
                        batch_size,
                        starter_learning_rate,
                        weight_decay,
                        l1_weight,
                        patience,
                        with_italy_validation
                    )

                    years_list.append(pred_year)
                    run_numbers.append(run_number)
                    times_list.append(time)
                    ### [CS4245] ###
                    training_times_list.append(training_time) 
                    epochs_list.append(num_epoch_run)
                    ################

                    if self.gp is not None:
                        rmse, me, rmse_gp, me_gp = results
                        rmse_gp_list.append(rmse_gp)
                        me_gp_list.append(me_gp)
                    else:
                        rmse, me = results
                    rmse_list.append(rmse)
                    me_list.append(me)
                print("-----------")

        # save results to a csv file
        data = {
            "year": years_list,
            "run_number": run_numbers,
            "time_idx": times_list,
            "RMSE": rmse_list,
            "ME": me_list,
        }


        if self.gp is not None:
            data["RMSE_GP"] = rmse_gp_list
            data["ME_GP"] = me_gp_list
        results_df = pd.DataFrame(data=data)

        ### [CS4245] ###
        # add training time and number of epochs to the results dataframe
        results_df["training_time (s)"] = training_times_list
        results_df["num_epochs"] = epochs_list


        # add hyperparameters to the results dataframe
        # F:
        if self.hyperparameters_df is not None:
            results_df = pd.concat([results_df, self.hyperparameters_df], axis=1)

        
        # F:
        average_values = {
            "RMSE_avg": results_df["RMSE"].mean(),
            "ME_avg": results_df["ME"].mean()
        }

        if self.gp is not None:
          average_values["RMSE_GP_avg"] = results_df["RMSE_GP"].mean(),
          average_values["ME_GP_avg"] = results_df["ME_GP"].mean(),
        
        results_df = results_df.append(average_values, ignore_index=True)

        


        # F:
        # write hyperparameters in the title and also time stamp
        """
        title = "_".join(
            [
                f"{key}={val}"
                for key, val in self.hyperparameters_df.iloc[0].to_dict().items()
            ]
        )
        """
        ### [CS4245] ###
        
        title = ""
        if self.gp is not None:
          # add the RMSE_GP_avg at the beginning of the title : "RMSE_GP_avg=0.123_..."
          title = f"RMSE_GP_avg={average_values['RMSE_GP_avg']}_" + title
        else:
          title = f"RMSE_avg={average_values['RMSE_avg']}_" + title
        

        results_df.to_csv(self.savedir / f"{title}_{datetime.now()}.csv", index=False)
        
        # return RMSE_GP_avg loss 
        if self.gp is not None:
          return average_values["RMSE_GP_avg"]
        else:
          return average_values["RMSE_avg"]
        
        ###############
    def _run_1_year(
        self,
        images,
        yields,
        years,
        locations,
        indices,
        predict_year,
        time,
        run_number,
        train_steps,
        batch_size,
        starter_learning_rate,
        weight_decay,
        l1_weight,
        patience,
        with_italy_validation
    ):
        """
        Train one model on one year of data, and then save the model predictions.
        To be called by run().
        """
        train_data, test_data = self.prepare_arrays(
            images, yields, locations, indices, years, predict_year, time
        )

        ### [CS4245] ###
        it_data = []
        if with_italy_validation:
            with np.load(Path("data_italy/img_output") / "histogram_all_full.npz") as hist:
                it_images = hist["output_image"]
                it_yields = hist["output_yield"]
                it_years = hist["output_year"]
                it_indices = hist["output_index"]

            it_data = self.prepare_italy_array(it_images, it_yields, it_indices, it_years, predict_year, time, images, years)

            print("Italy data prepared")
        ################

        # reinitialize the model, since self.model may be trained multiple
        # times in one call to run()
        self.reinitialize_model(time=time)

        
        train_scores, val_scores, epoch_train_scores, epoch_val_scores, training_time, num_epochs = self._train( # [CS4245]
            train_data.images,
            train_data.yields,
            train_steps,
            batch_size,
            starter_learning_rate,
            weight_decay,
            l1_weight,
            patience,
        )

        results = self._predict_it(*train_data, *test_data, *it_data, batch_size) if with_italy_validation else self._predict(*train_data, *test_data, batch_size) # [CS4245]

        model_information = {
            "state_dict": self.model.state_dict(),
            "val_loss": val_scores["loss"],
            "train_loss": train_scores["loss"],
            "epoch_train_loss": epoch_train_scores, # [CS4245] 
            "epoch_val_loss": epoch_val_scores, # [CS4245]
        }
        for key in results:
            model_information[key] = results[key]

        # finally, get the relevant weights for the Gaussian Process
        model_weight = self.model.state_dict()[self.model_weight]
        model_bias = self.model.state_dict()[self.model_bias]

        if self.model.state_dict()[self.model_weight].device != "cpu":
            model_weight, model_bias = model_weight.cpu(), model_bias.cpu()

        model_information["model_weight"] = model_weight.numpy()
        model_information["model_bias"] = model_bias.numpy()

        model_information["training_time"] = training_time # [CS4245]
        model_information["num_epochs"] = num_epochs # [CS4245]
        
        if self.gp is not None:
            print("Running Gaussian Process!")
            gp_pred = self.gp.run(
                model_information["train_feat"],
                model_information["test_feat"],
                model_information["train_loc"],
                model_information["test_loc"],
                model_information["train_years"],
                model_information["test_years"],
                model_information["train_real"],
                model_information["model_weight"],
                model_information["model_bias"],
            )
            model_information["test_pred_gp"] = gp_pred.squeeze(1)

        filename = f'{predict_year}_{run_number}_{time}_{"gp" if (self.gp is not None) else ""}.pth.tar'
        torch.save(model_information, self.savedir / filename)
        return self.analyze_results(
            model_information["test_real"],
            model_information["test_pred"],
            model_information["test_pred_gp"] if self.gp is not None else None,
        ), training_time, num_epochs

    def _train(
        self,
        train_images,
        train_yields,
        train_steps,
        batch_size,
        starter_learning_rate,
        weight_decay,
        l1_weight,
        patience,
    ):
        """Defines the training loop for a model"""

        # split the training dataset into a training and validation set
        total_size = train_images.shape[0]
        # "Learning rates and stopping criteria are tuned on a held-out
        # validation set (10%)."
        val_size = total_size // 10
        train_size = total_size - val_size
        print(
            f"After split, training on {train_size} examples, "
            f"validating on {val_size} examples"
        )
        train_dataset, val_dataset = random_split(
            TensorDataset(train_images, train_yields), (train_size, val_size)
        )

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = torch.optim.Adam(
            [pam for pam in self.model.parameters()],
            lr=starter_learning_rate,
            weight_decay=weight_decay,
        )

        num_epochs = int(train_steps / (train_images.shape[0] / batch_size))
        print(f"Training for {num_epochs} epochs")

        train_scores = defaultdict(list)
        val_scores = defaultdict(list)

        ### [CS4245] ###
        epoch_train_scores = []
        epoch_val_scores = []
        ################

        step_number = 0
        min_loss = np.inf
        best_state = self.model.state_dict()

        if patience is not None:
            epochs_without_improvement = 0

        ### [CS4245] ###
        # Compute training time
        start_time = datetime.now()
        ################

        for epoch in range(num_epochs):
            self.model.train()

            # running train and val scores are only for printing out
            # information
            running_train_scores = defaultdict(list)

            for train_x, train_y in tqdm(train_dataloader):
                optimizer.zero_grad()
                pred_y = self.model(train_x)

                loss, running_train_scores = l1_l2_loss(
                    pred_y, train_y, l1_weight, running_train_scores
                )
                loss.backward()
                optimizer.step()

                train_scores["loss"].append(loss.item())

                step_number += 1

                if step_number in [4000, 20000]:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] /= 10
            ### [CS4245] ###
            avg_train_loss = np.mean(train_scores["loss"][-len(train_dataloader):])
            epoch_train_scores.append(avg_train_loss)
            ################
            train_output_strings = []
            for key, val in running_train_scores.items():
                train_output_strings.append(
                    "{}: {}".format(key, round(np.array(val).mean(), 5))
                )

            running_val_scores = defaultdict(list)
            self.model.eval()
            with torch.no_grad():
                for (
                    val_x,
                    val_y,
                ) in tqdm(val_dataloader):
                    val_pred_y = self.model(val_x)

                    val_loss, running_val_scores = l1_l2_loss(
                        val_pred_y, val_y, l1_weight, running_val_scores
                    )

                    val_scores["loss"].append(val_loss.item())
            ### [CS4245] ###
            avg_val_loss = np.mean(val_scores["loss"][-len(val_dataloader):])
            epoch_val_scores.append(avg_val_loss)
            ################

            val_output_strings = []
            for key, val in running_val_scores.items():
                val_output_strings.append(
                    "{}: {}".format(key, round(np.array(val).mean(), 5))
                )

            print("TRAINING: {}".format(", ".join(train_output_strings)))
            print("VALIDATION: {}".format(", ".join(val_output_strings)))

            epoch_val_loss = np.array(running_val_scores["loss"]).mean()

            if epoch_val_loss < min_loss:
                best_state = self.model.state_dict()
                min_loss = epoch_val_loss

                if patience is not None:
                    epochs_without_improvement = 0
            elif patience is not None:
                epochs_without_improvement += 1

                if epochs_without_improvement == patience:
                    # revert to the best state dict
                    self.model.load_state_dict(best_state)
                    print("Early stopping!")
                    break
        
        ### [CS4245] ###
        # Find how many epochs were run
        num_epochs = len(epoch_train_scores)

        # F:
        # Compute training time
        end_time = datetime.now()
        training_time = end_time - start_time
        training_time = training_time.seconds
        ################

        self.model.load_state_dict(best_state)
        return train_scores, val_scores, epoch_train_scores, epoch_val_scores, training_time, num_epochs # [CS4245]

    def _predict(
        self,
        train_images,
        train_yields,
        train_locations,
        train_indices,
        train_years,
        test_images,
        test_yields,
        test_locations,
        test_indices,
        test_years,
        batch_size,
    ):
        """
        Predict on the training and validation data. Optionally, return the last
        feature vector of the model.
        """
        train_dataset = TensorDataset(
            train_images, train_yields, train_locations, train_indices, train_years
        )

        test_dataset = TensorDataset(
            test_images, test_yields, test_locations, test_indices, test_years
        )

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        results = defaultdict(list)

        self.model.eval()
        with torch.no_grad():
            for train_im, train_yield, train_loc, train_idx, train_year in tqdm(
                train_dataloader
            ):
                model_output = self.model(
                    train_im, return_last_dense=True if (self.gp is not None) else False
                )
                if self.gp is not None:
                    pred, feat = model_output
                    if feat.device != "cpu":
                        feat = feat.cpu()
                    results["train_feat"].append(feat.numpy())
                else:
                    pred = model_output
                results["train_pred"].extend(pred.squeeze(1).tolist())
                results["train_real"].extend(train_yield.squeeze(1).tolist())
                results["train_loc"].append(train_loc.numpy())
                results["train_indices"].append(train_idx.numpy())
                results["train_years"].extend(train_year.tolist())

            for test_im, test_yield, test_loc, test_idx, test_year in tqdm(
                test_dataloader
            ):
                model_output = self.model(
                    test_im, return_last_dense=True if (self.gp is not None) else False
                )
                if self.gp is not None:
                    pred, feat = model_output
                    if feat.device != "cpu":
                        feat = feat.cpu()
                    results["test_feat"].append(feat.numpy())
                else:
                    pred = model_output
                results["test_pred"].extend(pred.squeeze(1).tolist())
                results["test_real"].extend(test_yield.squeeze(1).tolist())
                results["test_loc"].append(test_loc.numpy())
                results["test_indices"].append(test_idx.numpy())
                results["test_years"].extend(test_year.tolist())

        for key in results:
            if key in [
                "train_feat",
                "test_feat",
                "train_loc",
                "test_loc",
                "train_indices",
                "test_indices",
            ]:
                results[key] = np.concatenate(results[key], axis=0)
            else:
                results[key] = np.array(results[key])
        return results
    
    # CS4245: 
    # Added a separate function for making regular predictions together with predictions for Italy
    # (as part of the validation process)
    def _predict_it(
        self,
        train_images,
        train_yields,
        train_locations,
        train_indices,
        train_years,
        test_images,
        test_yields,
        test_locations,
        test_indices,
        test_years,
        it_images,
        it_yields,
        it_indices,
        it_years,
        batch_size,
    ):
        """
        Predict on the training and validation data. Optionally, return the last
        feature vector of the model.
        """
        train_dataset = TensorDataset(
            train_images, train_yields, train_locations, train_indices, train_years
        )

        test_dataset = TensorDataset(
            test_images, test_yields, test_locations, test_indices, test_years
        )

        italy_validation_dataset = TensorDataset(
            it_images, it_yields, it_indices, it_years
        )

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
        italy_validation_dataset = DataLoader(italy_validation_dataset, batch_size=batch_size)

        results = defaultdict(list)

        self.model.eval()
        with torch.no_grad():
            for train_im, train_yield, train_loc, train_idx, train_year in tqdm(
                train_dataloader
            ):
                model_output = self.model(
                    train_im, return_last_dense=True if (self.gp is not None) else False
                )
                if self.gp is not None:
                    pred, feat = model_output
                    if feat.device != "cpu":
                        feat = feat.cpu()
                    results["train_feat"].append(feat.numpy())
                else:
                    pred = model_output
                results["train_pred"].extend(pred.squeeze(1).tolist())
                results["train_real"].extend(train_yield.squeeze(1).tolist())
                results["train_loc"].append(train_loc.numpy())
                results["train_indices"].append(train_idx.numpy())
                results["train_years"].extend(train_year.tolist())

            for test_im, test_yield, test_loc, test_idx, test_year in tqdm(
                test_dataloader
            ):
                model_output = self.model(
                    test_im, return_last_dense=True if (self.gp is not None) else False
                )
                if self.gp is not None:
                    pred, feat = model_output
                    if feat.device != "cpu":
                        feat = feat.cpu()
                    results["test_feat"].append(feat.numpy())
                else:
                    pred = model_output
                results["test_pred"].extend(pred.squeeze(1).tolist())
                results["test_real"].extend(test_yield.squeeze(1).tolist())
                results["test_loc"].append(test_loc.numpy())
                results["test_indices"].append(test_idx.numpy())
                results["test_years"].extend(test_year.tolist())

            for it_im, it_yield, it_idx, it_year in tqdm(
                italy_validation_dataset
            ):
                model_output = self.model(
                    it_im, return_last_dense=True if (self.gp is not None) else False
                )
                if self.gp is not None:
                    pred, feat = model_output
                    if feat.device != "cpu":
                        feat = feat.cpu()
                    results["it_feat"].append(feat.numpy())
                else:
                    pred = model_output
                results["it_pred"].extend(pred.squeeze(1).tolist())
                results["it_real"].extend(it_yield.squeeze(1).tolist())
                results["it_indices"].append(it_idx.numpy())
                results["it_years"].extend(it_year.tolist())

            print("Finished validating italy")

        for key in results:
            if key in [
                "train_feat",
                "test_feat",
                "it_feat",
                "train_loc",
                "test_loc",
                "train_indices",
                "test_indices",
                "it_indices",
            ]:
                results[key] = np.concatenate(results[key], axis=0)
            else:
                results[key] = np.array(results[key])

        return results

    def prepare_arrays(
        self, images, yields, locations, indices, years, predict_year, time
    ):
        """Prepares the inputs for the model, in the following way:
        - normalizes the images
        - splits into a train and val set
        - turns the numpy arrays into tensors
        - removes excess months, if monthly predictions are being made
        """
        train_idx = np.nonzero(years < predict_year)[0]
        test_idx = np.nonzero(years == predict_year)[0]

        train_images, test_images = self._normalize(images[train_idx], images[test_idx])

        print(
            f"Train set size: {train_idx.shape[0]}, Test set size: {test_idx.shape[0]}"
        )

        Data = namedtuple("Data", ["images", "yields", "locations", "indices", "years"])

        train_data = Data(
            images=torch.as_tensor(
                train_images[:, :, :time, :], device=self.device
            ).float(),
            yields=torch.as_tensor(yields[train_idx], device=self.device)
            .float()
            .unsqueeze(1),
            locations=torch.as_tensor(locations[train_idx]),
            indices=torch.as_tensor(indices[train_idx]),
            years=torch.as_tensor(years[train_idx]),
        )

        test_data = Data(
            images=torch.as_tensor(
                test_images[:, :, :time, :], device=self.device
            ).float(),
            yields=torch.as_tensor(yields[test_idx], device=self.device)
            .float()
            .unsqueeze(1),
            locations=torch.as_tensor(locations[test_idx]),
            indices=torch.as_tensor(indices[test_idx]),
            years=torch.as_tensor(years[test_idx]),
        )

        return train_data, test_data

    # CS4245: Added a separate function for preparing the Italy data array
    def prepare_italy_array(
        self, images, yields, indices, years, predict_year, time, train_images, train_years
    ):
        """Prepares the inputs for the model, in the following way:
        - normalizes the images
        - splits into a train and val set
        - turns the numpy arrays into tensors
        - removes excess months, if monthly predictions are being made
        """
        train_idx = np.nonzero(train_years < predict_year)[0]
        idx = np.nonzero(years == predict_year)[0]

        mean = np.mean(train_images[train_idx], axis=(0, 2, 3))
        year_images = images[idx]
        year_images = (year_images.transpose(0, 2, 3, 1) - mean).transpose(0, 3, 1, 2)

        Data = namedtuple("Data", ["images", "yields", "indices", "years"])

        data = Data(
            images=torch.as_tensor(
                year_images[:, :, :time, :], device=self.device
            ).float(),
            yields=torch.as_tensor(yields[idx], device=self.device)
            .float()
            .unsqueeze(1),
            indices=torch.as_tensor(indices[idx]),
            years=torch.as_tensor(years[idx]),
        )

        return data

    @staticmethod
    def _normalize(train_images, val_images):
        """
        Find the mean values of the bands in the train images. Use these values
        to normalize both the training and validation images.

        A little awkward, since transpositions are necessary to make array broadcasting work
        """
        mean = np.mean(train_images, axis=(0, 2, 3))

        train_images = (train_images.transpose(0, 2, 3, 1) - mean).transpose(0, 3, 1, 2)
        val_images = (val_images.transpose(0, 2, 3, 1) - mean).transpose(0, 3, 1, 2)

        return train_images, val_images

    @staticmethod
    def analyze_results(true, pred, pred_gp):
        """Calculate ME and RMSE"""
        rmse = np.sqrt(np.mean((true - pred) ** 2))
        me = np.mean(true - pred)

        print(f"Without GP: RMSE: {rmse}, ME: {me}")

        if pred_gp is not None:
            rmse_gp = np.sqrt(np.mean((true - pred_gp) ** 2))
            me_gp = np.mean(true - pred_gp)
            print(f"With GP: RMSE: {rmse_gp}, ME: {me_gp}")
            return rmse, me, rmse_gp, me_gp
        return rmse, me

    def reinitialize_model(self, time=None):
        raise NotImplementedError
