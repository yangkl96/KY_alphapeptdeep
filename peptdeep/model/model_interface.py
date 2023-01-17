# AUTOGENERATED! DO NOT EDIT! File to edit: nbdev_nbs/model/model_interface.ipynb (unless otherwise specified).

__all__ = ['get_cosine_schedule_with_warmup', 'append_nAA_column_if_missing', 'ModelInterface']

# Cell
import os
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

import torch.multiprocessing as mp
import functools

from zipfile import ZipFile
from typing import IO, Tuple, List, Union
from alphabase.yaml_utils import save_yaml, load_yaml
from alphabase.peptide.precursor import is_precursor_sorted

from peptdeep.settings import model_const
from peptdeep.utils import logging
from peptdeep.settings import global_settings

from peptdeep.model.featurize import (
    get_ascii_indices, get_batch_aa_indices,
    get_batch_mod_feature
)


# Cell
from torch.optim.lr_scheduler import LambdaLR

# copied from huggingface
def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps,
    num_training_steps, num_cycles=0.5,
    last_epoch=-1
):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(
            current_step - num_warmup_steps
        ) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (
            1.0 + np.cos(np.pi * num_cycles * 2.0 * progress)
        ))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def append_nAA_column_if_missing(precursor_df):
    """
    column containing the number of Amino Acids
    """
    if 'nAA' not in precursor_df.columns:
        precursor_df['nAA'] = precursor_df.sequence.str.len()
        precursor_df.sort_values('nAA', inplace=True)
        precursor_df.reset_index(drop=True,inplace=True)
    return precursor_df

# Cell
class ModelInterface(object):
    """
    Provides standardized methods to interact
    with ml models. Inherit into new class and override
    the abstract (i.e. not implemented) methods.
    """
    def __init__(self,
        device:str='gpu',
        **kwargs
    ):
        self.model:torch.nn.Module = None
        self.optimizer = None
        self.model_params:dict = {}
        self.set_device(device)

        self._min_pred_value = 0.0

    @property
    def target_column_to_predict(self)->str:
        return self._target_column_to_predict

    @target_column_to_predict.setter
    def target_column_to_predict(self, column:str):
        self._target_column_to_predict = column

    @property
    def target_column_to_train(self)->str:
        return self._target_column_to_train

    @target_column_to_train.setter
    def target_column_to_train(self, column:str):
        self._target_column_to_train = column

    def set_device(self, device_type = 'gpu'):
        """
        Sets the device (e.g. gpu (cuda), cpu) to be used in the model.
        """
        self.device_type = device_type.lower().replace('gpu','cuda')
        if not torch.cuda.is_available():
            self.device_type = 'cpu'
        self.device = torch.device(self.device_type)
        if self.model is not None:
            self.model.to(self.device)

    def build(self,
        model_class: torch.nn.Module,
        **kwargs
    ):
        """
        Builds the model by specifying the PyTorch module,
        the parameters, the device, the loss function ...
        """
        self.model = model_class(**kwargs)
        self.model_params = kwargs
        self.model.to(self.device)
        self._init_for_training()

    def train_with_warmup(self,
        precursor_df: pd.DataFrame,
        *,
        batch_size=1024,
        epoch=10,
        warmup_epoch=5,
        lr=1e-4,
        verbose=False,
        verbose_each_epoch=False,
        **kwargs
    ):
        """
        Trains the model according to specifications. Includes a warumup
        phase with linear rate scheduling (cosine schedule).
        """
        self._prepare_training(precursor_df, lr, **kwargs)

        lr_scheduler = self._get_lr_schedule_with_warmup(
            warmup_epoch, epoch
        )

        #split into train-val split
        train_df = precursor_df.sample(frac = 0.9)
        val_df = precursor_df.drop(train_df.index)
        min_val_loss = sys.maxsize
        best_epoch = 1
        state_dict = {}
        for epoch in range(epoch):
            self.model.train()
            batch_cost = self._train_one_epoch(False,
                train_df, epoch,
                batch_size, verbose_each_epoch,
                **kwargs
            )
            lr_scheduler.step()

            #validation set
            self.model.eval()
            val_batch_cost = self._train_one_epoch(True,
                                               val_df, epoch,
                                               batch_size, verbose_each_epoch,
                                               **kwargs
                                               )

            if verbose:
                print(f'[Training] Epoch={epoch+1}, lr={lr_scheduler.get_last_lr()[0]},  train loss={np.mean(batch_cost)},'
                      f'  val loss={np.mean(val_batch_cost)}')
                with open(global_settings['model_mgr']["log_file"], "a") as f:
                    f.write(f'[Training] Epoch={epoch+1}, lr={lr_scheduler.get_last_lr()[0]},  '
                            f'train loss={np.mean(batch_cost)},  val loss={np.mean(val_batch_cost)}\n')

            #save intermediate state_dict
            if np.mean(val_batch_cost) < min_val_loss:
                min_val_loss = np.mean(val_batch_cost)
                state_dict = deepcopy(self.model.state_dict())
                best_epoch = epoch + 1

        torch.cuda.empty_cache()
        print("Best model was from epoch " + str(best_epoch))
        return state_dict

    def train(self,
        precursor_df: pd.DataFrame,
        *,
        batch_size=1024,
        epoch=10,
        warmup_epoch:int=0,
        lr=1e-4,
        verbose=False,
        verbose_each_epoch=False,
        **kwargs
    ):
        """
        Trains the model according to specifications.
        """
        if warmup_epoch > 0:
            state_dict = self.train_with_warmup(
                precursor_df,
                batch_size=batch_size,
                epoch=epoch,
                warmup_epoch=warmup_epoch,
                lr=lr,
                verbose=verbose,
                verbose_each_epoch=verbose_each_epoch,
                **kwargs
            )
            return state_dict
        else:
            self._prepare_training(precursor_df, lr, **kwargs)

            for epoch in range(epoch):
                batch_cost = self._train_one_epoch(
                    precursor_df, epoch,
                    batch_size, verbose_each_epoch,
                    **kwargs
                )
                if verbose: print(f'[Training] Epoch={epoch+1}, Mean Loss={np.mean(batch_cost)}')

            torch.cuda.empty_cache()

    def predict(self,
        precursor_df:pd.DataFrame,
        *,
        batch_size:int=1024,
        verbose:bool=False,
        **kwargs
    )->pd.DataFrame:
        """
        The model predicts the properties based on the inputs it has been trained for.
        Returns the ouput as a pandas dataframe.
        """
        precursor_df = append_nAA_column_if_missing(precursor_df)
        self._check_predict_in_order(precursor_df)
        self._prepare_predict_data_df(precursor_df,**kwargs)
        self.model.eval()

        _grouped = precursor_df.groupby('nAA')
        if verbose:
            batch_tqdm = tqdm(_grouped)
        else:
            batch_tqdm = _grouped
        with torch.no_grad():
            for nAA, df_group in batch_tqdm:
                for i in range(0, len(df_group), batch_size):
                    batch_end = i+batch_size

                    batch_df = df_group.iloc[i:batch_end,:]

                    features = self._get_features_from_batch_df(
                        batch_df, **kwargs
                    )

                    if isinstance(features, tuple):
                        predicts = self._predict_one_batch(*features)
                    else:
                        predicts = self._predict_one_batch(features)

                    self._set_batch_predict_data(
                        batch_df, predicts,
                        **kwargs
                    )

        torch.cuda.empty_cache()
        return self.predict_df

    def predict_mp(self,
        precursor_df:pd.DataFrame,
        *,
        batch_size:int=1024,
        mp_batch_size:int=100000,
        process_num:int=global_settings['thread_num'],
        **kwargs
    )->pd.DataFrame:
        """
        Predicting with multiprocessing is no GPUs are availible.
        Note this multiprocessing method only works for models those predict
        values within (inplace of) the precursor_df.
        """
        precursor_df = append_nAA_column_if_missing(precursor_df)

        if self.device_type != 'cpu':
            return self.predict(
                precursor_df,
                batch_size=batch_size,
                verbose=False,
                **kwargs
            )

        _predict_func = functools.partial(self.predict,
            batch_size=batch_size, verbose=False, **kwargs
        )

        from peptdeep.utils import process_bar

        def batch_df_gen(precursor_df, mp_batch_size):
            for i in range(0, len(precursor_df), mp_batch_size):
                yield precursor_df.iloc[i:i+mp_batch_size]

        self._check_predict_in_order(precursor_df)
        self._prepare_predict_data_df(precursor_df,**kwargs)

        print("Predicting with multiprocessing ...")
        self.model.share_memory()
        df_list = []
        with mp.Pool(process_num) as p:
            for ret_df in process_bar(p.imap(
                    _predict_func,
                    batch_df_gen(precursor_df, mp_batch_size),
                ), len(precursor_df)//mp_batch_size+1
            ):
                df_list.append(ret_df)

        self.predict_df = pd.concat(df_list)
        self.predict_df.reset_index(drop=True, inplace=True)

        return self.predict_df

    def save(self, filename):
        """
        Save the model state, the constants used, the code defining the model and the model parameters.
        """
        # TODO save tf.keras.Model
        dir = os.path.dirname(filename)
        if not dir: dir = './'
        if not os.path.exists(dir): os.makedirs(dir)
        torch.save(self.model.state_dict(), filename)
        with open(filename+'.txt','w') as f: f.write(str(self.model))
        save_yaml(filename+'.model_const.yaml', model_const)
        self._save_codes(filename+'.model.py')
        save_yaml(filename+'.param.yaml', self.model_params)

    def load(
        self,
        model_file: Tuple[str, IO],
        model_path_in_zip: str = None,
        **kwargs
    ):
        """
        Load a model specified in a zip file, a text file or a file stream.
        """
        # TODO load tf.keras.Model
        if isinstance(model_file, str):
            # We may release all models (msms, rt, ccs, ...) in a single zip file
            if model_file.lower().endswith('.zip'):
                self._load_model_from_zipfile(model_file, model_path_in_zip)
            else:
                self._load_model_from_pytorchfile(model_file)
        else:
            self._load_model_from_stream(model_file)

    def get_parameter_num(self):
        """
        Get total number of parameters in model.
        """
        return np.sum([p.numel() for p in self.model.parameters()])

    def build_from_py_codes(self,
        model_code_file_or_zip:str,
        code_file_in_zip:str=None,
        include_model_params_yaml:bool=True,
        **kwargs
    ):
        """
        Build the model based on a python file. Must contain a PyTorch
        model implemented as 'class Model(...'
        """
        if model_code_file_or_zip.lower().endswith('.zip'):
            with ZipFile(model_code_file_or_zip, 'r') as model_zip:
                with model_zip.open(code_file_in_zip,'r') as f:
                    codes = f.read()
                if include_model_params_yaml:
                    with model_zip.open(
                        code_file_in_zip[:-len('model.py')]+'param.yaml',
                        'r'
                    ) as f:
                        params = yaml.load(f, yaml.FullLoader)
        else:
            with open(model_code_file_or_zip, 'r') as f:
                codes = f.read()
            if include_model_params_yaml:
                params = load_yaml(
                    model_code_file_or_zip[:-len('model.py')]+'param.yaml'
                )

        compiled_codes = compile(
            codes,
            filename='model_file_py',
            mode='exec'
        )
        from types import ModuleType
        _module = ModuleType('_apd_nn_codes')
        #codes must contains torch model codes 'class Model(...'
        exec(compiled_codes, _module.__dict__)

        if include_model_params_yaml:
            for key, val in params.items():
                if key not in kwargs:
                    kwargs[key] = val

        self.model = _module.Model(**kwargs)
        self.model_params = kwargs
        self.model.to(self.device)
        self._init_for_training()

    def _init_for_training(self):
        """
        Set the loss function, and more attributes for different tasks.
        The default loss function is nn.L1Loss.
        """
        self.loss_func = torch.nn.L1Loss()

    def _as_tensor(self,
        data:np.ndarray,
        dtype:torch.dtype=torch.float32
    )->torch.Tensor:
        """Convert numerical np.array to pytorch tensor.
        The tensor will be stored in self.device

        Args:
            data (np.array): numerical np.array
            dtype (torch.dtype, optional): dtype. The dtype of the indices
                used for embedding should be `torch.long`.
                Defaults to `torch.float32`.
        Returns:
            torch.Tensor: the tensor stored in self.device
        """
        return torch.tensor(data, dtype=dtype, device=self.device)

    def _load_model_from_zipfile(self, model_file, model_path_in_zip):
        with ZipFile(model_file) as model_zip:
            with model_zip.open(model_path_in_zip,'r') as pt_file:
                self._load_model_from_stream(pt_file)

    def _load_model_from_pytorchfile(self, model_file):
        with open(model_file,'rb') as pt_file:
            self._load_model_from_stream(pt_file)

    def _load_model_from_stream(self, stream):
        #can replace output_nn.nn.2.weight and output_nn.nn.2.bias with new length tensors
        loaded_model_params = torch.load(stream, map_location=self.device)
        if "output_nn.nn.2.bias" in self.model.state_dict(): #only apply to ms2 model
            if loaded_model_params["output_nn.nn.2.bias"].size() != self.model.state_dict()["output_nn.nn.2.bias"].size():
                print("Adjusting output size for fragments")
                loaded_model_params["output_nn.nn.2.bias"] = torch.rand(
                    self.model.state_dict()["output_nn.nn.2.bias"].size())
                loaded_model_params["output_nn.nn.2.weight"] = torch.rand(
                    self.model.state_dict()["output_nn.nn.2.weight"].size())
        #self.model.requires_grad_(True)
        (
            missing_keys, unexpect_keys
        ) = self.model.load_state_dict(loaded_model_params,
            strict=False
        )
        if len(missing_keys) > 0:
            logging.warn(f"nn parameters {missing_keys} are MISSING while loading models in {self.__class__}")
        if len(unexpect_keys) > 0:
            logging.warn(f"nn parameters {unexpect_keys} are UNEXPECTED while loading models in {self.__class__}")

    def _save_codes(self, save_as):
        try:
            import inspect
            code = '''import torch\n'''
            code += '''import peptdeep.model.building_block as building_block\n'''
            code += '''from peptdeep.model.model_shop import *\n'''
            class_code = inspect.getsource(self.model.__class__)
            code += 'class Model' + class_code[class_code.find('('):]
            with open(save_as, 'w') as f:
                f.write(code)
        except (TypeError, ValueError, KeyError) as e:
            logging.info(f'Cannot save model source codes: {str(e)}')

    def _train_one_epoch(self, val,
        precursor_df, epoch, batch_size, verbose_each_epoch,
        **kwargs
    ):
        """Training for an epoch"""
        batch_cost = []
        _grouped = list(precursor_df.sample(frac=1).groupby('nAA'))
        rnd_nAA = np.random.permutation(len(_grouped))
        if verbose_each_epoch:
            batch_tqdm = tqdm(rnd_nAA)
        else:
            batch_tqdm = rnd_nAA
        for i_group in batch_tqdm:
            nAA, df_group = _grouped[i_group]
            # df_group = df_group.reset_index(drop=True)
            for i in range(0, len(df_group), batch_size):
                batch_end = i+batch_size

                batch_df = df_group.iloc[i:batch_end,:]
                targets = self._get_targets_from_batch_df(
                    batch_df, **kwargs
                )
                features = self._get_features_from_batch_df(
                    batch_df, **kwargs
                )
                if isinstance(features, tuple):
                    batch_cost.append(
                        self._train_one_batch(val, targets, *features)
                    )
                else:
                    batch_cost.append(
                        self._train_one_batch(val, targets, features)
                    )

            if verbose_each_epoch:
                batch_tqdm.set_description(
                    f'Epoch={epoch+1}, nAA={nAA}, batch={len(batch_cost)}, loss={batch_cost[-1]:.4f}'
                )
        return batch_cost

    def _train_one_batch(
        self, val:bool,
        targets:torch.Tensor,
        *features,
    ):
        """Training for a mini batch"""
        if not val:
            self.optimizer.zero_grad()
        predicts = self.model(*features)
        cost = self.loss_func(predicts, targets)
        if not val:
            cost.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        return cost.item()

    def _predict_one_batch(self,
        *features
    ):
        """Predicting for a mini batch"""
        return self.model(
            *features
        ).cpu().detach().numpy()

    def _get_targets_from_batch_df(self,
        batch_df:pd.DataFrame, **kwargs,
    )->torch.Tensor:
        """Tell the `train()` method how to get target values from the `batch_df`.
           All sub-classes must re-implement this method.
           Use torch.tensor(np.array, dtype=..., device=self.device) to convert tensor.

        Args:
            batch_df (pd.DataFrame): Dataframe of each mini batch.
            nAA (int, optional): Peptide length. Defaults to None.

        Raises:
            NotImplementedError: 'Must implement _get_targets_from_batch_df() method'

        Returns:
            torch.Tensor: Target value tensor
        """
        return self._as_tensor(
            batch_df[self.target_column_to_train].values,
            dtype=torch.float32
        )

    def _get_aa_indice_features(
        self, batch_df:pd.DataFrame
    )->torch.LongTensor:
        """
        Get indices values for 128 ascii codes.
        """
        return self._as_tensor(
            get_ascii_indices(
                batch_df['sequence'].values.astype('U')
            ),
            dtype=torch.long
        )

    def _get_26aa_indice_features(
        self, batch_df:pd.DataFrame
    )->torch.LongTensor:
        """
        Get indices values for 26 upper-case letters (amino acids),
        from 1 to 26. 0 is used for padding.
        """
        return self._as_tensor(
            get_batch_aa_indices(
                batch_df['sequence'].values.astype('U')
            ),
            dtype=torch.long
        )

    def _get_mod_features(
        self, batch_df:pd.DataFrame
    )->torch.Tensor:
        """
        Get modification features.
        """
        return self._as_tensor(
            get_batch_mod_feature(batch_df)
        )

    def _get_aa_mod_features(self,
        batch_df:pd.DataFrame, **kwargs,
    )->Tuple[torch.Tensor]:
        return (
            self._get_aa_indice_features(batch_df),
            self._get_mod_features(batch_df)
        )

    def _get_features_from_batch_df(self,
        batch_df:pd.DataFrame, **kwargs,
    )->Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Get input feature tensors of a batch of the precursor dataframe for the model.
        This will call `self._get_aa_indice_features(batch_df)` for sequence-level prediciton,
        or `self._get_aa_mod_features(batch_df)` for modified sequence-level.

        Args:
            batch_df (pd.DataFrame): Batch of precursor dataframe.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor]]:
                A feature tensor if call `self._get_aa_indice_features(batch_df)` (default).
                Or a tuple of tensors if call `self._get_aa_mod_features(batch_df)`.
        """
        return self._get_aa_indice_features(batch_df)

    def _prepare_predict_data_df(self,
        precursor_df:pd.DataFrame,
        **kwargs
    ):
        """
        This methods fills 0s in the column of
        `self.target_column_to_predict` in `precursor_df`,
        and then does `self.predict_df=precursor_df`.
        """
        precursor_df[self.target_column_to_predict] = 0.0
        self.predict_df = precursor_df

    def _prepare_train_data_df(self,
        precursor_df:pd.DataFrame,
        **kwargs
    ):
        """Changes to the training dataframe can be implemented here.

        Args:
            precursor_df (pd.DataFrame): Dataframe containing the training data.
        """
        pass

    def _set_batch_predict_data(self,
        batch_df:pd.DataFrame,
        predict_values:np.ndarray,
        **kwargs
    ):
        """Set predicted values into `self.predict_df`.

        Args:
            batch_df (pd.DataFrame): Dataframe of mini batch when predicting
            predict_values (np.array): Predicted values
        """
        predict_values[predict_values<self._min_pred_value] = self._min_pred_value
        if self._predict_in_order:
            self.predict_df.loc[:,self.target_column_to_predict].values[
                batch_df.index.values[0]:batch_df.index.values[-1]+1
            ] = predict_values
        else:
            self.predict_df.loc[
                batch_df.index,self.target_column_to_predict
            ] = predict_values

    def _set_optimizer(self, lr):
        """Set optimizer"""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr
        )

    def set_lr(self, lr:float):
        """Set learning rate"""
        if self.optimizer is None:
            self._set_optimizer(lr)
        else:
            for g in self.optimizer.param_groups:
                g['lr'] = lr

    def _get_lr_schedule_with_warmup(self, warmup_epoch, epoch):
        if warmup_epoch > epoch:
            warmup_epoch = epoch//2
        return get_cosine_schedule_with_warmup(
            self.optimizer, warmup_epoch, epoch
        )

    def _prepare_training(self, precursor_df, lr, **kwargs):
        if 'nAA' not in precursor_df.columns:
            precursor_df['nAA'] = precursor_df.sequence.str.len()
        self._prepare_train_data_df(precursor_df, **kwargs)
        self.model.train()

        self.set_lr(lr)

    def _check_predict_in_order(self, precursor_df:pd.DataFrame):
        if is_precursor_sorted(precursor_df):
            self._predict_in_order = True
        else:
            self._predict_in_order = False