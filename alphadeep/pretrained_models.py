# AUTOGENERATED! DO NOT EDIT! File to edit: nbdev_nbs/pretrained_models.ipynb (unless otherwise specified).

__all__ = ['download_models', 'sandbox_dir', 'model_zip', 'load_phos_models', 'load_models',
           'load_models_by_model_type_in_zip', 'ModelManager']

# Cell
import os
import io
import wget
import pandas as pd
from zipfile import ZipFile
from typing import Tuple

sandbox_dir = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    ),
    'sandbox'
)

if not os.path.exists(sandbox_dir):
    os.makedirs(sandbox_dir)

model_zip = os.path.join(
    sandbox_dir,
    'released_models/alphadeep_models.zip'
)

def download_models(
    url='https://datashare.biochem.mpg.de/s/ABnQuD2KkXfIGF3/download',
    overwrite=True
):
    downloaded_zip = os.path.join(
        sandbox_dir,'released_models.zip'
    )
    if os.path.exists(model_zip):
        if overwrite:
            os.remove(model_zip)
        else:
            return

    print('[Start] Downloading alphadeep_models.zip ...')
    wget.download(url, downloaded_zip)
    _zip = ZipFile(downloaded_zip)
    _zip.extract('released_models/alphadeep_models.zip', sandbox_dir)
    _zip.close()
    os.remove(downloaded_zip)
    print('[Done] Downloading alphadeep_models.zip')

if not os.path.exists(model_zip):
    download_models()

# Cell
from alphadeep.model.ms2 import (
    pDeepModel, normalize_training_intensities
)
from alphadeep.model.rt import AlphaRTModel, uniform_sampling
from alphadeep.model.ccs import AlphaCCSModel

def load_phos_models(mask_phos_modloss=False):
    ms2_model = pDeepModel(mask_modloss=mask_phos_modloss)
    ms2_model.load(model_zip, model_path_in_zip='phospho/ms2_phos.pth')
    rt_model = AlphaRTModel()
    rt_model.load(model_zip, model_path_in_zip='phospho/irt_phos.pth')
    ccs_model = AlphaCCSModel()
    ccs_model.load(model_zip, model_path_in_zip='regular/ccs.pth')
    return ms2_model, rt_model, ccs_model

def load_models():
    ms2_model = pDeepModel()
    ms2_model.load(model_zip, model_path_in_zip='regular/ms2.pth')
    rt_model = AlphaRTModel()
    rt_model.load(model_zip, model_path_in_zip='regular/rt.pth')
    ccs_model = AlphaCCSModel()
    ccs_model.load(model_zip, model_path_in_zip='regular/ccs.pth')
    return ms2_model, rt_model, ccs_model

def load_models_by_model_type_in_zip(model_type_in_zip:str):
    ms2_model = pDeepModel()
    ms2_model.load(model_zip, model_path_in_zip=f'{model_type_in_zip}/ms2.pth')
    rt_model = AlphaRTModel()
    rt_model.load(model_zip, model_path_in_zip=f'{model_type_in_zip}/rt.pth')
    ccs_model = AlphaCCSModel()
    ccs_model.load(model_zip, model_path_in_zip=f'{model_type_in_zip}/ccs.pth')
    return ms2_model, rt_model, ccs_model


# Cell
class ModelManager(object):
    def __init__(self):
        self.ms2_model:pDeepModel = None
        self.rt_model:AlphaRTModel = None
        self.ccs_model:AlphaCCSModel = None

        self.grid_nce_search = True

        self.n_psm_to_tune_ms2 = 5000
        self.epoch_to_tune_ms2 = 10
        self.n_psm_to_tune_rt_ccs = 3000
        self.epoch_to_tune_rt_ccs = 20

    def load_installed_models(self, model_type='regular', mask_modloss=True):
        """ Load built-in MS2/CCS/RT models.
        Args:
            model_type (str, optional): To load the installed MS2/RT/CCS models
                or phos MS2/RT/CCS models. It could be 'phospho', 'regular', or
                model_type (model sub-folder) in alphadeep_models.zip.
                Defaults to 'regular'.
            mask_modloss (bool, optional): If modloss ions are masked to zeros
                in the ms2 model. `modloss` ions are mostly useful for phospho
                MS2 prediciton model. Defaults to True.
        """
        if model_type.lower() in ['phospho','phos']:
            (
                self.ms2_model, self.rt_model, self.ccs_model
            ) = load_phos_models(mask_modloss)
        elif model_type.lower() in ['regular','common']:
            (
                self.ms2_model, self.rt_model, self.ccs_model
            ) = load_models()
        else:
            (
                self.ms2_model, self.rt_model, self.ccs_model
            ) = load_models_by_model_type_in_zip(model_type)

    def load_external_models(self,
        *,
        ms2_model_file: Tuple[str, io.BytesIO]=None,
        rt_model_file: Tuple[str, io.BytesIO]=None,
        ccs_model_file: Tuple[str, io.BytesIO]=None,
        mask_modloss=True
    ):
        """Load external MS2/RT/CCS models

        Args:
            ms2_model_file (Tuple[str, io.BytesIO], optional): ms2 model file or stream.
                It will load the installed model if the value is None. Defaults to None.
            rt_model_file (Tuple[str, io.BytesIO], optional): rt model file or stream.
                It will load the installed model if the value is None. Defaults to None.
            ccs_model_file (Tuple[str, io.BytesIO], optional): ccs model or stream.
                It will load the installed model if the value is None. Defaults to None.
            mask_modloss (bool, optional): If modloss ions are masked to zeros
                in the ms2 model. Defaults to True.
        """
        self.ms2_model = pDeepModel(mask_modloss=mask_modloss)
        self.rt_model = AlphaRTModel()
        self.ccs_model = AlphaCCSModel()

        if ms2_model_file is not None:
            self.ms2_model.load(ms2_model_file)
        else:
            self.ms2_model.load(model_zip, model_path_in_zip='regular/ms2.pth')
        if rt_model_file is not None:
            self.rt_model.load(rt_model_file)
        else:
            self.rt_model.load(model_zip, model_path_in_zip='regular/rt.pth')
        if ccs_model_file is not None:
            self.ccs_model.load(ccs_model_file)
        else:
            self.ccs_model.load(model_zip, model_path_in_zip='regular/ccs.pth')

    def fine_tune_rt_model(self,
        psm_df:pd.DataFrame
    ):
        """ Fine-tune the RT model. The fine-tuning will be skipped
            if `n_rt_ccs_tune` is zero.

        Args:
            psm_df (pd.DataFrame): training psm_df which contains 'rt_norm' column.
        """
        if self.n_psm_to_tune_rt_ccs > 0:
            tr_df = uniform_sampling(
                psm_df, target='rt_norm',
                n_train=self.n_psm_to_tune_rt_ccs,
                return_test_df=False
            )
            self.rt_model.train(tr_df,
                epoch=self.epoch_to_tune_rt_ccs
            )

    def fine_tune_ccs_model(self,
        psm_df:pd.DataFrame,
    ):
        """ Fine-tune the CCS model. The fine-tuning will be skipped
            if `n_rt_ccs_tune` is zero.

        Args:
            psm_df (pd.DataFrame): training psm_df which contains 'ccs' column.
        """

        if self.n_psm_to_tune_rt_ccs > 0:
            tr_df = uniform_sampling(
                psm_df, target='ccs',
                n_train=self.n_psm_to_tune_rt_ccs,
                return_test_df=False
            )
            self.ccs_model.train(tr_df,
                epoch=self.epoch_to_tune_rt_ccs
            )

    def fine_tune_ms2_model(self,
        psm_df: pd.DataFrame,
        matched_intensity_df: pd.DataFrame
    ):
        if self.n_psm_to_tune_ms2 > 0:
            tr_df = psm_df.sample(self.n_psm_to_tune_ms2).copy()
            tr_df, frag_df = normalize_training_intensities(
                tr_df, matched_intensity_df
            )
            tr_inten_df = pd.DataFrame()
            for frag_type in self.ms2_model.charged_frag_types:
                if frag_type in frag_df.columns:
                    tr_inten_df[frag_type] = frag_df[frag_type]
                else:
                    tr_inten_df[frag_type] = 0

            if self.grid_nce_search:
                nce, instrument = self.ms2_model.grid_nce_search(
                    tr_df, tr_inten_df
                )
                tr_df['nce'] = nce
                tr_df['instrument'] = instrument
                psm_df['nce'] = nce
                psm_df['instrument'] = instrument

            self.ms2_model.train(tr_df,
                fragment_inten_df=tr_inten_df,
                epoch=self.epoch_to_tune_ms2
            )
