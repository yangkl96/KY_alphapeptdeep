# AUTOGENERATED! DO NOT EDIT! File to edit: nbdev_nbs/pretrained_models.ipynb (unless otherwise specified).

__all__ = ['download_models', 'sandbox_dir', 'model_zip', 'count_mods', 'psm_samping_with_important_mods',
           'load_phos_models', 'load_HLA_models', 'load_models', 'load_models_by_model_type_in_zip', 'mgr_settings',
           'ModelManager']

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

from alphadeep.settings import global_settings
mgr_settings = global_settings['model_mgr']

def count_mods(psm_df)->pd.DataFrame:
    mods = psm_df[
        psm_df.mods.str.len()>0
    ].mods.apply(lambda x: x.split(';'))
    mod_dict = {}
    mod_dict['mutation'] = {}
    mod_dict['mutation']['spec_count'] = 0
    for one_mods in mods.values:
        for mod in set(one_mods):
            items = mod.split('->')
            if (
                len(items)==2
                and len(items[0])==3
                and len(items[1])==5
            ):
                mod_dict['mutation']['spec_count'] += 1
            elif mod not in mod_dict:
                mod_dict[mod] = {}
                mod_dict[mod]['spec_count'] = 1
            else:
                mod_dict[mod]['spec_count'] += 1
    return pd.DataFrame().from_dict(
            mod_dict, orient='index'
        ).reset_index(drop=False).rename(
            columns={'index':'mod'}
        ).sort_values(
            'spec_count',ascending=False
        ).reset_index(drop=True)

def psm_samping_with_important_mods(
    psm_df, n_sample,
    top_n_mods = 10,
    n_sample_each_mod = 0,
    uniform_sampling_column = None,
    random_state=1337,
):
    psm_df_list = []
    if uniform_sampling_column is None:
        def _sample(psm_df, n):
            if n < len(psm_df):
                return psm_df.sample(
                    n, random_state=random_state
                ).copy()
            else:
                return psm_df.copy()
    else:
        def _sample(psm_df, n):
            return uniform_sampling(
                psm_df, target=uniform_sampling_column,
                n_train = n, random_state=random_state
            )

    psm_df_list.append(_sample(psm_df, n_sample))
    if n_sample_each_mod > 0:
        mod_df = count_mods(psm_df)
        mod_df = mod_df[mod_df!='mutation']

        if len(mod_df) > top_n_mods:
            mod_df = mod_df.iloc[:top_n_mods,:]
        for mod in mod_df.mod.values:
            psm_df_list.append(
                _sample(
                    psm_df[psm_df.mods.str.contains(mod)],
                    n_sample_each_mod,
                )
            )
    return pd.concat(psm_df_list)

def load_phos_models(mask_phos_modloss=False):
    ms2_model = pDeepModel(mask_modloss=mask_phos_modloss)
    ms2_model.load(model_zip, model_path_in_zip='phospho/ms2.pth')
    rt_model = AlphaRTModel()
    rt_model.load(model_zip, model_path_in_zip='phospho/rt.pth')
    ccs_model = AlphaCCSModel()
    ccs_model.load(model_zip, model_path_in_zip='regular/ccs.pth')
    return ms2_model, rt_model, ccs_model

def load_HLA_models():
    ms2_model = pDeepModel(mask_modloss=True)
    ms2_model.load(model_zip, model_path_in_zip='HLA/ms2.pth')
    rt_model = AlphaRTModel()
    rt_model.load(model_zip, model_path_in_zip='HLA/rt.pth')
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

        self.grid_nce_search = mgr_settings[
            'fine_tune'
        ]['grid_nce_search']

        self.n_psm_to_tune_ms2 = 5000
        self.n_mod_psm_to_tune_ms2 = 100
        self.epoch_to_tune_ms2 = mgr_settings[
            'fine_tune'
        ]['epoch_ms2']
        self.n_psm_to_tune_rt_ccs = 3000
        self.n_mod_psm_to_tune_rt_ccs = 100
        self.epoch_to_tune_rt_ccs = mgr_settings[
            'fine_tune'
        ]['epoch_rt_ccs']

        self.top_n_mods_to_tune = 10

        self.nce = mgr_settings[
            'predict'
        ]['default_nce']
        self.instrument = mgr_settings[
            'predict'
        ]['default_instrument']

    def set_default_nce(self, df):
        df['nce'] = self.nce
        df['instrument'] = self.instrument

    def load_installed_models(self, model_type='regular', mask_modloss=True):
        """ Load built-in MS2/CCS/RT models.
        Args:
            model_type (str, optional): To load the installed MS2/RT/CCS models
                or phos MS2/RT/CCS models. It could be 'phospho', 'HLA', 'regular', or
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
        elif model_type.lower() in [
            'hla','unspecific','non-specific', 'nonspecific'
        ]:
            (
                self.ms2_model, self.rt_model, self.ccs_model
            ) = load_HLA_models()
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
        psm_df:pd.DataFrame,
    ):
        """ Fine-tune the RT model. The fine-tuning will be skipped
            if `n_rt_ccs_tune` is zero.

        Args:
            psm_df (pd.DataFrame): training psm_df which contains 'rt_norm' column.
        """
        if self.n_psm_to_tune_rt_ccs > 0:
            tr_df = psm_samping_with_important_mods(
                psm_df, self.n_psm_to_tune_rt_ccs,
                self.top_n_mods_to_tune,
                self.n_mod_psm_to_tune_rt_ccs,
                uniform_sampling_column='rt_norm'
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
            tr_df = psm_samping_with_important_mods(
                psm_df, self.n_psm_to_tune_rt_ccs,
                self.top_n_mods_to_tune,
                self.n_mod_psm_to_tune_rt_ccs,
                uniform_sampling_column='ccs'
            )
            self.rt_model.train(tr_df,
                epoch=self.epoch_to_tune_rt_ccs
            )

    def fine_tune_ms2_model(self,
        psm_df: pd.DataFrame,
        matched_intensity_df: pd.DataFrame,
    ):
        if self.n_psm_to_tune_ms2 > 0:
            tr_df = psm_samping_with_important_mods(
                psm_df, self.n_psm_to_tune_ms2,
                self.top_n_mods_to_tune,
                self.n_mod_psm_to_tune_ms2
            )
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
                self.nce, self.instrument = self.ms2_model.grid_nce_search(
                    tr_df, tr_inten_df,
                    nce_first=mgr_settings['fine_tune'][
                        'grid_nce_first'
                    ],
                    nce_last=mgr_settings['fine_tune'][
                        'grid_nce_last'
                    ],
                    nce_step=mgr_settings['fine_tune'][
                        'grid_nce_step'
                    ],
                    search_instruments=mgr_settings['fine_tune'][
                        'grid_instrument'
                    ],
                )
                tr_df['nce'] = self.nce
                tr_df['instrument'] = self.instrument

            self.ms2_model.train(tr_df,
                fragment_intensity_df=tr_inten_df,
                epoch=self.epoch_to_tune_ms2
            )

    def predict_ms2(self, psm_df:pd.DataFrame,
        *, batch_size=mgr_settings[
             'predict'
           ]['batch_size_ms2']
    ):
        if 'nce' not in psm_df.columns:
            self.set_default_nce(psm_df)
        return self.ms2_model.predict(psm_df,
            batch_size=batch_size
        )

    def predict_rt(self, psm_df:pd.DataFrame,
        *, batch_size=mgr_settings[
             'predict'
           ]['batch_size_rt_ccs']
    ):
        return self.rt_model.predict(psm_df,
            batch_size=batch_size
        )

    def predict_mobility(self, psm_df:pd.DataFrame,
        *, batch_size=mgr_settings[
             'predict'
           ]['batch_size_rt_ccs']
    ):
        return self.ccs_model.predict(psm_df,
            batch_size=batch_size
        )
