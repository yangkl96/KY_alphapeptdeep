# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbdev_nbs/settings.ipynb.

# %% auto 0
__all__ = ['global_settings', 'model_const', 'update_settings', 'update_modifications']

# %% ../nbdev_nbs/settings.ipynb 1
# for nbdev_build_docs
# import os
# __file__ = os.path.expanduser('~/Workspace/alphapeptdeep/peptdeep/settings.py')

# %% ../nbdev_nbs/settings.ipynb 4
import os
import collections

from alphabase.yaml_utils import load_yaml
from alphabase.constants.modification import (
    load_mod_df, keep_modloss_by_importance
)

_base_dir = os.path.dirname(__file__)

global_settings = load_yaml(
    os.path.join(_base_dir, 'constants/default_settings.yaml')
)
for key, val in list(global_settings['model_mgr'][
    'instrument_group'
].items()):
    global_settings['model_mgr'][
        'instrument_group'
    ][key.upper()] = val

model_const = load_yaml(
    os.path.join(_base_dir, 'constants/model_const.yaml')
)

def update_settings(dict_, new_dict):
    for k, v in new_dict.items():
        if isinstance(v, collections.abc.Mapping):
            dict_[k] = update_settings(dict_.get(k, {}), v)
        else:
            dict_[k] = v
    return dict_

def update_modifications(tsv:str="", 
    modloss_importance_level:bool=global_settings[
        'common']['modloss_importance_level'
    ]
):
    if os.path.isfile(tsv):
        load_mod_df(tsv, modloss_importance_level=modloss_importance_level)
        
        from peptdeep.model.featurize import get_all_mod_features

        get_all_mod_features()
    else:
        keep_modloss_by_importance(modloss_importance_level)

update_modifications()
