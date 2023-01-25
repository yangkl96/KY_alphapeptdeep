def getHeader(row, pred_entry):
    title = row["base"]

    # convert title to basecharge via dict
    pred_entry += "TITLE=" + title + "\n"
    pred_entry += "CHARGE=" + str(row["charge"]) + "\n"
    pred_entry += "RTINSECONDS=" + str(row["rt_norm_pred"]) + "\n"
    return (pred_entry)


def getMzsAndIntensities(start, end, pred_entry):
    my_mzs = mzs[start:end, :].flatten()
    my_intensities = intensities[start:end, :].flatten()
    my_fragments = fragment_ion_types * (end - start)

    for m, i, frag in zip(my_mzs, my_intensities, my_fragments):
        if i != 0:
            pred_entry += str(m) + "\t" + str(i) + " " + frag + "\n"
    return (pred_entry)

def getPredictionEntry(row):
    pred_entry = "BEGIN IONS\n"
    pred_entry = getHeader(row, pred_entry)
    pred_entry = getMzsAndIntensities(row["frag_start_idx"], row["frag_stop_idx"], pred_entry)
    pred_entry += "END IONS\n"
    return (pred_entry)

from multiprocessing import freeze_support
if __name__ == '__main__':
    freeze_support()
    print("Loading packages")
    import argparse
    import sys
    import time

    parser = argparse.ArgumentParser(description='Running base model')
    parser.add_argument('spectraRTinput', type=str, help='peptdeep input spectraRT.csv from MSBooster')
    parser.add_argument('--peptdeep_folder', type=str, help='folder for peptdeep', default = ".")
    parser.add_argument('--model_type', type=str, help='generic, phos, hla, or digly', default = "generic")
    parser.add_argument('--external_ms2_model', type=str, help='path to external ms2 model', default = "")
    parser.add_argument('--external_rt_model', type=str, help='path to external rt model', default="")
    parser.add_argument('--external_ccs_model', type=str, help='path to external ccs model', default="")
    parser.add_argument('--predict_ccs', action=argparse.BooleanOptionalAction, help='whether to predict ccs')
    parser.add_argument('--settings_type', type=str, help='settings.yaml to use', default="hcd")
    parser.add_argument('--mask_mods', action=argparse.BooleanOptionalAction, help='whether to mask modloss fragments',
                        default=False)
    parser.add_argument('--additional_mods', type=str,
                        help='path to additional PTMs to consider. Please format as a tsv with the following columns: '
                             'mod_name	unimod_mass	unimod_avge_mass	'
                             'composition	unimod_modloss	modloss_composition	'
                             'classification	unimod_id	modloss_importance. '
                             'Please refer to modification_alphapeptdeep.tsv for proper formatting.', default=None)

    args = parser.parse_args()
    sys.path.insert(1, args.peptdeep_folder)

    import os
    import pandas as pd
    import alphabase
    import shutil

    alphabase_path = os.path.dirname(alphabase.__file__)

    # make sure mods with modlosses get modloss importance changed to 1 so they can be predicted
    modification_tsv = os.path.join(alphabase_path, "constants", "const_files", "modification.tsv")

    # good to get a copy so that modifications.tsv is always preserved
    if os.path.isfile(os.path.join(alphabase_path, "constants", "const_files", "tmp_modification.tsv")):
        shutil.copy(os.path.join(alphabase_path, "constants", "const_files", "tmp_modification.tsv"),
                    os.path.join(alphabase_path, "constants", "const_files", "modification.tsv"))
    else:
        shutil.copy(os.path.join(alphabase_path, "constants", "const_files", "modification.tsv"),
                    os.path.join(alphabase_path, "constants", "const_files", "tmp_modification.tsv"))

    mod_df = pd.read_csv(modification_tsv, sep="\t")

    # add more modifications if necessary
    if args.additional_mods:
        add_df = pd.read_csv(args.additional_mods, sep="\t")
        mod_df = pd.concat([mod_df, add_df], axis=0)

    # make sure modlosses are enabled
    mod_df["modloss_importance"] = mod_df.apply(
        lambda x: max(1, x["modloss_importance"]) if x["unimod_modloss"] != 0 else 0, axis=1)
    mod_df.to_csv(modification_tsv, sep="\t", index=False)

    # write file that contains settings_type
    with open("peptdeep/constants/settings_type.txt", "w") as f:
        f.write(args.settings_type)
    from peptdeep.settings import global_settings
    from peptdeep.pretrained_models import ModelManager

    model_mgr_settings = global_settings['model_mgr']
    model_mgr_settings["model_type"] = args.model_type
    if args.external_ms2_model != "":
        ms2_model_list = args.external_ms2_model.split(",")
        rt_model_list = args.external_rt_model.split(",")
        if len(ms2_model_list) != len(rt_model_list):
            print("There must be the same number of ms2 and rt models provided, separated by commas.")
            sys.exit(1)
        if args.predict_ccs:
            ccs_model_list = args.external_ccs_model.split(",")
            if len(ms2_model_list) != len(ccs_model_list):
                print("There must be the same number of ms2, rt, and ccs models provided, separated by commas.")
                sys.exit(1)
        else:
            ccs_model_list = [""] * len(ms2_model_list)
    else:
        ms2_model_list = [""]
        rt_model_list = [""]
        ccs_model_list = [""]
    num_models = len(ms2_model_list)

    # predict
    start = time.time()
    predict_items = ['rt', 'ms2']
    if args.predict_ccs:
        predict_items.append("ccs")

    print("Loading peptides")
    total_df = pd.read_csv(args.spectraRTinput)
    total_df.fillna('', inplace=True)

    output_dir = os.path.dirname(args.spectraRTinput)
    if output_dir == "":
        output_dir = "."

    mgf_file_path = output_dir + "/spectraRT_alphapeptdeep.mgf"
    with open(mgf_file_path, "w") as f:
        f.write("")

    for ms2_model, rt_model, ccs_model, model_num in \
            zip(ms2_model_list, rt_model_list, ccs_model_list, range(num_models)):
        # setting parameters
        model_mgr = None

        if ms2_model != "":
            model_mgr_settings["external_ms2_model"] = ms2_model
            print("Using " + model_mgr_settings["external_ms2_model"] + " as ms2 model")
        if rt_model != "":
            model_mgr_settings["external_rt_model"] = rt_model
            print("Using " + model_mgr_settings["external_rt_model"] + " as rt model")
        if ccs_model != "":
            model_mgr_settings["external_ccs_model"] = ccs_model
            print("Using " + model_mgr_settings["external_ccs_model"] + " as ccs model")

        if not args.mask_mods:
            model_mgr = ModelManager(mask_modloss=False)
        else:
            model_mgr = ModelManager(mask_modloss=True)

        #get PSMs to be predicted by each model
        df = total_df[total_df["scan_num"] % num_models == model_num].copy()
        predict_dict = model_mgr.predict_all(df, predict_items = predict_items)

        # get mgf entries using pd.apply
        mzs = predict_dict["fragment_mz_df"].to_numpy()
        intensities = predict_dict["fragment_intensity_df"].to_numpy()
        fragment_ion_types = list(predict_dict["fragment_mz_df"].columns)
        # replace fragment ion types with names supported by msbooster
        # might consider making separate category for PTM neutral losses
        fragment_name_replace = {}
        fragment_name_replace["b_z1"] = "b"
        fragment_name_replace["b_z2"] = "b"
        fragment_name_replace["b_H2O_z1"] = "b-NL"
        fragment_name_replace["b_H2O_z2"] = "b-NL"
        fragment_name_replace["b_NH3_z1"] = "b-NL"
        fragment_name_replace["b_NH3_z2"] = "b-NL"
        fragment_name_replace["b_modloss_z1"] = "b-NL"
        fragment_name_replace["b_modloss_z2"] = "b-NL"
        fragment_name_replace["y_z1"] = "y"
        fragment_name_replace["y_z2"] = "y"
        fragment_name_replace["y_H2O_z1"] = "y-NL"
        fragment_name_replace["y_H2O_z2"] = "y-NL"
        fragment_name_replace["y_NH3_z1"] = "y-NL"
        fragment_name_replace["y_NH3_z2"] = "y-NL"
        fragment_name_replace["y_modloss_z1"] = "y-NL"
        fragment_name_replace["y_modloss_z2"] = "y-NL"
        fragment_name_replace["c_z1"] = "c"
        fragment_name_replace["c_z2"] = "c"
        fragment_name_replace["z_z1"] = "z"
        fragment_name_replace["z_z2"] = "z"
        for i in range(len(fragment_ion_types)):
            fragment_ion_types[i] = fragment_name_replace[fragment_ion_types[i]]

        mgf_series = predict_dict["precursor_df"].apply(lambda x: getPredictionEntry(x), axis=1)
        f = open(mgf_file_path, "a")
        for entry in mgf_series:
            f.write(entry)
        f.close()
    end = time.time()
    print("Prediction took " + str(end - start) + " sec")