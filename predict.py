def getPredictionEntry(row):
    pred_entry = "BEGIN IONS\n"

    def getHeader(row, pred_entry):
        # formatting to base format

        # traverse backwards through mods
        # need to map ptm name to mod mass
        # title = row["sequence"]
        # sites = row["mod_sites"].split(";")[::-1]
        # mods = row["mods"].split(";")[::-1]
        # for mod, site in zip(mods, sites):
        #    if site == "":
        #        break
        #    title = title[0:int(site)] + "[" + mod.split("@")[0] + "]" + title[int(site)+1:len(title)]
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

    pred_entry = getHeader(row, pred_entry)
    pred_entry = getMzsAndIntensities(row["frag_start_idx"], row["frag_end_idx"], pred_entry)
    pred_entry += "END IONS\n"
    return (pred_entry)

from multiprocessing import freeze_support
if __name__ == '__main__':
    freeze_support()
    print("Loading packages")
    import argparse
    import sys
    from alphabase.yaml_utils import load_yaml
    import time

    parser = argparse.ArgumentParser(description='Running base model')
    parser.add_argument('spectraRTinput', type=str, help='peptdeep input spectraRT.csv from MSBooster')
    parser.add_argument('--peptdeep_folder', type=str, help='folder for peptdeep', default = ".")
    parser.add_argument('--model_type', type=str, help='generic, phospho, hla, or digly', default = "generic")
    parser.add_argument('--external_ms2_model', type=str, help='path to external ms2 model', default = "")
    parser.add_argument('--external_rt_model', type=str, help='path to external rt model', default="")
    #parser.add_argument('--frag_types', type=str, help='fragment ion types, separated by a comma and nothing else', default = "b,y,b_modloss,y_modloss")
    parser.add_argument('--settings_type', type=str, help='settings.yaml to use', default="hcd")
    parser.add_argument('--mask_mods', type=bool, help='whether to mask modloss fragments', default=False)

    args = parser.parse_args()
    sys.path.insert(1, args.peptdeep_folder)

    import os
    import pandas as pd
    #write file that contains settings_type
    with open("peptdeep/constants/settings_type.txt", "w") as f:
        f.write(args.settings_type)
    from peptdeep.settings import global_settings
    #_base_dir = os.path.dirname(__file__)
    #global_settings = load_yaml(
    #    os.path.join(_base_dir, "peptdeep/constants/" + settings_type + '_settings.yaml')
    #)
    from peptdeep.pretrained_models import ModelManager

    #from alphabase.yaml_utils import load_yaml
    #peptdeep.settings.global_settings = load_yaml(
    #    os.path.join(os.path.dirname(__file__), settings)
    #)
    #global_settings = peptdeep.settings.global_settings


    # setting parameters
    #global_settings['model']['frag_types'] = frag_types.split(",")
    model_mgr_settings = global_settings['model_mgr']

    model_mgr = ""
    if args.external_ms2_model != "":
        model_mgr_settings["external_ms2_model"] = args.external_ms2_model
    if args.external_rt_model != "":
        model_mgr_settings["external_rt_model"] = args.external_rt_model
    print("Using " + model_mgr_settings["external_ms2_model"] + " as ms2 model")

    model_mgr_settings["model_type"] = args.model_type
    if model_mgr_settings["model_type"] == "phos" or not args.mask_mods:
        model_mgr = ModelManager(mask_modloss=False)
    else:
        model_mgr = ModelManager(mask_modloss=True)

    # predict
    start = time.time()
    print("Loading peptides")
    df = pd.read_csv(args.spectraRTinput)
    df.fillna('', inplace=True)
    predict_dict = model_mgr.predict_all(df, predict_items=['rt', 'ms2'])

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

    print("Writing mgf")
    output_dir = os.path.dirname(args.spectraRTinput)
    if output_dir == "":
        output_dir = "."

    mgf_file_path = output_dir + "/spectraRT_alphapeptdeep.mgf"
    with open(mgf_file_path, "w") as f:
        f.write("")

    mgf_series = predict_dict["precursor_df"].apply(lambda x: getPredictionEntry(x), axis=1)
    f = open(mgf_file_path, "a")
    for entry in mgf_series:
        f.write(entry)
    f.close()
    end = time.time()
    print("prediction took " + str(end - start) + " sec")