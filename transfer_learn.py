#need to download alphapeptdeep from github
if __name__ == '__main__':
    print("Loading packages")
    import os
    import pandas as pd
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Training transfer model')
    parser.add_argument('peptdeep_folder', type = str, help='folder for peptdeep')
    parser.add_argument('psm_folder', type = str, help='folder for PSMs')
    parser.add_argument('ms_folder', type = str, help='folder for mass spec output files')
    parser.add_argument('output_folder', type = str, help='folder for saving transfer model')
    parser.add_argument('fragmentation', type = str, help='fragmentation method')

    parser.add_argument('--psm_type', type = str, help='type of PSMs; default = maxquant', nargs='?', default="maxquant")
    parser.add_argument('--ms_file_type', type = str, help='type of ms file; default = raw', nargs='?', default="raw")
    parser.add_argument('--min_score', type = int, help='minimum Andromeda score', nargs='?', default=100)
    parser.add_argument('--instrument', type = str, help='what mass spec; default = QE', nargs='?', default="Lumos")
    parser.add_argument('--model_type', type = str, help='generic, phos, hla, or digly; default = generic',
                        nargs='?', default="generic") #default
    parser.add_argument('--external_ms2_model', type=str, help='path to external ms2 model', nargs='?', default = "")
    parser.add_argument('--no_train_rt_ccs', action=argparse.BooleanOptionalAction, help='whether to train rt and ccs. Adding this flag will not train RT or CCS models')
    parser.add_argument('--alphapept_folder', type=str, help='folder for alphapept', nargs='?', default="./alphapept/")
    parser.add_argument('--settings_type', type=str, help='settings.yaml to use', default="default")
    parser.add_argument('--skip_filtering', type=bool, help='settings.yaml to use', default=False)
    parser.add_argument('--mask_mods', type=bool, help='whether to mask modloss fragments', default=True)

    args = parser.parse_args()
    peptdeep_folder = args.peptdeep_folder
    psm_folder = args.psm_folder
    ms_folder = args.ms_folder
    output_folder = args.output_folder
    fragmentation = args.fragmentation

    psm_type = args.psm_type
    ms_file_type = args.ms_file_type
    min_score = args.min_score
    instrument = args.instrument
    model_type = args.model_type
    external_ms2_model = args.external_ms2_model
    no_train_rt_ccs = args.no_train_rt_ccs
    alphapept_folder = args.alphapept_folder
    settings_type = args.settings_type
    skip_filtering = args.skip_filtering
    mask_mods = args.mask_mods
    if model_type == "phos":
        mask_mods = False

    sys.path.insert(1, peptdeep_folder)
    sys.path.insert(2, alphapept_folder)

    with open("peptdeep/constants/settings_type.txt", "w") as f:
        f.write(settings_type)
    from peptdeep.pipeline_api import transfer_learn
    from peptdeep.settings import global_settings

    # enforce min score by writing a new msms.txt?
    mgr_settings = global_settings['model_mgr']
    mgr_settings["mask_modloss"] = mask_mods
    mgr_settings['transfer']['psm_type'] = 'maxqaunt'
    mgr_settings["transfer"]["grid_nce_search"] = True

    print("Reading in files")
    all_psm_files = []
    for root, dirs, files in os.walk(psm_folder):
        for file in files:
            if file == "msms.txt":
                if not skip_filtering:
                    df = pd.read_csv(os.path.join(root, file), sep="\t")

                    df = df[df["Score"] >= min_score]
                    df = df[df["Fragmentation"].str.lower() == fragmentation.lower()]
                    df.to_csv(os.path.join(root, "msms_filter.txt"), sep = "\t", index=False)

                all_psm_files.append(os.path.join(root, "msms_filter.txt"))

    print("done finding psm files")
    mgr_settings["transfer"]["psm_files"] = all_psm_files
    #mgr_settings["transfer"]["psm_files"] = ["Z:\\yangkl\\hla_etd_transfer_learn\\msms\\TUM_HLA_1_01_01_ETD-1h-R4-unspecific\\msms_filter.txt"]
    #mgr_settings["transfer"]["psm_files"] = [
    #    "Z:\\yangkl\\predfullTransferLearn\\msms\\TUM_aspn_1_01_01_DDA-1h-R1-AspN\\msms_filter.txt"]
    mgr_settings["transfer"]["psm_type"] = "maxquant"

    all_ms_files = []
    for root, dirs, files in os.walk(ms_folder):
        for file in files:
            if file.endswith(".raw"):
                all_ms_files.append(os.path.join(root, file))

    print("done finding msms files")
    mgr_settings["transfer"]["ms_files"] = all_ms_files
    #mgr_settings["transfer"]["ms_files"] = ["Z:\\yangkl\\hla_etd_transfer_learn\\raw\\02445d_BA1-TUM_HLA_1_01_01-ETD-1h-R4.raw"]
    #mgr_settings["transfer"]["ms_files"] = [
    #    "Z:\\yangkl\\predfullTransferLearn\\raw\\03210a_BE1-TUM_aspn_1_01_01-DDA-1h-R1.raw"]
    mgr_settings["transfer"]["ms_file_type"] = "thermo"

    mgr_settings["transfer"]["model_output_folder"] = output_folder
    
    mgr_settings["model_type"] = model_type
    mgr_settings["external_ms2_model"] = external_ms2_model
    mgr_settings["grid_instrument"] = instrument

    if no_train_rt_ccs:
        mgr_settings["transfer"]['epoch_rt_ccs'] = 0
        mgr_settings["transfer"]['psm_num_to_train_rt_ccs'] = 0
        mgr_settings["transfer"]['psm_num_per_mod_to_train_rt_ccs'] = 0

    print("Transfer learning beep boop")
    transfer_learn()