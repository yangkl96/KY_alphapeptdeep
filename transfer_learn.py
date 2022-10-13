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
    parser.add_argument('--min_score', type = int, help='minimum Andromeda score', nargs='?', default=150)
    parser.add_argument('--instrument', type = str, help='what mass spec; default = QE', nargs='?', default="Lumos")
    parser.add_argument('--model_type', type = str, help='generic, phos, hla, or digly; default = generic',
                        nargs='?', default="generic") #default
    parser.add_argument('--external_ms2_model', type=str, help='path to external ms2 model', nargs='?', default = "")
    parser.add_argument('--no_train_rt_ccs', action=argparse.BooleanOptionalAction,
                        help='whether to train rt and ccs. Adding this flag will not train RT or CCS models')
    parser.add_argument('--no_train_ms2', action=argparse.BooleanOptionalAction,
                        help='whether to train ms2. Adding this flag will not train ms2 models')
    parser.add_argument('--alphapept_folder', type=str, help='folder for alphapept', nargs='?', default="./alphapept/")
    parser.add_argument('--settings_type', type=str, help='settings.yaml to use', default="default")
    parser.add_argument('--skip_filtering', type=bool, help='settings.yaml to use', default=False)
    parser.add_argument('--mask_mods', type=bool, help='whether to mask modloss fragments', default=True)
    parser.add_argument('--lr_ms2', help='learning rate for ms2', default=0.0001)
    parser.add_argument('--epoch_ms2', help='number of epochs to train ms2', default=20)
    parser.add_argument('--lr_rt', help='learning rate for rt', default=0.0001)
    parser.add_argument('--epoch_rt', help='number of epochs to train rt', default=40)
    parser.add_argument('--batch_size_ms2', help='batch size for ms2', default=512)
    parser.add_argument('--batch_size_rt', help='batch size for rt', default=512)
    parser.add_argument('--grid_search', action=argparse.BooleanOptionalAction, help='whether to grid search over parameters separated by commas', default=False)
    parser.add_argument('--processing_only', action=argparse.BooleanOptionalAction, help='whether to just do preprocessing of files', default=False)

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
    no_train_ms2 = args.no_train_ms2
    alphapept_folder = args.alphapept_folder
    settings_type = args.settings_type
    skip_filtering = args.skip_filtering
    mask_mods = args.mask_mods
    lr_ms2 = args.lr_ms2
    epoch_ms2 = args.epoch_ms2
    lr_rt = args.lr_rt
    epoch_rt = args.epoch_rt
    batch_size_ms2 = args.batch_size_ms2
    batch_size_rt = args.batch_size_rt
    grid_search = args.grid_search

    if model_type == "phos":
        mask_mods = False

    sys.path.insert(1, peptdeep_folder)
    sys.path.insert(2, alphapept_folder)

    with open("peptdeep/constants/settings_type.txt", "w") as f:
        f.write(settings_type)
    from peptdeep.pipeline_api import transfer_learn
    from peptdeep.settings import global_settings
    import datetime

    #general settings
    mgr_settings = global_settings['model_mgr']
    mgr_settings["mask_modloss"] = mask_mods
    mgr_settings['transfer']['psm_type'] = 'maxqaunt'
    mgr_settings["transfer"]["grid_nce_search"] = True
    mgr_settings["model_type"] = model_type
    mgr_settings["external_ms2_model"] = external_ms2_model
    mgr_settings["grid_instrument"] = instrument

    if no_train_rt_ccs:
        mgr_settings["transfer"]['epoch_rt_ccs'] = 0
        mgr_settings["transfer"]['psm_num_to_train_rt_ccs'] = 0
        mgr_settings["transfer"]['psm_num_per_mod_to_train_rt_ccs'] = 0
    if no_train_ms2:
        mgr_settings["transfer"]['epoch_ms2'] = 0
        mgr_settings["transfer"]['psm_num_to_train_ms2'] = 0
        mgr_settings["transfer"]['psm_num_per_mod_to_train_ms2'] = 0

    print("Reading in files")
    all_psm_files = []
    psm_folders = psm_folder.split(",")
    for psm_f in psm_folders:
        print(psm_f)
        for root, dirs, files in os.walk(psm_f):
            for file in files:
                if file == "msms.txt":
                    if not skip_filtering:
                        print(os.path.join(root, file))
                        df = pd.read_csv(os.path.join(root, file), sep="\t")

                        df = df[df["Score"] >= min_score]
                        df = df[df["Fragmentation"].str.lower() == fragmentation.lower()]
                        df.to_csv(os.path.join(root, "msms_filter.txt"), sep = "\t", index=False)

                    all_psm_files.append(os.path.join(root, "msms_filter.txt"))

    print("done finding psm files")
    if (args.processing_only):
        print("processing done")
        sys.exit(0)
    mgr_settings["transfer"]["psm_files"] = all_psm_files
    mgr_settings["transfer"]["psm_type"] = "maxquant"

    all_ms_files = []
    ms_folders = ms_folder.split(",")
    for ms_f in ms_folders:
        for root, dirs, files in os.walk(ms_folder):
            for file in files:
                if file.endswith(".raw"):
                    all_ms_files.append(os.path.join(root, file))

    print("done finding msms files")
    mgr_settings["transfer"]["ms_files"] = all_ms_files
    mgr_settings["transfer"]["ms_file_type"] = "thermo"

    if not grid_search:
        mgr_settings["transfer"]["model_output_folder"] = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        mgr_settings["transfer"]["lr_ms2"] = float(lr_ms2)
        mgr_settings["transfer"]["epoch_ms2"] = int(epoch_ms2)
        mgr_settings["transfer"]["batch_size_ms2"] = int(batch_size_ms2)
        mgr_settings["transfer"]["lr_rt_ccs"] = float(lr_rt)
        mgr_settings["transfer"]["epoch_rt_ccs"] = int(epoch_rt)
        mgr_settings["transfer"]["batch_size_rt_ccs"] = int(batch_size_rt)

        #write log file
        mgr_settings["log_file"] = output_folder + "/alphapeptdeep_tf" + str(datetime.datetime.now()).replace(" ", "_").replace(":", "_") \
                                   + "_lr-ms" + str(lr_ms2).split(".")[1] + "_epoch-ms" + str(epoch_ms2) + "_lr-rt" \
                                   + str(lr_rt).split(".")[1] + "_epoch-rt" + str(epoch_rt) + ".log"
        with open(mgr_settings["log_file"], "a") as f:
            for key in mgr_settings["transfer"]:
                f.write(key + ": " + str(mgr_settings["transfer"][key]) + "\n")

        print("Transfer learning beep boop")
        transfer_learn()
    else:
        lr_ms2_list = lr_ms2.split(",")
        epoch_ms2_list = epoch_ms2.split(",")
        batch_size_ms2_list = batch_size_ms2.split(",")
        lr_rt_list = lr_rt.split(",")
        epoch_rt_list = epoch_rt.split(",")
        batch_size_rt_list = batch_size_rt.split(",")

        for lr_ms2 in lr_ms2_list:
            for epoch_ms2 in epoch_ms2_list:
                for batch_size_ms2 in batch_size_ms2_list:
                    for lr_rt in lr_rt_list:
                        for epoch_rt in epoch_rt_list:
                            for batch_size_rt in batch_size_rt_list:
                                new_output_folder = output_folder + \
                                                    "lr-ms" + str(lr_ms2) + \
                                                    "epoch-ms" + str(epoch_ms2) + \
                                                    "batch-ms" + str(batch_size_ms2) + \
                                                    "lr-rt" + str(lr_rt) + \
                                                    "epoch-rt" + str(epoch_rt) + \
                                                    "batch-rt" + str(batch_size_rt)
                                if not os.path.exists(new_output_folder):
                                    os.makedirs(new_output_folder)
                                mgr_settings["transfer"]["model_output_folder"] = new_output_folder
                                mgr_settings["transfer"]["lr_ms2"] = float(lr_ms2)
                                mgr_settings["transfer"]["epoch_ms2"] = int(epoch_ms2)
                                mgr_settings["transfer"]["batch_size_ms2"] = int(batch_size_ms2)
                                mgr_settings["transfer"]["lr_rt_ccs"] = float(lr_rt)
                                mgr_settings["transfer"]["epoch_rt_ccs"] = int(epoch_rt)
                                mgr_settings["transfer"]["batch_size_rt_ccs"] = int(batch_size_rt)

                                # write log file
                                mgr_settings["log_file"] = new_output_folder + "/alphapeptdeep_tf" + str(datetime.datetime.now()).replace(" ",
                                                                                                                                      "_").replace(
                                    ":", "_") \
                                                           + "_lr-ms" + str(lr_ms2).split(".")[1] + "_epoch-ms" + str(epoch_ms2) + "_lr-rt" \
                                                           + str(lr_rt).split(".")[1] + "_epoch-rt" + str(epoch_rt) + ".log"
                                with open(mgr_settings["log_file"], "a") as f:
                                    for key in mgr_settings["transfer"]:
                                        f.write(key + ": " + str(mgr_settings["transfer"][key]) + "\n")

                                print("Transfer learning beep boop")
                                transfer_learn()