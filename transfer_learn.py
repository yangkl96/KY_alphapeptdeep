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
    parser.add_argument('--ms_file_type', type = str, help='type of ms file; default = thermo', nargs='?', default="thermo")
    parser.add_argument('--min_score', type = int, help='minimum Andromeda score', nargs='?', default=150)
    parser.add_argument('--instrument', type = str, help='what mass spec; default = Lumos', nargs='?', default="Lumos")
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
    parser.add_argument('--lr', help='learning rate for both', default=0.0001)
    parser.add_argument('--epoch', help='number of epochs to train both', default=40)
    parser.add_argument('--batch_size', help='batch size for both', default=512)
    parser.add_argument('--grid_search', action=argparse.BooleanOptionalAction, help='whether to grid search over parameters separated by commas', default=False)
    parser.add_argument('--processing_only', action=argparse.BooleanOptionalAction, help='whether to just do preprocessing of files', default=False)

    args = parser.parse_args()

    if args.model_type == "phos":
        mask_mods = False

    sys.path.insert(1, args.peptdeep_folder)
    sys.path.insert(2, args.alphapept_folder)

    with open("peptdeep/constants/settings_type.txt", "w") as f:
        f.write(args.settings_type)
    from peptdeep.pipeline_api import transfer_learn
    from peptdeep.settings import global_settings
    import datetime

    #general settings
    mgr_settings = global_settings['model_mgr']
    mgr_settings["mask_modloss"] = mask_mods
    mgr_settings['transfer']['psm_type'] = 'maxqaunt'
    mgr_settings["transfer"]["grid_nce_search"] = False
    mgr_settings["model_type"] = args.model_type
    mgr_settings["external_ms2_model"] = args.external_ms2_model
    mgr_settings["grid_instrument"] = args.instrument

    if args.no_train_rt_ccs:
        mgr_settings["transfer"]['epoch_rt_ccs'] = 0
        mgr_settings["transfer"]['psm_num_to_train_rt_ccs'] = 0
        mgr_settings["transfer"]['psm_num_per_mod_to_train_rt_ccs'] = 0
    if args.no_train_ms2:
        mgr_settings["transfer"]['epoch_ms2'] = 0
        mgr_settings["transfer"]['psm_num_to_train_ms2'] = 0
        mgr_settings["transfer"]['psm_num_per_mod_to_train_ms2'] = 0

    #dict of dict for NCE
    NCE_dict = {}
    def enterNCEdict(x, root):
        scan_type_split = x["scan_type"].split("@")
        nce = scan_type_split[len(scan_type_split) - 1]
        i = 0
        while True:
            if nce[i].isnumeric():
                continue
            i += 1
        nce = nce[i:]

        NCE_dict[root + "_" + str(x["scan_number"])] = float(nce)

    print("Reading in files")
    all_psm_files = []
    psm_folders = args.psm_folder.split(",")
    for psm_f in psm_folders:
        print(psm_f)
        for root, dirs, files in os.walk(psm_f):
            for file in files:
                if file == "msms.txt":
                    if not args.skip_filtering:
                        print(os.path.join(root, file))
                        df = pd.read_csv(os.path.join(root, file), sep="\t")

                        df = df[df["Score"] >= args.min_score]
                        df = df[df["Fragmentation"].str.lower() == args.fragmentation.lower()]
                        df.to_csv(os.path.join(root, "msms_filter_" + args.fragmentation.lower() + ".txt"), sep = "\t", index=False)

                        #read in scanHeaderOnly.csv for NCE information
                        df = pd.read_csv(os.path.join(root, "scanHeaderOnly.csv"))
                        df.apply(lambda x: enterNCEdict(x, root), axis = 1)

                    all_psm_files.append(os.path.join(root, "msms_filter_" + args.fragmentation.lower() + ".txt"))
    mgr_settings["default_nce"] = NCE_dict

    print("done finding psm files")
    if (args.processing_only):
        print("processing done")
        sys.exit(0)
    mgr_settings["transfer"]["psm_files"] = all_psm_files
    mgr_settings["transfer"]["psm_type"] = "maxquant"

    all_ms_files = []
    ms_folders = args.ms_folder.split(",")
    for ms_f in ms_folders:
        print(ms_f)
        for root, dirs, files in os.walk(ms_f):
            for file in files:
                if args.ms_file_type == "thermo":
                    if file.endswith(".raw"):
                        all_ms_files.append(os.path.join(root, file))
                elif args.ms_file_type == "mgf":
                    if file.endswith(".mgf"):
                        all_ms_files.append(os.path.join(root, file))

    print("done finding msms files")
    mgr_settings["transfer"]["ms_files"] = all_ms_files
    mgr_settings["transfer"]["ms_file_type"] = args.ms_file_type

    mgr_settings["transfer"]["model_output_folder"] = args.output_folder
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    if not args.grid_search:
        mgr_settings["transfer"]["lr_ms2"] = float(args.lr_ms2)
        mgr_settings["transfer"]["epoch_ms2"] = int(args.epoch_ms2)
        mgr_settings["transfer"]["batch_size_ms2"] = int(args.batch_size_ms2)
        mgr_settings["transfer"]["lr_rt_ccs"] = float(args.lr_rt)
        mgr_settings["transfer"]["epoch_rt_ccs"] = int(args.epoch_rt)
        mgr_settings["transfer"]["batch_size_rt_ccs"] = int(args.batch_size_rt)

        #write log file
        mgr_settings["log_file"] = args.output_folder + "/alphapeptdeep_tf" + str(datetime.datetime.now()).\
            replace(" ", "_").replace(":", "_") + \
            "_lr-ms" + str(args.lr_ms2).split(".")[1] + "_epoch-ms" + str(args.epoch_ms2) + "_batch-ms" + str(args.batch_size_ms2) + \
            "_lr-rt" + str(args.lr_rt).split(".")[1] + "_epoch-rt" + str(args.epoch_rt) + "_batch-rt" + str(args.batch_size_rt) + ".log"
        with open(mgr_settings["log_file"], "a") as f:
            for key in mgr_settings["transfer"]:
                f.write(key + ": " + str(mgr_settings["transfer"][key]) + "\n")

        print("Transfer learning beep boop")
        transfer_learn()
    else:
        lr_list = args.lr.split(",")
        epoch_list = args.epoch.split(",")
        batch_size_list = args.batch_size.split(",")

        for lr in lr_list:
            for epoch in epoch_list:
                for batch_size in batch_size_list:
                    new_output_folder = args.output_folder + \
                                        "/lr" + str(lr) + \
                                        "epoch" + str(epoch) + \
                                        "batch" + str(batch_size)
                    if not os.path.exists(new_output_folder):
                        os.makedirs(new_output_folder)
                    mgr_settings["transfer"]["model_output_folder"] = new_output_folder
                    mgr_settings["transfer"]["lr_ms2"] = float(lr)
                    mgr_settings["transfer"]["epoch_ms2"] = int(epoch)
                    mgr_settings["transfer"]["batch_size_ms2"] = int(batch_size)
                    mgr_settings["transfer"]["lr_rt_ccs"] = float(lr)
                    mgr_settings["transfer"]["epoch_rt_ccs"] = int(epoch)
                    mgr_settings["transfer"]["batch_size_rt_ccs"] = int(batch_size)

                    # write log file
                    mgr_settings["log_file"] = new_output_folder + "/alphapeptdeep_tf" \
                                               + str(datetime.datetime.now()).replace(" ", "_").replace(":", "_") \
                                               + "_lr" + str(lr).split(".")[1] + "_epoch" + str(epoch) \
                                               + "_batch" + str(batch_size) + ".log"
                    with open(mgr_settings["log_file"], "a") as f:
                        for key in mgr_settings["transfer"]:
                            f.write(key + ": " + str(mgr_settings["transfer"][key]) + "\n")
                    print(mgr_settings["log_file"])
                    print("Transfer learning beep boop")
                    transfer_learn()