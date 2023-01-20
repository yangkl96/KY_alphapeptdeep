#need to download alphapeptdeep from github
if __name__ == '__main__':
    print("Loading packages")
    import os
    import pandas as pd
    import argparse
    import sys
    from collections import Counter
    from xml.sax import saxutils
    import shutil
    from time import time

    parser = argparse.ArgumentParser(description='Training transfer model')
    parser.add_argument('psm_folder', type = str, help='folder for PSMs')
    parser.add_argument('ms_folder', type = str, help='folder for mass spec output files')
    parser.add_argument('output_folder', type = str, help='folder for saving transfer model')
    parser.add_argument('fragmentation', type = str, help='fragmentation method')

    parser.add_argument('--peptdeep_folder', type=str, help='folder for peptdeep', nargs='?', default=".")
    parser.add_argument('--psm_type', type = str, help='type of PSMs; default = msfragger_pepxml', nargs='?', default="msfragger_pepxml")
    parser.add_argument('--ms_file_type', type = str, help='type of ms file; default = mgf', nargs='?', default="mgf")
    parser.add_argument('--min_score', type = int, help='minimum Andromeda score', nargs='?', default=150)
    parser.add_argument('--expect', type=float, help='expectation value for MSFragger pepxml filtering', nargs='?', default=0.0001)
    parser.add_argument('--instrument', type = str, help='what mass spec; default: search in mzml file', nargs='?', default="TBD")
    parser.add_argument('--model_type', type = str, help='generic, phos, hla, or digly; default = generic',
                        nargs='?', default="generic") #default
    parser.add_argument('--external_ms2_model', type=str, help='path to external ms2 model', nargs='?', default = "")
    parser.add_argument('--no_train_rt_ccs', action=argparse.BooleanOptionalAction,
                        help='whether to train rt and ccs. Adding this flag will not train RT or CCS models')
    parser.add_argument('--no_train_ms2', action=argparse.BooleanOptionalAction,
                        help='whether to train ms2. Adding this flag will not train ms2 models')
    #parser.add_argument('--alphapept_folder', type=str, help='folder for alphapept', nargs='?', default="./alphapept/")
    parser.add_argument('--settings_type', type=str,
                        help='settings.yaml to use. Will use fragmentation parameter if this is not specified',
                        default="TBD")
    parser.add_argument('--skip_filtering', action=argparse.BooleanOptionalAction, #may decide to make this for internal use only
                        help='whether to skip some writing filtered pepxmls. ' +
                             'Only use if rerunning with same parameters as last run')
    parser.add_argument('--mask_mods', action=argparse.BooleanOptionalAction, help='whether to mask modloss fragments')
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
    #parser.add_argument('--processing_only', action=argparse.BooleanOptionalAction, help='whether to just do preprocessing of files', default=False)
    parser.add_argument('--num_models', type = int, help='number of splits/models to train', default = 1)
    parser.add_argument('--fragger', type = str, help = 'path to fragger.params from MSFragger', default = None)
    parser.add_argument('--modification_tsv', type = str, help = 'path to modification_alphapeptdeep.tsv', default = None)
    parser.add_argument('--diff_nce', action=argparse.BooleanOptionalAction,
                        help = 'whether nce is different within each mass spec run')

    args = parser.parse_args()

    if args.psm_type == "msfragger_pepxml" and args.fragger and args.modification_tsv:
        print("Detecting Unimod PTMs from MSFragger search results")
        import alphabase
        from alphabase.yaml_utils import load_yaml, save_yaml
        import shutil
        import msfragger_ptms

        alphabase_path = os.path.dirname(alphabase.__file__)
        psm_reader_yaml_path = os.path.join(alphabase_path, "io", "psm_reader", "psm_reader.yaml")

        # dump this yaml to tmp yaml
        if os.path.isfile(os.path.join(alphabase_path, "io", "psm_reader", "tmp_psm_reader.yaml")):
            shutil.copy(os.path.join(alphabase_path, "io", "psm_reader", "tmp_psm_reader.yaml"),
                        os.path.join(alphabase_path, "io", "psm_reader", "psm_reader.yaml"))
        else:
            shutil.copy(os.path.join(alphabase_path, "io", "psm_reader", "psm_reader.yaml"),
                        os.path.join(alphabase_path, "io", "psm_reader", "tmp_psm_reader.yaml"))

        # if there's any PTMs to add from fragger.params, add them here
        psm_reader_yaml = load_yaml(psm_reader_yaml_path)  # load yaml variable

        # read in fragger.params and process
        PTM_masses = msfragger_ptms.read_fragger_params(args.fragger)
        PTM_names = msfragger_ptms.search_modification_tsv(args.modification_tsv, PTM_masses)
        for name in PTM_names:
            psm_reader_yaml['msfragger_pepxml']['mass_mapped_mods'].append(name)
        save_yaml(psm_reader_yaml_path, psm_reader_yaml)

    if args.model_type == "phos":
        args.mask_mods = False
    if args.settings_type == "TBD":
        args.settings_type = args.fragmentation

    sys.path.insert(1, args.peptdeep_folder)
    #sys.path.insert(2, args.alphapept_folder)
    with open("peptdeep/constants/settings_type.txt", "w") as f:
        f.write(args.settings_type)
    from peptdeep.pipeline_api import transfer_learn
    from peptdeep.settings import global_settings
    import instrument_reader
    import datetime
    from pyteomics import mzml, pepxml

    #general settings
    mgr_settings = global_settings['model_mgr']
    mgr_settings["mask_modloss"] = args.mask_mods
    mgr_settings['transfer']['psm_type'] = 'maxqaunt'
    mgr_settings["transfer"]["grid_nce_search"] = False
    mgr_settings["model_type"] = args.model_type
    mgr_settings["external_ms2_model"] = args.external_ms2_model
    if args.instrument.lower() not in ["lumos", "qe", "sciextof", "timstof", "tbd"]:
        print("Please set instrument to lumos, qe, sciextof, or timstof")
        sys.exit(0)
    if args.psm_type == "maxquant":
        args.num_models = 1
    if args.skip_filtering and args.diff_nce:
        print("Skip filtering is only possible if diff_nce is false. Setting skip filtering to False")
        args.skip_filtering = False

    if args.no_train_rt_ccs:
        mgr_settings["transfer"]['epoch_rt_ccs'] = 0
        mgr_settings["transfer"]['psm_num_to_train_rt_ccs'] = 0
        mgr_settings["transfer"]['psm_num_per_mod_to_train_rt_ccs'] = 0
    if args.no_train_ms2:
        mgr_settings["transfer"]['epoch_ms2'] = 0
        mgr_settings["transfer"]['psm_num_to_train_ms2'] = 0
        mgr_settings["transfer"]['psm_num_per_mod_to_train_ms2'] = 0

    rawToPath = {}
    all_ms_files = []
    ms_folders = args.ms_folder.split(",")
    for ms_f in ms_folders:
        for root, dirs, files in os.walk(ms_f):
            for file in files:
                if args.ms_file_type == "thermo":
                    if file.endswith(".raw"):
                        all_ms_files.append(os.path.join(root, file))
                elif args.ms_file_type == "mgf":
                    if file.endswith(".mgf"):
                        all_ms_files.append(os.path.join(root, file))
                elif args.ms_file_type.lower() == "mzml":
                    if file.lower().endswith(".mzml"):
                        all_ms_files.append(os.path.join(root, file))

                splitFile = file.split(".")
                splitFile = ".".join(splitFile[0: len(splitFile) - 1])
                rawToPath[splitFile] = root

    mgr_settings["transfer"]["ms_files"] = all_ms_files
    mgr_settings["transfer"]["ms_file_type"] = args.ms_file_type

    #dicts for NCE and instruments
    #this makes skipping filtering difficult
    NCE_dict = {}
    instrument_dict = {}
    def enterNCEdict(x, mzmlFileDict, number_identifier = "start_scan"):
        entry = mzmlFileDict[x[number_identifier]]
        scan_type_split = entry["scanList"]["scan"][0]["filter string"].split("@")
        nce = ""
        for s in scan_type_split[1:]:
            if args.fragmentation.lower() == "etd":
                if "etd" in s:
                    nce = s.split(" ")[0]
            if args.fragmentation.lower() == "hcd" or args.fragmentation.lower() == "ethcd":
                if "hcd" in s:
                    nce = s.split(" ")[0]
        for i in range(len(nce)):
            if nce[i].isnumeric():
                nce = nce[i:]
                break

        if "spectrum" in x.index:
            xsplit = x["spectrum"].split(".")
            NCE_dict[".".join(xsplit[0:len(xsplit) - 3]) + "_" + str(x[number_identifier])] = float(nce)
        else:
            NCE_dict[x["Raw file"] + "_" + str(x[number_identifier])] = float(nce)
        return nce

    print("Reading in psm files")
    all_psm_files = {}
    psm_folders = args.psm_folder.split(",")
    i = 0

    for psm_f in psm_folders:
        if args.psm_type == "msfragger_pepxml":
            dfs = []
            # need to get filtered pepxml file for training
            # code adapted from pyteomics.pepxmltk

            peptide_counter = Counter() #only want to use peptides that appear once

            for root, dirs, files in os.walk(psm_f):
                for file in files:
                    if file.endswith("pepXML") and not file.endswith("filter.pepXML"):
                        if not args.skip_filtering:
                            #count how many times each modified peptide-charge combo appears,
                            #so as to only get unique ones for training
                            #df = pepxml.filter_df(os.path.join(root, file), fdr=1, decoy_prefix="rev_", correction=1)
                            print("\tLoading " + os.path.join(root, file))
                            df = pepxml.DataFrame(os.path.join(root, file))
                            df.loc[:, "pep_charge"] = df["modified_peptide"] + df["assumed_charge"].astype(str)
                            peptide_counter.update(df["pep_charge"])
                            dfs.append(df)
                        else:
                            df = pepxml.DataFrame(os.path.join(root, file))
                            dfs.append(df.iloc[0:2])

            file_num = -1
            for root, dirs, files in os.walk(psm_f):
                for file in files:
                    if file.endswith("pepXML") and not file.endswith("filter.pepXML"):
                        # HOW TO DEAL WITH DIFFERENT FRAGMENTATIONS? DO NOT ALLOW AT FIRST, BUT HOW?
                        file_num += 1

                        # get the PSMs below some expect
                        #df = pepxml.filter_df(os.path.join(root, file), fdr=1, decoy_prefix="rev_", correction=1)
                        df = dfs[file_num].copy()
                        if not args.skip_filtering:
                            df = df[df["expect"] < args.expect]
                            df.loc[:, "counter_id"] = df.apply(lambda x: x["modified_peptide"] +
                                                                  str(x["assumed_charge"]), axis = 1)
                            df.loc[:, "counts"] = df.apply(lambda x: peptide_counter[x["counter_id"]], axis = 1)
                            df = df[df["counts"] == 1]
                        df["nce"] = ""

                        scan_num_set = set(df["start_scan"].values)

                        print("Adding nce and instrument information for " + file)

                        #check how many PSMs there are
                        if not args.skip_filtering:
                            print(str(df.shape[0]) + " PSMs")
                        if df.shape[0] == 0: #WHAT TO DO IF NO ENTRIES IN END? BECOME LESS RESTRICTIVE??
                            print("Skipping to next file")
                            continue

                        #extracting info from mzml
                        base_name = file.replace(".pepXML", "")
                        mzml_name = rawToPath[base_name] + "/" + base_name + ".mzML"
                        print("\tLoading " + mzml_name)
                        if args.diff_nce:
                            try:
                                mzmlFile = mzml.read(mzml_name, use_index=True) #can not use index if skip filtering
                            except:
                                print(mzml_name + " does not exist, please add to ms folder")
                                sys.exit(1)
                            mzmlFileDict = {entry["index"] + 1: entry for entry in mzmlFile.map()
                                            if entry["index"] + 1 in scan_num_set}

                            df["nce"] = df.apply(lambda x: enterNCEdict(x, mzmlFileDict), axis=1)
                        else:
                            mzmlFileDict = {}
                            try:
                                mzmlFile = mzml.read(mzml_name)
                            except:
                                print(mzml_name + " does not exist, please add to ms folder")
                                sys.exit(1)

                            for entry in mzmlFile:
                                if entry["index"] + 1 in scan_num_set:
                                    mzmlFileDict[entry["index"] + 1] = entry
                                    mini_df = df[df["start_scan"] == entry["index"] + 1].copy()
                                    df["nce"] = enterNCEdict(mini_df.iloc[0], mzmlFileDict)
                                    break

                        # read in instruments
                        raw_name = mzml_name.replace(".mzml", "").replace(".mzML", "")
                        raw_name = raw_name.split("/")[-1].split("\"")[-1]
                        if args.instrument.lower() == "tbd":
                            instrument_dict[raw_name] = instrument_reader.read_instrument(mzml_name)
                        else:
                            instrument_dict[raw_name] = args.instrument
                        print("\tSetting instrument to " + instrument_dict[raw_name])

                        #add to final psm list
                        #split PSMs into subgroups
                        psms_list = []
                        for i in range(args.num_models):
                            new_pepxml = os.path.join(root, file).replace("pepXML", str(i) + "filter.pepXML")
                            if not args.skip_filtering:
                                print("\tWriting new pepxml at " + new_pepxml)

                                psms = set()  # for holding the best PSMs, those that will be used for transfer learning
                                if args.num_models == 1:
                                    psms.update(psm for psm in df['spectrum'])
                                else:
                                    psms.update(psm for psm, scan_num in zip(df['spectrum'], df['start_scan'])
                                                if scan_num % args.num_models != i)
                                psms_list.append(psms)

                                #write out pepxml
                                with open(new_pepxml, 'w') as output_file:
                                    unlocked = True

                                    with open(os.path.join(root, file)) as input_file:
                                        lines = input_file.readlines()
                                    for line in lines:
                                        if '<spectrum_query' in line:
                                            check_line = saxutils.unescape(line.split('spectrum="')[1].split('" ')[0],
                                                                           {'&quot;': '"'})
                                            if check_line not in psms:
                                                unlocked = False
                                            else:
                                                unlocked = True
                                        if unlocked:
                                            output_file.write(line)
                                        if '</spectrum_query>' in line:
                                            unlocked = True

                            #add files to separate indexes of list
                            if i not in all_psm_files.keys():
                                all_psm_files[i] = []
                            all_psm_files[i].append(new_pepxml)

        elif args.psm_type == "maxquant":
            all_psm_files[0] = []
            for root, dirs, files in os.walk(psm_f):
                for file in files:
                    if file == "msms.txt":
                        if not args.skip_filtering:
                            print(str(i) + ", " + os.path.join(root, file))
                            i += 1
                            df = pd.read_csv(os.path.join(root, file), sep="\t", low_memory = False)

                            #get only PSMs with relevant mzml files
                            df = df[df["Raw file"].isin(rawToPath.keys())]
                            df = df[df["Score"] >= args.min_score]
                            df = df[df["Fragmentation"].str.lower() == args.fragmentation.lower()]
                            df.reset_index(drop = True, inplace = True)

                            #read in mzmls for NCE information
                            total_columns = list(df.columns)
                            total_columns.append("nce")
                            total_df = pd.DataFrame(columns = total_columns)
                            for raw in set(df["Raw file"].unique()):
                                mzmlRoot = rawToPath[df["Raw file"][0]].replace("mgf", "mzml")
                                print("Reading " + mzmlRoot + "/" + raw + ".mzML")
                                try:
                                    mzmlFile = mzml.read(mzmlRoot + "/" + raw + ".mzML",
                                                         use_index = True)
                                except:
                                    print(mzmlRoot + "/" + raw + ".mzML does not exist")
                                    continue

                                mini_df = df[df["Raw file"] == raw].copy()
                                scan_num_set = set(mini_df["Scan number"].values)

                                print("Extracting nce and instrument information")

                                #extract NCE info for each row using scan number minus 1 to get index
                                print(str(mini_df.shape[0]) + " PSMs")
                                mzmlFileDict = {}
                                for entry in mzmlFile:
                                    if entry["index"] + 1 in scan_num_set:
                                        mzmlFileDict[entry["index"] + 1] = entry
                                mini_df["nce"] = mini_df.apply(
                                    lambda x: enterNCEdict(x, mzmlFileDict, number_identifier="Scan number"), axis = 1)

                                #read in instruments
                                if args.instrument.lower() == "tbd":
                                    instrument_dict[raw] =\
                                        instrument_reader.read_instrument(mzmlRoot + "/" + raw + ".mzML")
                                else:
                                    instrument_dict[raw] = args.instrument

                                #add df back together
                                total_df = pd.concat([total_df, mini_df])

                            #remove all but the best PSMs
                            df = total_df
                            df.sort_values(by = "Score", ascending = False, inplace = True)
                            df.drop_duplicates(subset = ["Modified sequence", "Charge", "nce"], inplace = True)
                            df = df.reset_index(drop = True)
                            df.to_csv(os.path.join(root, "msms_filter_" + args.fragmentation.lower() + ".txt"), sep = "\t", index=False)

                        all_psm_files[0].append(os.path.join(root, "msms_filter_" + args.fragmentation.lower() + ".txt"))
                        print("Done reading " + mzmlRoot + "/" + raw + ".mzML")

    if not args.diff_nce:
        for val in NCE_dict.values():
            mgr_settings["default_nce"] = val
            break
    else:
        mgr_settings["default_nce"] = NCE_dict
    mgr_settings["default_instrument"] = instrument_dict

    #if (args.processing_only):
    #    print("processing done")
    #    sys.exit(0)
    mgr_settings["transfer"]["psm_type"] = args.psm_type

    #appropriately name the sub folders
    for i in range(args.num_models):
        mgr_settings["transfer"]["psm_files"] = all_psm_files[i]
        if args.num_models == 1:
            mgr_settings["transfer"]["model_output_folder"] = args.output_folder
        else:
            mgr_settings["transfer"]["model_output_folder"] = args.output_folder + "_" + str(i)
        if not os.path.exists(mgr_settings["transfer"]["model_output_folder"]):
            os.makedirs(mgr_settings["transfer"]["model_output_folder"])

        if not args.grid_search:
            mgr_settings["transfer"]["lr_ms2"] = float(args.lr_ms2)
            mgr_settings["transfer"]["epoch_ms2"] = int(args.epoch_ms2)
            mgr_settings["transfer"]["batch_size_ms2"] = int(args.batch_size_ms2)
            mgr_settings["transfer"]["lr_rt_ccs"] = float(args.lr_rt)
            mgr_settings["transfer"]["epoch_rt_ccs"] = int(args.epoch_rt)
            mgr_settings["transfer"]["batch_size_rt_ccs"] = int(args.batch_size_rt)

            #write log file
            mgr_settings["log_file"] = mgr_settings["transfer"]["model_output_folder"] + "/alphapeptdeep_tf" + str(datetime.datetime.now()).\
                replace(" ", "_").replace(":", "_") + \
                "_lr-ms" + str(args.lr_ms2).split(".")[1] + "_epoch-ms" + str(args.epoch_ms2) + "_batch-ms" + str(args.batch_size_ms2) + \
                "_lr-rt" + str(args.lr_rt).split(".")[1] + "_epoch-rt" + str(args.epoch_rt) + "_batch-rt" + str(args.batch_size_rt) + ".log"
            with open(mgr_settings["log_file"], "a") as f:
                for key in mgr_settings["transfer"]:
                    f.write(key + ": " + str(mgr_settings["transfer"][key]) + "\n")

            print("Transfer learning beep boop")
            transfer_learn()
        else:
            if type(args.lr) == str:
                lr_list = args.lr.split(",")
            else:
                lr_list = [args.lr]
            if type(args.epoch) == str:
                epoch_list = args.epoch.split(",")
            else:
                epoch_list = [args.epoch]
            if type(args.batch_size) == str:
                batch_size_list = args.batch_size.split(",")
            else:
                batch_size_list = [args.batch_size]

            best_val_ccs = [sys.maxsize, ""]
            best_val_rt = [sys.maxsize, ""]
            best_val_ms2 = [sys.maxsize, ""]

            for lr in lr_list:
                for epoch in epoch_list:
                    for batch_size in batch_size_list:
                        if args.num_models == 1:
                            mgr_settings["transfer"]["model_output_folder"] = args.output_folder
                        else:
                            mgr_settings["transfer"]["model_output_folder"] = args.output_folder + "_" + str(i)
                        new_output_folder = mgr_settings["transfer"]["model_output_folder"] + \
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
                        val_dict = transfer_learn()

                        #be able to get an output from transfer_learn that maps model to best validation
                        for key, val in val_dict.items():
                            if key == "ccs":
                                if val < best_val_ccs[0]:
                                    best_val_ccs[0] = val
                                    best_val_ccs[1] = mgr_settings["transfer"]["model_output_folder"]
                            elif key == "rt":
                                if val < best_val_rt[0]:
                                    best_val_rt[0] = val
                                    best_val_rt[1] = mgr_settings["transfer"]["model_output_folder"]
                            elif key == "ms2":
                                if val < best_val_ms2[0]:
                                    best_val_ms2[0] = val
                                    best_val_ms2[1] = mgr_settings["transfer"]["model_output_folder"]

            #copy best model to output folder
            target = args.output_folder + "_" + str(i)

            for origin, model_type in zip([best_val_ccs[1], best_val_rt[1], best_val_ms2[1]],
                                          ["ccs", "rt", "ms2"]):
                if origin == "":
                    continue
                files = os.listdir(origin)
                for file_name in files:
                    if file_name.startswith(model_type):
                        shutil.copy(os.path.join(origin, file_name), os.path.join(target, file_name))

    if args.psm_type == "msfragger_pepxml":
        #copy tmp_psm_reader back to psm_reader
        shutil.copy(os.path.join(alphabase_path, "io", "psm_reader", "tmp_psm_reader.yaml"),
                    os.path.join(alphabase_path, "io", "psm_reader", "psm_reader.yaml"))