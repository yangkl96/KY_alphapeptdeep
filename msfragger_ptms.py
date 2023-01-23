import pandas as pd

def read_fragger_params(fragger):
    PTM_masses = set()
    AA_masses = {}
    with open(fragger, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("mass_offsets"):
                mass_offsets = line.split("mass_offsets")[1].split("=")[1].replace("\t", " ").strip().split(" ")[0]
                if "/" in mass_offsets:
                    mass_offsets = mass_offsets.split("/")
                else:
                    mass_offsets = [mass_offsets]
                for offset in mass_offsets:
                    PTM_masses.add(float(offset))
            elif line.startswith("add_") and not line.startswith("add_topN_complementary"):
                line_split = line.split("add_")[1].split("=")
                AA = line_split[0].replace("\t", " ").strip()
                mass = line_split[1].replace("\t", " ").strip()
                if "term" not in AA:
                    AA = AA[0]
                AA_masses[AA] = [float(mass)]
            elif line.startswith("allow_multiple_variable_mods_on_residue"):
                val = line.split("=")[1].replace("\t", " ").strip()
                if not val.startswith("0"):
                    print("Currently does not support multiple variable mods on each residue. " +
                          "Some PTMs may not be considered during prediction.")

        for line in lines: #now ok to add variable mod masses
            if line.startswith("variable_mod"):
                line_split = line.split("=")[1].replace("\t", " ").strip().split(" ")
                mass = float(line_split[0])
                AAs = line_split[1]
                skip = False
                for i in range(len(AAs)):
                    if skip:
                        skip = False
                        continue
                    AA = AAs[i]
                    if AA == "[":
                        AA = "Nterm_protein"
                        skip = True
                    elif AA == "]":
                        AA = "Cterm_protein"
                        skip = True
                    elif AA == "n":
                        AA = "Nterm_peptide"
                        skip = True
                    elif AA == "c":
                        AA = "Cterm_peptide"
                        skip = True
                    else:
                        pass #all other amino acids

                    AA_mass = AA_masses[AA]
                    AA_mass.append(AA_mass[0] + mass)
                    AA_masses[AA] = AA_mass

        for _, mass in AA_masses.items():
            for m in mass:
                PTM_masses.add(m)

    return PTM_masses

def search_modification_tsv(mod_tsv, PTM_masses, mass_tol = 0.1):
    PTM_names = set()
    df = pd.read_csv(mod_tsv, sep = "\t", usecols = ["mod_name", "unimod_mass"])
    for mass in PTM_masses:
        min_mass = mass - mass_tol
        max_mass = mass + mass_tol
        sub_df = df[(df["unimod_mass"] > min_mass) & (df["unimod_mass"] < max_mass)]
        PTM_names.update(list(sub_df["mod_name"]))

    return PTM_names