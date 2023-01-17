import xml.etree.ElementTree as ET

def read_instrument(mzml):
    instrument = "Lumos"

    # read in mzml
    #"Z:/data/HLA/Klaeger_2021_PXD027165/AG20210507_SR_HLA_A0201_W632_2IPs_Fxn01.mzML"
    mzmlFile = ET.parse(mzml)

    lumos_keys = ["LTQ", "Lumos", "Fusion", "Elite", "Velos", "Eclipse", "Tribrid"]
    qe_keys = ["QE", "Exactive", "Exploris"]
    sciex_keys = ["Sciex", "TripleTOF"]
    root = mzmlFile.getroot()
    if root[0].attrib["id"].endswith(".d"):
        instrument = "timsTOF"
    else:
        for entry in root[0]:
            if "referenceableParamGroupList" in entry.tag:
                for entry2 in entry[0]:
                    if "name" in entry2.keys():
                        my_value = entry2.attrib["name"]
                        if any(k in my_value for k in lumos_keys):
                            instrument = "Lumos"
                            break
                        elif any(k in my_value for k in qe_keys):
                            instrument = "QE"
                            break
                        elif any(k in my_value for k in sciex_keys):
                            instrument = "SciexTOF"
                            break
            elif "instrumentConfigurationList" in entry.tag:
                for entry2 in entry[0]:
                    if "name" in entry2.keys():
                        my_value = entry2.attrib["name"]
                        if any(k in my_value for k in lumos_keys):
                            instrument = "Lumos"
                            break
                        elif any(k in my_value for k in qe_keys):
                            instrument = "QE"
                            break
                        elif any(k in my_value for k in sciex_keys):
                            instrument = "SciexTOF"
                            break
    return(instrument)