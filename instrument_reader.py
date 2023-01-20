from lxml import etree

def read_instrument(mzml):
    instrument = "Lumos"

    lumos_keys = ["LTQ", "Lumos", "Fusion", "Elite", "Velos", "Eclipse", "Tribrid"]
    qe_keys = ["QE", "Exactive", "Exploris"]
    sciex_keys = ["Sciex", "TripleTOF"]

    for event, elem in etree.iterparse(mzml):
        if "name" in elem.attrib.keys():
            name = elem.attrib["name"]
            if name.endswith(".d"):
                return "timsTOF"
            elif name == "ms level": #done checking all metadata and couldn't find anything
                return "Lumos"
            elif any(k in name for k in lumos_keys):
                return "Lumos"
            elif any(k in name for k in qe_keys):
                return "QE"
            elif any(k in name for k in sciex_keys):
                return "SciexTOF"
    return "Lumos"