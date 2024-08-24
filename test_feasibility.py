import subprocess, os
from mp_api.client import MPRester
from ase.io import read
import json
from pymatgen_testing import evaluate_cif_files
from custom_test import predict_and_save_to_csv
import pandas as pd

f = open("./config.json")
config = json.load(f)

def identify_duplicates():
    for file in os.listdir("./save/generated_crystal_for_check"):
        atoms = read("./save/generated_crystal_for_check/" + file,format = "cif")
        with MPRester(config["MP_API_KEY"]) as mpr:
            docs = mpr.materials.summary.search(formula=atoms.get_chemical_formula())
            print(atoms.get_chemical_formula())
            if docs:
                print("Removing duplicate " + file)
                os.system("rm ./save/generated_crystal_for_check/" + file) 

if __name__ == "__main__":
    identify_duplicates()
    cif_directory = "./save/generated_crystal_for_check/"
    output_csv = "./output.csv"
    predict_and_save_to_csv(cif_directory, output_csv)
    df = pd.read_csv("./output.csv")

    print(df[(df['Formation Energy (eV/atom)'] < -1.5) &
                 (df['Energy Above Hull (eV/atom)'] < 0.08)])

    filtered_df = df[(df['Formation Energy (eV/atom)'] < -1.5) &
                 (df['Energy Above Hull (eV/atom)'] < 0.08)]
    output_path = './filtered_structures_unconditional.csv'
    filtered_df.to_csv(output_path, index=False)
    evaluate_cif_files()
