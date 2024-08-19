import subprocess, os
from mp_api.client import MPRester
from ase.io import read
import json
from pymatgen_testing import evaluate_cif_files

f = open("./config.json")
config = json.load(f)

def filter_by_fEV():
    stable = 0

    if not os.path.exists("./stable"):
        os.makedirs("./stable")

    for filename in os.listdir("./save/generated_crystal_for_check"):
        x = subprocess.check_output("python ./alignn/alignn/pretrained.py --model_name jv_formation_energy_peratom_alignn --file_format cif --file_path ./save/generated_crystal_for_check/" + filename, shell=True)
        formation_energy = float(((str(x).split("[")[2]).split("]"[0]))[0])
        print(formation_energy)
        if (formation_energy < 0):
            stable+=1
            print(filename)
            os.system("cp ./save/generated_crystal_for_check/" + filename + " ./stable/" + filename)
    print("Stable: " + str(stable))

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
    filter_by_fEV()
    evaluate_cif_files()
