import subprocess, os
from mp_api.client import MPRester
from ase.io import read
import json

f = open("./config.json")
config = json.load(f)

def filter_by_fEV():
    theoretically_metastable = 0
    synthesizable = 0

    if not os.path.exists("./theoretically_metastable"):
        os.makedirs("./theoretically_metastable")

    if not os.path.exists("./synthesizable"):
        os.makedirs("./synthesizable")

    for filename in os.listdir("./save/generated_crystal_for_check"):
        x = subprocess.check_output("python ./alignn/alignn/pretrained.py --model_name jv_formation_energy_peratom_alignn --file_format cif --file_path ./save/generated_crystal_for_check/" + filename, shell=True)
        formation_energy = float(((str(x).split("[")[2]).split("]"[0]))[0])
        print(abs(formation_energy))
        if (abs(formation_energy) < 0.08):
            synthesizable+=1
            print(filename)
            os.system("cp ./save/generated_crystal_for_check/" + filename + " ./synthesizable/" + filename)
        elif (abs(formation_energy) < 0.2):
            theoretically_metastable+=1
            print(filename)
            os.system("cp ./save/generated_crystal_for_check/" + filename + " ./theoretically_metastable/" + filename)


    print("Synthesizable: " + str(synthesizable))
    print("Theoretically Metastable: " + str(theoretically_metastable))

def identify_duplicates():
    for file in os.listdir("./save/generated_crystal_for_check"):
        atoms = read("./save/generated_crystal_for_check/" + file,format = "cif")
        with MPRester(config["MP_API_KEY"]) as mpr:
            docs = mpr.materials.summary.search(formula=atoms.get_chemical_formula())
            print(atoms.get_chemical_formula())
            if docs:
                print("Removing duplicate " + file)
                os.system("rm ./save/generated_crystal_for_check/" + file) 

           

identify_duplicates()
filter_by_fEV()
