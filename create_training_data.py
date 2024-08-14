from mp_api.client import MPRester
import pandas as pd
from ase import io
import numpy
import os
import glob
from multiprocessing.pool import ThreadPool as Pool
import json

## Configure CSV file and load config
f = open("./config.json")
config = json.load(f)
MP_API_KEY = config["MP_API_KEY"]
data = pd.read_csv(config["CSV_FILE"])
cifs = data["cif"].tolist()
mpids = data["material_id"].to_list()

## Clean directories if they exist
files = glob.glob('./temp/*') 
for f in files:
    os.remove(f)
files2 = glob.glob('./database/geometries/*')
for g in files2:
    os.remove(g)
files3 = glob.glob('./database/properties/formation_energy/*')
for h in files3:
    os.remove(h) 
files4 = glob.glob('./database/properties/band_gap/*')
for i in files4:
    os.remove(i) 

## Create required directories
os.makedirs("./temp", exist_ok=True)
os.makedirs(os.path.join("database", "geometries"), exist_ok=True)
os.makedirs(os.path.join("database", "properties", "formation_energy"), exist_ok=True)
os.makedirs(os.path.join("database", "properties", "band_gap"), exist_ok=True)

## Convert each CIF file into a VASP file and store formation energy and band gap properties in NPY files
def worker_process(index, cif):
    try:
        with open("./temp/" + str(mpids[index]) + ".cif", "w+") as writer:
            writer.write(cif)
        atoms = io.read("./temp/" + str(mpids[index]) + ".cif")    
        atoms.write("./database/geometries/" + str(mpids[index]) + ".vasp", format='vasp')
        with MPRester(MP_API_KEY) as mpr:
            doc = mpr.materials.summary.search(material_ids=mpids[index])
            if 0 in range(len(doc)):
                numpy.save("./database/properties/formation_energy/" + str(mpids[index]) + "_formation_energy.npy", doc[0].formation_energy_per_atom)
                numpy.save("./database/properties/band_gap/" + str(mpids[index]) + "_band_gap.npy", doc[0].band_gap)
            else: 
                print("Skipping " + str(mpids[index]))
                os.remove("./database/geometries/" + str(mpids[index]) + ".vasp")
    except Exception as e:
        print(e)
pool = Pool(config["POOL_SIZE"])
for index, cif in enumerate(cifs):
    pool.apply_async(worker_process, (index, cif,))
pool.close()
pool.join()

## Delete temp directory
files = glob.glob('./temp/*') 
for z in files:
    os.remove(z)
os.rmdir("./temp")