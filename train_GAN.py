import prepare.data_transformation as dt
import prepare.generate_train as gt
import os

GAN_database_folder_path = "./database/"
GAN_calculation_folder_path = "./calculation/"

##### 1. Generate lattice and sites graphs

##### 1.1. Generate 3D lattice voxel graphs
print("Generating 3D lattice voxel graphs")
dt.generate_lattice_graph(
    lattice_graph_path=GAN_calculation_folder_path + "original_lattice_graph/",
    atomlisttype="all element",
    a_list=None,
    data_path=GAN_database_folder_path + "geometries/",
    data_type="vasp",
)

##### 1.2. Generate 3D sites voxel graphs
print("Generating 3D sites voxel graphs")
dt.generate_sites_graph(
    sites_graph_path=GAN_calculation_folder_path + "original_sites_graph/",
    atomlisttype="all element",
    a_list=None,
    data_path=GAN_database_folder_path + "geometries/",
    data_type="vasp",
)

##### 2. Train the Autoencoder for 3D voxel graph and generate the encoded lattices and sites of the voxel graphs

##### 2.1. Generate the encoded lattice and save the trained model
print("Lattice Autoencoder running:")
import prepare.lattice_autoencoder as la

la.lattice_autocoder(
    lattice_graph_path=GAN_calculation_folder_path + "original_lattice_graph/",
    encoded_graph_path=GAN_calculation_folder_path + "original_encoded_lattice/",
    model_path=GAN_calculation_folder_path + "model/",
)

##### 2.2. Generate the encoded sites and save the trained model
print("Sites Autoencoder running:")
import prepare.sites_autoencoder as sa

sa.sites_autocoder(
    sites_graph_path=GAN_calculation_folder_path + "original_sites_graph/",
    encoded_graph_path=GAN_calculation_folder_path + "original_encoded_sites/",
    model_path=GAN_calculation_folder_path + "model/",
)


##### 3. Generate 2D graphs

##### 3.1. Directly combine the lattices and sites together
print("Generating 2D graphs")
dt.generate_crystal_2d_graph(
    encodedgraphsavepath=GAN_calculation_folder_path + "original_encoded_sites/",
    encodedlatticesavepath=GAN_calculation_folder_path + "original_encoded_lattice/",
    crystal_2d_graph_path=GAN_calculation_folder_path + "original_crystal_2d_graphs/",
)

##### 3.2. Transform the 2D graph into square shapes and combine them into one file in order to generate the training file for the GAN
print("Converting 2D graphs into square shapes and combining all the training data")
gt.generate_train_X(
    encodedsavepath=GAN_calculation_folder_path + "original_crystal_2d_graphs/",
    X_train_savepath=GAN_calculation_folder_path,
    X_train_name="train_X.npy",
)

##### 4. Train constraints

##### 4.1. Prepare data to train formation energy and band gap reg networks
import prepare.data_for_constrains as dfc
print("Generate formation energy reg constrain data")
dfc.get_formation_energy_constrain_reg_train_y(data_path=GAN_database_folder_path+'properties/formation_energy/',directory=GAN_calculation_folder_path+'train_formation_energy_reg.npy')
print("Generate band gap reg constrain data")
dfc.get_band_gap_constrain_reg_train_y(data_path=GAN_database_folder_path+'properties/band_gap/',directory=GAN_calculation_folder_path+'train_band_gap_reg.npy')

##### 4.2. Train the formation energy and band gap reg networks
import prepare.constrain_reg as con_reg

print("Train formation energy reg constrain")
constrainfe = con_reg.constrain()
constrainfe.train(
    X_npy=GAN_calculation_folder_path + "train_X.npy",
    y_npy=GAN_calculation_folder_path + "train_formation_energy_reg.npy",
    model_path=GAN_calculation_folder_path + "model/formation_energy_reg.h5",
    epochs=10001,
    batch_size=64,
    save_interval=5000,
)
print("Train band gap reg constrain")
constrainbg = con_reg.constrain()
constrainbg.train(
    X_npy=GAN_calculation_folder_path + "train_X.npy",
    y_npy=GAN_calculation_folder_path + "train_band_gap_reg.npy",
    model_path=GAN_calculation_folder_path + "model/band_gap_reg.h5",
    epochs=10001,
    batch_size=64,
    save_interval=5000,
)

##### 5. Train the CCDCGAN model
import gan.ccdcgan as gan

ccdcgan = gan.CCDCGAN()
ccdcgan.train(
    epochs=100000,
    save_interval=5000,
    GAN_calculation_folder_path=GAN_calculation_folder_path,
    X_train_name="train_X.npy",
)

##### 6. Remove unnecessary files

command = "rm -r ./calculation/original_sites_graph/;"
os.system(command)
command = "rm -r ./calculation/original_lattice_graph/;"
os.system(command)

print("Finished!")
