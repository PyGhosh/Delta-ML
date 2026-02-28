import os
import sys
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem

# === Configuration ===
xyz_dir = "xtb_opt"
com_dir = "sTDA_calculation"

# === Read Input file ===
if len(sys.argv) < 2:
    print("Usage: python3 xtb_std.py <input_file.smi>")
    sys.exit(1)

smiles_file = sys.argv[1].strip()

#smiles_file = input('Enter the name of the input file in .smi format: ').strip()

# === Helper: Write XYZ file from RDKit ===
def write_xyz(mol, filename):
    conf = mol.GetConformer()
    with open(filename, 'w') as f:
        f.write(f"{mol.GetNumAtoms()}\n\n")
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            f.write(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")

# === Step 1: Read SMILES ===
with open(smiles_file, 'r') as f:
    smiles_list = [line.strip() for line in f if line.strip()]

os.makedirs(xyz_dir, exist_ok=True)
os.makedirs(com_dir, exist_ok=True)

# === Step 2: SMILES → 3D XYZ ===
for i, smi in enumerate(smiles_list):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print(f"[ERROR] Invalid SMILES at line {i+1}: {smi}")
        continue
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) != 0:
        print(f"[ERROR] RDKit 3D embedding failed for molecule {i+1}")
        continue
    AllChem.MMFFOptimizeMolecule(mol)

    mol_folder = os.path.join(xyz_dir, str(i+1))
    os.makedirs(mol_folder, exist_ok=True)

    xyz_path = os.path.join(mol_folder, f"{i+1}.xyz")
    write_xyz(mol, xyz_path)
    print(f"[INFO] Molecule {i+1}: 3D structure written to {xyz_path}")

# === Step 3: Run xTB Optimization ===
current_dir = os.getcwd()
for i in range(len(smiles_list)):
    mol_folder = os.path.join(xyz_dir, str(i+1))
    xyz_file = os.path.join(mol_folder, f"{i+1}.xyz")

    if not os.path.exists(xyz_file):
        print(f"[WARNING] Skipping xTB: {xyz_file} not found")
        continue

    os.chdir(mol_folder)
    try:
        print(f"[INFO] Running xTB optimization for molecule {i+1}")
        subprocess.run(["xtb", f"{i+1}.xyz", "--opt", "--silent"], check=True)
    except subprocess.CalledProcessError:
        print(f"[ERROR] xTB failed for molecule {i+1}")
    os.chdir(current_dir)

# === Step 4: Write ORCA .inp from xtbopt.xyz ===
for i in range(len(smiles_list)):
    xtb_xyz = os.path.join(xyz_dir, str(i+1), "xtbopt.xyz")
    if not os.path.exists(xtb_xyz):
        print(f"[WARNING] xtbopt.xyz missing for molecule {i+1}, skipping ORCA input")
        continue

    try:
        with open(xtb_xyz, 'r') as f:
            coords = f.readlines()[2:]  # Skip first two lines of XYZ
    except Exception as e:
        print(f"[ERROR] Failed to read xtbopt.xyz for molecule {i+1}: {e}")
        continue

    mol_folder = os.path.join(com_dir, str(i+1))
    os.makedirs(mol_folder, exist_ok=True)

    inp_path = os.path.join(mol_folder, f"{i+1}.inp")

    with open(inp_path, 'w') as f:
        f.write('! B3LYP def2-SVP\n\n')
        f.write('%pal nprocs 10\nend\n\n')
        f.write('%maxcore 5000\n\n')
        f.write('%tddft\n Mode sTDA\n Ethresh 10.0\n PThresh 1e-4\n PTLimit 30\n maxcore 20000\n triplets true\nend\n\n')
        f.write('* xyz 0 1\n')
        f.writelines(coords)
        f.write('*\n\n')

    os.chdir(current_dir)

# === Step 5: Submit ORCA jobs one by one ===
current_dir = os.getcwd()
for i in range(len(smiles_list)):
    input_folder = f"sTDA_calculation/{i+1}"
    try:
        os.chdir(input_folder)
        input_file = f"{i+1}.inp"
        log_file = f"{i+1}.out"
        command = [f"/home/cmclab/orca_6_1_0/orca {input_file} > {log_file}"]
#        pid = os.getpid()
        print(f"Submitted std job in folder {i+1}")
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f'Command failed for {i+1}')
#        time.sleep(5)
    finally:
        os.chdir(current_dir)
#        time.sleep(5)

# === Step 6: Create csv file of std T1 energy ===
import csv

# Conversion factor
csv_output = "sTDA_Energy.csv"

# Open CSV for writing
with open(csv_output, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Id', 'smiles', 'T_stda'])  # Header

    for i, smi in enumerate(smiles_list):
        out_file = os.path.join(com_dir, str(i+1), f"{i+1}.out")
        t_std_ev = None

        if not os.path.isfile(out_file):
            print(f"[WARNING] Output file not found for molecule {i+1}")
            continue

        try:
            with open(out_file, 'r') as f:
                lines = f.readlines()

            found_block = False
            for idx, line in enumerate(lines):
                if "excitation energies, transition moments and amplitudes" in line:
                    found_block = True
                elif found_block and "state   eV" in line:
                    # Read the next non-empty lines
                    for j in range(idx + 1, len(lines)):
                        state_line = lines[j].strip()
                        if not state_line or not state_line.startswith("0"):
                            continue
                        parts = state_line.split()
                        if len(parts) >= 2:
                            t_std_ev = float(parts[1])
                        break
                    break

            if t_std_ev is not None:
                t_std_kcal = round(t_std_ev, 3)
                writer.writerow([i+1, smi, t_std_kcal])
                print(f"[INFO] Molecule {i+1}: {t_std_ev} eV → {t_std_kcal} kcal/mol")
            else:
                print(f"[WARNING] No T_std found in output for molecule {i+1}")

        except Exception as e:
            print(f"[ERROR] Failed to parse {out_file}: {e}")

exit()


    

