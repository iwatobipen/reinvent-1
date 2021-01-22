# coding=utf-8

## load cpyshapeit
import os
import sys
sys.path.append(os.environ["PYSHAPEIT_PATH"])


from typing import List
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

import utils

try:
    import cpyshapeit
except:
    print("ouldn't find cpyshapeit")
    exit()

class shape_tanimoto(object):
    """Scores based on the Tanimoto similarity to a query SMILES. Supports a similarity cutoff k."""

    def __init__(self, query_molfile: str, k=1.0, maxIter=10):
        self.k = k
        self.maxIter = maxIter
        self.query_mol = Chem.MolFromMolFile(query_molfile)
        AllChem.MMFFOptimizeMolecule(self.query_mol)

    def __call__(self, smiles: List[str]) -> dict:
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        valid = [1 if mol is not None else 0 for mol in mols]
        tmp_valid = []
        for idx, mol in enumerate(mols):
            if valid[idx] == 1:
                #mol = Chem.AddHs(mol)
                res = AllChem.EmbedMolecule(mol)
                if res==0:
                    try:
                        res=AllChem.MMFFOptimizeMolecule(mol, confId=0)
                        if res==0:
                            tmp_valid.append(1)
                        else:
                            tmp_valid.append(0)
                    except: pass
                else:
                    tmp_valid.append(0)
            else:
                tmp_valid.append(0)
        valid = tmp_valid
        valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean == 1]
        valid_mols = [mols[idx] for idx in valid_idxs]
        for idx, mol in enumerate(valid_mols):
            mol.SetProp("_Name",str(idx))
            AllChem.EmbedMolecule(mol)
        #fps = [AllChem.GetMorganFingerprint(mol) for mol in valid_mols]
        shape_score = np.array([cpyshapeit.AlignMol(self.query_mol, m, maxIter=self.maxIter, whichScore="Shape-it::Tversky_ref") for m in valid_mols])
        tanimoto = np.minimum(shape_score, self.k) / self.k
        # tversky_similarity = np.square(tanimoto)
        score = np.full(len(smiles), 0, dtype=np.float32)

        for idx, value in zip(valid_idxs, tanimoto):
            score[idx] = value
        return {"total_score": np.array(score, dtype=np.float32)}
