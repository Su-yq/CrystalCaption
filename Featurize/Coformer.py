# -*- coding: utf-8 -*-
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from .CalcuDescriptors import CalcuDescriptors
from .AdjacentTensor import AdjacentTensor
from .Fingerprint import Fingerprint
from .VertexMatrix import VertexMatrix
from .Atom_Bond import Atom, Bond
import numpy as np
import math
import os

from .HBondCriterion import HydrogenBondCriterion

HBondCriterion = HydrogenBondCriterion()


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        print(x, type(x))
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def get_edges(rdkit_mol):
    edges = {}
    for b in rdkit_mol.GetBonds():
        start = b.GetBeginAtomIdx()
        end = b.GetEndAtomIdx()
        edges.setdefault((start, end), Bond(b, rdkit_mol))
    return edges


class Coformer(object):
    def __init__(self, mol_file, removeh=False, hb_criterion=HBondCriterion):
        self.removeh = removeh
        self.hb_criterion = hb_criterion

        self.rdkit_mol = None
        self.molname = "Unknown"

        if os.path.isfile(mol_file):
            file_extension = mol_file.lower().split('.')[-1]

            if file_extension in ['sdf', 'mol']:
                self.rdkit_mol = AllChem.MolFromMolFile(mol_file, removeHs=False)
            elif file_extension == 'mol2':
                self.rdkit_mol = AllChem.MolFromMol2File(mol_file, removeHs=False)
            else:
                try:
                    self.rdkit_mol = Chem.MolFromMolFile(mol_file, removeHs=False)
                except:
                    try:
                        self.rdkit_mol = Chem.MolFromMol2File(mol_file, removeHs=False)
                    except:
                        raise ValueError(f"无法读取文件格式: {mol_file}")

            self.molname = os.path.basename(mol_file).split('.')[0]
        else:
            self.rdkit_mol = Chem.MolFromMolBlock(mol_file, removeHs=False)
            if self.rdkit_mol is None:
                self.rdkit_mol = Chem.MolFromMol2Block(mol_file, removeHs=False)

            lines = mol_file.strip().split('\n')
            if lines:
                self.molname = lines[0].strip()

        if self.rdkit_mol is None:
            from openbabel import openbabel as ob

            obmol = ob.OBMol()
            obConversion = ob.OBConversion()

            if os.path.isfile(mol_file):
                file_ext = mol_file.lower().split('.')[-1]
                if file_ext == 'sdf':
                    obConversion.SetInFormat("sdf")
                elif file_ext == 'mol':
                    obConversion.SetInFormat("mol")
                elif file_ext == 'mol2':
                    obConversion.SetInFormat("mol2")
                else:
                    obConversion.SetInFormat("sdf")

                obConversion.ReadFile(obmol, mol_file)
            else:
                obConversion.SetInFormat("mol")
                obConversion.ReadString(obmol, mol_file)

            mol_block = obConversion.WriteString(obmol)
            self.rdkit_mol = Chem.MolFromMolBlock(mol_block, removeHs=False)

            if self.rdkit_mol is None:
                obConversion.SetOutFormat("sdf")
                mol_block = obConversion.WriteString(obmol)
                self.rdkit_mol = Chem.MolFromMolBlock(mol_block, removeHs=False)

        if self.rdkit_mol is None:
            raise ValueError(f'无法读取分子文件或字符串: {mol_file[:100]}...')

        if self.rdkit_mol.GetNumConformers() == 0:
            try:
                AllChem.EmbedMolecule(self.rdkit_mol, randomSeed=0xf00d)
                AllChem.MMFFOptimizeMolecule(self.rdkit_mol)
            except:
                try:
                    AllChem.UFFOptimizeMolecule(self.rdkit_mol)
                except:
                    print("警告: 无法优化分子结构，使用初始嵌入")

        if self.removeh:
            self.rdkit_mol = Chem.RemoveHs(self.rdkit_mol)

        self.atoms = {}
        for ix, atom in enumerate(self.rdkit_mol.GetAtoms()):
            self.atoms.setdefault(ix, Atom(atom, hb_criterion=hb_criterion, mol=self.rdkit_mol))

        self.atom_number = len(self.atoms)

    def descriptors(self, includeSandP=True, charge_model='eem2015bm'):
        return CalcuDescriptors(self, includeSandP=includeSandP, charge_model=charge_model)

    @property
    def AdjacentTensor(self):
        return AdjacentTensor(self.atoms, self.get_edges, self.atom_number)

    @property
    def Fingerprint(self):
        return Fingerprint(self.rdkit_mol)

    @property
    def VertexMatrix(self):
        return VertexMatrix(self.atoms)

    @property
    def get_edges(self):
        return get_edges(self.rdkit_mol)

    @property
    def hbond_donors(self):
        hbond_donor_ix = {}
        for ix in self.atoms:
            if self.atoms[ix].feature.is_donor:
                Hs = self.atoms[ix].get_adjHs
                hbond_donor_ix.setdefault(ix, Hs)
        return hbond_donor_ix

    @property
    def hbond_acceptors(self):
        self.hbond_acceptor_ix = []
        for ix in self.atoms:
            if self.atoms[ix].feature.is_acceptor:
                self.hbond_acceptor_ix.append(ix)
        return self.hbond_acceptor_ix

    @property
    def get_DHs(self):
        DHs = []
        for D in self.hbond_donors:
            DHs = DHs + self.hbond_donors[D]
        return DHs

    @property
    def get_CHs(self):
        CHs = []
        for ix in self.atoms:
            if self.atoms[ix].feature.symbol == 'C':
                Hs = self.atoms[ix].get_adjHs
                if len(Hs) != 0:
                    CHs = CHs + Hs
        return CHs

    @property
    def aromatic_atoms(self):
        return [a.GetIdx() for a in self.rdkit_mol.GetAromaticAtoms()]