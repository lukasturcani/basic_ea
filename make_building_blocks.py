import rdkit.Chem.AllChem as rdkit
import vabene as vb
import numpy as np


def vabene_to_smiles(molecule):
    editable = rdkit.EditableMol(rdkit.Mol())
    for atom in molecule.get_atoms():
        rdkit_atom = rdkit.Atom(atom.get_atomic_number())
        rdkit_atom.SetFormalCharge(atom.get_charge())
        editable.AddAtom(rdkit_atom)

    for bond in molecule.get_bonds():
        editable.AddBond(
            beginAtomIdx=bond.get_atom1_id(),
            endAtomIdx=bond.get_atom2_id(),
            order=rdkit.BondType(bond.get_order()),
        )

    rdkit_molecule = editable.GetMol()
    rdkit.SanitizeMol(rdkit_molecule)
    rdkit.Kekulize(rdkit_molecule)
    return rdkit.MolToSmiles(rdkit_molecule)


def get_smiles(generator, atomic_number):
    # The number of atoms, excluding hydrogen, in our building
    # block.
    num_atoms = generator.randint(7, 16)
    # The distance between the bromine or fluorine atoms in our
    # building block.
    fg_separation = generator.randint(2, num_atoms-3)

    atom_factory = vb.RandomAtomFactory(
        atoms=(
            vb.Atom(6, 0, 4),
            vb.Atom(6, 0, 3),
            vb.Atom(7, 0, 3),
            vb.Atom(8, 0, 2),
        ),
        # All of our building blocks will have 2 halogen atoms,
        # separated by a random number of carbon atoms.
        required_atoms=(
            (vb.Atom(atomic_number, 0, 1), )
            +
            (vb.Atom(6, 0, 4), ) * fg_separation
            +
            (vb.Atom(atomic_number, 0, 1), )
        ),
        num_atoms=num_atoms,
        random_seed=generator.randint(0, 1000),
    )
    atoms = tuple(atom_factory.get_atoms())
    bond_factory = vb.RandomBondFactory(
        required_bonds=tuple(
            vb.Bond(i, i+1, 1) for i in range(fg_separation+1)
        ),
        random_seed=generator.randint(0, 1000),
    )
    bonds = bond_factory.get_bonds(atoms)
    return vabene_to_smiles(vb.Molecule(atoms, bonds))


def write_building_blocks():
    random_seed = 4
    generator = np.random.RandomState(random_seed)

    fluoros = '\n'.join(get_smiles(generator, 9) for i in range(1000))
    bromos = '\n'.join(get_smiles(generator, 35) for i in range(1000))

    with open('fluoros.txt', 'w') as f:
        f.write(fluoros)

    with open('bromos.txt', 'w') as f:
        f.write(bromos)


if __name__ == '__main__':
    write_building_blocks()
