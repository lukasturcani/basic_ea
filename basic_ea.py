import stk
import rdkit.Chem.AllChem as rdkit
from rdkit import RDLogger
import pymongo
import vabene as vb
import numpy as np
import itertools as it
import logging

rdkit_logger = RDLogger.logger()
rdkit_logger.setLevel(RDLogger.CRITICAL)
logger = logging.getLogger(__name__)


def vabene_to_rdkit(molecule):
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
    rdkit_molecule = rdkit.AddHs(rdkit_molecule)
    rdkit.Kekulize(rdkit_molecule)
    rdkit_molecule.AddConformer(
        conf=rdkit.Conformer(rdkit_molecule.GetNumAtoms()),
    )
    return rdkit_molecule


def get_building_block(
    generator,
    atomic_number,
    functional_group_factory,
):
    # The number of atoms, excluding hydrogen, in our building
    # block.
    num_atoms = generator.randint(7, 15)
    # The distance between the bromine or fluorine atoms in our
    # building block.
    fg_separation = generator.randint(1, num_atoms-3)

    atom_factory = vb.RandomAtomFactory(
        atoms=(vb.Atom(6, 0, 4), ),
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
    building_block = stk.BuildingBlock.init_from_rdkit_mol(
        molecule=vabene_to_rdkit(vb.Molecule(atoms, bonds)),
        functional_groups=[functional_group_factory],
    )
    return building_block.with_position_matrix(
        position_matrix=generator.uniform(
            low=-100,
            high=100,
            size=(building_block.get_num_atoms(), 3),
        ),
    )


def get_initial_population(fluoros, bromos):
    for fluoro, bromo in it.product(fluoros[:5], bromos[:5]):
        yield stk.MoleculeRecord(
            topology_graph=stk.polymer.Linear(
                building_blocks=(fluoro, bromo),
                repeating_unit='AB',
                num_repeating_units=1,
            ),
        )


def get_rigidity(molecule):
    rdkit_molecule = molecule.to_rdkit_mol()
    rdkit.SanitizeMol(rdkit_molecule)
    num_rotatable_bonds = rdkit.CalcNumRotatableBonds(
        mol=rdkit_molecule,
    )
    # Add 1 to the denominator to prevent division by 0.
    return 1 / (num_rotatable_bonds + 1)


def get_functional_group_type(building_block):
    functional_group, = building_block.get_functional_groups(0)
    return functional_group.__class__


def is_fluoro(building_block):
    functional_group, = building_block.get_functional_groups(0)
    return functional_group.__class__ is stk.Fluoro


def is_bromo(building_block):
    functional_group, = building_block.get_functional_groups(0)
    return functional_group.__class__ is stk.Bromo


def get_num_rotatable_bonds(record):
    molecule = record.get_molecule().to_rdkit_mol()
    rdkit.SanitizeMol(molecule)
    return rdkit.CalcNumRotatableBonds(molecule)


def with_structure(molecule):
    generator = np.random.RandomState(4)
    position_matrix = generator.uniform(
        low=-500,
        high=500,
        size=(molecule.get_num_atoms(), 3),
    )
    molecule = molecule.with_position_matrix(position_matrix)
    rdkit_molecule = molecule.to_rdkit_mol()
    rdkit.SanitizeMol(rdkit_molecule)
    rdkit_molecule = rdkit.RemoveHs(rdkit_molecule)
    rdkit.Compute2DCoords(rdkit_molecule)
    try:
        rdkit.MMFFOptimizeMolecule(rdkit_molecule)
    except Exception:
        pass
    return stk.BuildingBlock.init_from_rdkit_mol(rdkit_molecule)


def main():
    logging.basicConfig(level=logging.INFO)

    # Use a random seed to get reproducible results.
    random_seed = 4
    generator = np.random.RandomState(random_seed)

    logger.info('Making building blocks.')

    # Make 1000 fluoro building bocks.
    fluoros = tuple(
        get_building_block(generator, 9, stk.FluoroFactory())
        for i in range(1000)
    )
    # Make 1000 bromo building blocks.
    bromos = tuple(
        get_building_block(generator, 35, stk.BromoFactory())
        for i in range(1000)
    )

    initial_population = tuple(get_initial_population(fluoros, bromos))
    # Write the initial population.
    for i, record in enumerate(initial_population):
        with_structure(record.get_molecule()).write(f'initial_{i}.mol')

    db = stk.ConstructedMoleculeMongoDb(pymongo.MongoClient())
    ea = stk.EvolutionaryAlgorithm(
        initial_population=initial_population,
        fitness_calculator=stk.FitnessFunction(get_rigidity),
        mutator=stk.RandomMutator(
            mutators=(
                stk.RandomBuildingBlock(
                    building_blocks=fluoros,
                    is_replaceable=is_fluoro,
                    random_seed=generator.randint(0, 1000),
                ),
                stk.SimilarBuildingBlock(
                    building_blocks=fluoros,
                    is_replaceable=is_fluoro,
                    random_seed=generator.randint(0, 1000),
                ),
                stk.RandomBuildingBlock(
                    building_blocks=bromos,
                    is_replaceable=is_bromo,
                    random_seed=generator.randint(0, 1000),
                ),
                stk.SimilarBuildingBlock(
                    building_blocks=bromos,
                    is_replaceable=is_bromo,
                    random_seed=generator.randint(0, 1000),
                ),
            ),
            random_seed=generator.randint(0, 1000),
        ),
        crosser=stk.GeneticRecombination(
            get_gene=get_functional_group_type,
        ),
        generation_selector=stk.Best(
            num_batches=25,
            duplicate_molecules=False,
        ),
        mutation_selector=stk.Roulette(
            num_batches=5,
            random_seed=generator.randint(0, 1000),
        ),
        crossover_selector=stk.Roulette(
            num_batches=3,
            batch_size=2,
            random_seed=generator.randint(0, 1000),
        ),
        # We don't need to do a normalization in this example.
        fitness_normalizer=stk.NullFitnessNormalizer(),
    )

    logger.info('Starting EA.')

    generations = []
    for generation in ea.get_generations(15):
        for record in generation.get_molecule_records():
            db.put(record.get_molecule())
        generations.append(generation)

    # Write the final population.
    for i, record in enumerate(generation.get_molecule_records()):
        with_structure(record.get_molecule()).write(f'final_{i}.mol')

    logger.info('Making fitness plot.')

    fitness_progress = stk.ProgressPlotter(
        generations=generations,
        get_property=lambda record: record.get_fitness_value(),
        y_label='Fitness Value',
    )
    fitness_progress.write('fitness_progress.png')

    logger.info('Making rotatable bonds plot.')

    rotatable_bonds_progress = stk.ProgressPlotter(
        generations=generations,
        get_property=get_num_rotatable_bonds,
        y_label='Number of Rotatable Bonds',
    )
    rotatable_bonds_progress.write('rotatable_bonds_progress.png')


if __name__ == '__main__':
    main()
