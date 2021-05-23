import argparse
import stk
import rdkit.Chem.AllChem as rdkit
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit import RDLogger
import pymongo
import numpy as np
import itertools as it
import logging
import pathlib

rdkit_logger = RDLogger.logger()
rdkit_logger.setLevel(RDLogger.CRITICAL)
logger = logging.getLogger(__name__)


def get_building_blocks(path, functional_group_factory):
    with open(path, 'r') as f:
        content = f.readlines()

    for smiles in content:
        molecule = rdkit.AddHs(rdkit.MolFromSmiles(smiles))
        molecule.AddConformer(rdkit.Conformer(molecule.GetNumAtoms()))
        rdkit.Kekulize(molecule)
        building_block = stk.BuildingBlock.init_from_rdkit_mol(
            molecule=molecule,
            functional_groups=[functional_group_factory],
        )
        yield building_block.with_position_matrix(
            position_matrix=get_position_matrix(building_block),
        )


def get_position_matrix(molecule):
    generator = np.random.RandomState(4)
    position_matrix = generator.uniform(
        low=-500,
        high=500,
        size=(molecule.get_num_atoms(), 3),
    )
    molecule = molecule.with_position_matrix(position_matrix)
    rdkit_molecule = molecule.to_rdkit_mol()
    rdkit.SanitizeMol(rdkit_molecule)
    rdkit.Compute2DCoords(rdkit_molecule)
    try:
        rdkit.MMFFOptimizeMolecule(rdkit_molecule)
    except Exception:
        pass
    return rdkit_molecule.GetConformer().GetPositions()


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
    num_rotatable_bonds = rdkit.CalcNumRotatableBonds(molecule)
    # Add 1 to the denominator to prevent division by 0.
    return 1 / (num_rotatable_bonds + 1)


def get_complexity(molecule):
    num_bad_rings = sum(
        1 for ring in rdkit.GetSymmSSSR(molecule) if len(ring) < 5
    )
    return BertzCT(molecule) + 10*num_bad_rings**2


def get_fitness_value(molecule):
    rdkit_molecule = molecule.to_rdkit_mol()
    rdkit.SanitizeMol(rdkit_molecule)
    return 100*(
        get_rigidity(rdkit_molecule)
        / get_complexity(rdkit_molecule)
    )


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


def write(molecule, path):
    rdkit_molecule = molecule.to_rdkit_mol()
    rdkit.SanitizeMol(rdkit_molecule)
    rdkit_molecule = rdkit.RemoveHs(rdkit_molecule)
    building_block = stk.BuildingBlock.init_from_rdkit_mol(
        molecule=rdkit_molecule,
    )
    building_block.with_position_matrix(
        position_matrix=get_position_matrix(building_block),
    ).write(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mongodb_uri',
        help='The MongoDB URI for the database to connect to.',
        default='mongodb://localhost:27017/',
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Use a random seed to get reproducible results.
    random_seed = 4
    generator = np.random.RandomState(random_seed)

    logger.info('Making building blocks.')

    # Load the building block databases.
    fluoros = tuple(get_building_blocks(
        path=pathlib.Path(__file__).parent / 'fluoros.txt',
        functional_group_factory=stk.FluoroFactory(),
    ))
    bromos = tuple(get_building_blocks(
        path=pathlib.Path(__file__).parent / 'bromos.txt',
        functional_group_factory=stk.BromoFactory(),
    ))

    initial_population = tuple(get_initial_population(fluoros, bromos))
    # Write the initial population.
    for i, record in enumerate(initial_population):
        write(record.get_molecule(), f'initial_{i}.mol')

    client = pymongo.MongoClient(args.mongodb_uri)
    db = stk.ConstructedMoleculeMongoDb(client)
    ea = stk.EvolutionaryAlgorithm(
        initial_population=initial_population,
        fitness_calculator=stk.FitnessFunction(get_fitness_value),
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
    )

    logger.info('Starting EA.')

    generations = []
    for generation in ea.get_generations(50):
        for record in generation.get_molecule_records():
            db.put(record.get_molecule())
        generations.append(generation)

    # Write the final population.
    for i, record in enumerate(generation.get_molecule_records()):
        write(record.get_molecule(), f'final_{i}.mol')

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
