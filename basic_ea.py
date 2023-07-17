import argparse
import itertools
import logging
import pathlib
from collections.abc import Iterable, Iterator

import atomlite
import numpy as np
import numpy.typing as npt
import rdkit.Chem.AllChem as rdkit
import stk
from rdkit import RDLogger
from rdkit.Chem.GraphDescriptors import BertzCT

rdkit_logger = RDLogger.logger()
rdkit_logger.setLevel(RDLogger.CRITICAL)
logger = logging.getLogger(__name__)


def get_building_blocks(
    path: pathlib.Path,
    functional_group_factory: stk.FunctionalGroupFactory,
    generator: np.random.Generator,
) -> Iterator[stk.BuildingBlock]:
    with open(path, "r") as f:
        content = f.readlines()

    for smiles in content:
        molecule = rdkit.AddHs(rdkit.MolFromSmiles(smiles))
        molecule.AddConformer(rdkit.Conformer(molecule.GetNumAtoms()))
        rdkit.Kekulize(molecule)
        building_block = stk.BuildingBlock.init_from_rdkit_mol(
            molecule=molecule,
            functional_groups=functional_group_factory,
        )
        yield building_block.with_position_matrix(
            position_matrix=get_position_matrix(building_block, generator),
        )


def get_position_matrix(
    molecule: stk.BuildingBlock,
    generator: np.random.Generator,
) -> npt.NDArray[np.float64]:
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


def get_initial_population(
    fluoros: Iterable[stk.BuildingBlock],
    bromos: Iterable[stk.BuildingBlock],
) -> Iterator[stk.MoleculeRecord[stk.polymer.Linear]]:
    for fluoro, bromo in itertools.product(fluoros, bromos):
        yield stk.MoleculeRecord(
            topology_graph=stk.polymer.Linear(
                building_blocks=[fluoro, bromo],
                repeating_unit="AB",
                num_repeating_units=1,
            ),
        )


def get_rigidity(molecule: rdkit.Mol) -> float:
    num_rotatable_bonds = rdkit.CalcNumRotatableBonds(molecule)
    # Add 1 to the denominator to prevent division by 0.
    return 1 / (num_rotatable_bonds + 1)


def get_complexity(molecule: rdkit.Mol) -> float:
    num_bad_rings = sum(1 for ring in rdkit.GetSymmSSSR(molecule) if len(ring) < 5)
    return BertzCT(molecule) + 10 * num_bad_rings**2


def get_fitness_value(
    record: stk.MoleculeRecord[stk.polymer.Linear],
) -> float:
    rdkit_molecule = record.get_molecule().to_rdkit_mol()
    rdkit.SanitizeMol(rdkit_molecule)
    return 100 * (get_rigidity(rdkit_molecule) / get_complexity(rdkit_molecule))


def get_functional_group_type(building_block: stk.BuildingBlock) -> type:
    (functional_group,) = building_block.get_functional_groups(0)
    return functional_group.__class__


def is_fluoro(building_block: stk.BuildingBlock) -> bool:
    (functional_group,) = building_block.get_functional_groups(0)
    return functional_group.__class__ is stk.Fluoro


def is_bromo(building_block: stk.BuildingBlock) -> bool:
    (functional_group,) = building_block.get_functional_groups(0)
    return functional_group.__class__ is stk.Bromo


def get_num_rotatable_bonds(
    record: stk.MoleculeRecord[stk.polymer.Linear],
) -> float:
    molecule = record.get_molecule().to_rdkit_mol()
    rdkit.SanitizeMol(molecule)
    return rdkit.CalcNumRotatableBonds(molecule)


def write(
    molecule: stk.ConstructedMolecule,
    path: pathlib.Path,
    generator: np.random.Generator,
) -> None:
    rdkit_molecule = molecule.to_rdkit_mol()
    rdkit.SanitizeMol(rdkit_molecule)
    rdkit_molecule = rdkit.RemoveHs(rdkit_molecule)
    building_block = stk.BuildingBlock.init_from_rdkit_mol(rdkit_molecule)
    building_block.with_position_matrix(
        position_matrix=get_position_matrix(building_block, generator),
    ).write(path)


def get_entry(record: stk.MoleculeRecord[stk.polymer.Linear]) -> atomlite.Entry:
    return atomlite.Entry.from_rdkit(
        key=stk.Smiles().get_key(record.get_molecule()),
        molecule=record.get_molecule().to_rdkit_mol(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database",
        help="Path to an AtomLite database holding EA results.",
        type=pathlib.Path,
        default=pathlib.Path("basic_ea.db"),
    )
    parser.add_argument(
        "--fluoros",
        help="Path to file holding SMILES of Fluoro building blocks.",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent / "fluoros.txt",
    )
    parser.add_argument(
        "--bromos",
        help="Path to file holding SMILES of Bromo building blocks.",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent / "bromos.txt",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    generator = np.random.default_rng(4)

    logger.info("Making building blocks.")

    # Load the building block databases.
    fluoros = tuple(
        get_building_blocks(
            path=args.fluoros,
            functional_group_factory=stk.FluoroFactory(),
            generator=generator,
        )
    )
    bromos = tuple(
        get_building_blocks(
            path=args.bromos,
            functional_group_factory=stk.BromoFactory(),
            generator=generator,
        )
    )

    initial_population = tuple(get_initial_population(fluoros[:5], bromos[:5]))

    # Write the initial population.
    initial_population_directory = pathlib.Path("initial_population")
    initial_population_directory.mkdir(exist_ok=True, parents=True)
    for i, record in enumerate(initial_population):
        write(
            molecule=record.get_molecule(),
            path=initial_population_directory / f"initial_{i}.mol",
            generator=generator,
        )

    db = atomlite.Database(args.database)
    ea: stk.EvolutionaryAlgorithm[
        stk.MoleculeRecord[stk.polymer.Linear]
    ] = stk.EvolutionaryAlgorithm(
        initial_population=initial_population,
        fitness_calculator=stk.FitnessFunction(get_fitness_value),
        mutator=stk.RandomMutator(
            mutators=(
                stk.RandomBuildingBlock(
                    building_blocks=fluoros,
                    is_replaceable=is_fluoro,
                    random_seed=generator,
                ),
                stk.SimilarBuildingBlock(
                    building_blocks=fluoros,
                    is_replaceable=is_fluoro,
                    random_seed=generator,
                ),
                stk.RandomBuildingBlock(
                    building_blocks=bromos,
                    is_replaceable=is_bromo,
                    random_seed=generator,
                ),
                stk.SimilarBuildingBlock(
                    building_blocks=bromos,
                    is_replaceable=is_bromo,
                    random_seed=generator,
                ),
            ),
            random_seed=generator,
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
            random_seed=generator,
        ),
        crossover_selector=stk.Roulette(
            num_batches=3,
            batch_size=2,
            random_seed=generator,
        ),
    )

    logger.info("Starting EA.")

    num_rotatable_bonds = []
    fitness_values = []
    for generation in ea.get_generations(50):
        db.update_entries(map(get_entry, generation.get_molecule_records()))
        num_rotatable_bonds.append(
            [
                get_num_rotatable_bonds(record)
                for record in generation.get_molecule_records()
            ]
        )
        fitness_values.append(
            [
                fitness_value.normalized
                for fitness_value in generation.get_fitness_values().values()
            ]
        )

    # Write the final population.
    final_population_directory = pathlib.Path("final_population")
    final_population_directory.mkdir(exist_ok=True, parents=True)
    for i, record in enumerate(generation.get_molecule_records()):
        write(
            molecule=record.get_molecule(),
            path=final_population_directory / f"final_{i}.mol",
            generator=generator,
        )

    logger.info("Making fitness plot.")

    fitness_progress = stk.ProgressPlotter(
        property=fitness_values,
        y_label="Fitness Value",
    )
    fitness_progress.write("fitness_progress.png")

    logger.info("Making rotatable bonds plot.")

    rotatable_bonds_progress = stk.ProgressPlotter(
        property=num_rotatable_bonds,
        y_label="Number of Rotatable Bonds",
    )
    rotatable_bonds_progress.write("rotatable_bonds_progress.png")


if __name__ == "__main__":
    main()
