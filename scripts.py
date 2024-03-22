import cProfile
import logging
import os
import pickle
import pstats
import time
from typing import Optional

from tqdm import tqdm
from Graph import Arena
from Solver import Solver
from plot_graph import plot_graph
import json

from argparse import ArgumentParser


def run_solver(num_nodes: Optional[int] = None,
            edge_probability: Optional[float] = None, 
            seed: int | None = None,
            plot: bool = False, 
            save: bool = False, 
            optimize: bool = False, 
            arena: Optional[Arena] = None):

    if not arena:
        arena = Arena(num_nodes=num_nodes,
                    edge_probability=edge_probability, 
                    seed=seed) 
        arena.generate()
    if arena and (num_nodes or edge_probability):
        raise ValueError("You must provide either an arena or the number of nodes and edge probability")
    if plot:
        plot_graph(arena)
    if save:
        arena.save(f"arena_{num_nodes}_{edge_probability}.pkl")
    
    solver = Solver(arena)
    if optimize:
        solver.optimized_value_iteration()
    else:
        solver.value_iteration()
    min_energy_dict = solver.get_min_energy()
    return min_energy_dict


def run_multiple(n_runs: int = 100, plot: bool = False, save: bool = False):
    times = []
    for seed in range(0, n_runs):
        try:
            start = time.time()
            solution = run_solver(num_nodes=5000, edge_probability=0.1, seed=seed, plot=plot, save=save)
            end = time.time()
            times.append(end - start)
            logging.info(f"Solution: {solution}")
        except AssertionError as e:
            logging.error(f"Seed {seed} failed with error: {e}")
            break
    avg_time = (sum(times) / len(times)) * 1000
    logging.info(f"Average time: {avg_time:f} ms")

def profile(seed: int = 0, plot: bool = False, save: bool = False):
    profiler = cProfile.Profile()
    profiler.enable()

    # Run your function
    solution = run_solver(num_nodes=5000, edge_probability=0.2, seed=seed, plot=plot, save=save)    
    logging.info(f"Solution: {solution}")
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()

def generate_multiple_arenas():
    num_nodes = [10, 100, 500, 1000, 5000, 10_000]
    edge_probability = [0.01, 0.05, 0.1, 0.2, 0.5]
    SEED = 42
    pbar = tqdm(total=len(num_nodes) * len(edge_probability), desc="Generating arena")
    for n in num_nodes:
        for p in edge_probability:
            pbar.set_description(f"Generating arena_{n}_{p}")
            if os.path.exists(f"arenas/arena_{n}_{p}.pkl"):
                pbar.update(1)
                continue
            arena = Arena(num_nodes=n, edge_probability=p, seed=42)
            start = time.time()
            arena.generate()
            end = time.time()
            time_to_generate = (end - start) * 1000
            arena_name = f"arena_{n}_{p}"
            arena.save(f"arenas/{arena_name}.pkl")
            with open("arena_times.json", "r") as f:
                data = json.load(f) 
                data.update({arena_name: time_to_generate})
            with open("arena_times.json", "w") as f:
                json.dump(data, f)

            pbar.update(1)

def solve_game(num_nodes: int = 10, edge_probability: float = 0.1, seed: int | None = None, plot: bool = False, save: bool = False):
    arena = Arena(num_nodes=num_nodes,
                  edge_probability=edge_probability, 
                  seed=seed) 
    arena.generate()
    if plot:
        plot_graph(arena)
    if save:
        arena.save(f"arena_{num_nodes}_{edge_probability}.pkl")
    arena.value_iteration()
    min_energy_dict = arena.get_min_energy()
    return min_energy_dict

def solve_all():
    arenas_path = "arenas"
    for file in tqdm(os.listdir(arenas_path)):
        if file.endswith(".pkl"):
            arena = Arena()
            arena = arena.load(os.path.join(arenas_path, file))
            start = time.time()
            _, steps = arena.value_iteration()
            end = time.time()
            final_time = (end - start) * 1000
            min_energy_dict = arena.get_min_energy()
            min_energy_dict.update({"time_to_complete_ms": final_time})
            min_energy_dict.update({"converged_in_steps": steps})
            min_energy_dict.update({"num_nodes": arena.num_nodes})
            min_energy_dict.update({"edge_probability": arena.edge_probability})
            min_energy_dict.update({"player_1_nodes": len([node for node, player in arena.player_mapping.items() if player == 1])}) 
            min_energy_dict.update({"player_2_nodes": len([node for node, player in arena.player_mapping.items() if player == 2])})
            min_energy_dict.update({"edges": len(arena.edges)})
            min_energy_dict.update({"sum_player_1_weights": sum([arena.value_mapping[node] for node, player in arena.player_mapping.items() if player == 1])})
            min_energy_dict.update({"sum_player_2_weights": sum([arena.value_mapping[node] for node, player in arena.player_mapping.items() if player == 2])})
            min_energy_dict.update({"min_energy_1": min_energy_dict[1]})
            min_energy_dict.update({"min_energy_2": min_energy_dict[2]})
            del min_energy_dict[1]
            del min_energy_dict[2]
            with open(f"results/min_energy_{file}.json", "w") as f:
                json.dump(min_energy_dict, f)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--solve", action="store_true")

    parser.add_argument("--num-nodes", dest="num_nodes", type=int, default=100)
    parser.add_argument("--edge-probability", dest="edge_probability", type=float, default=0.4)
    parser.add_argument("--seed", dest="seed", type=int, default=0)
    parser.add_argument("--plot", dest="plot", action="store_true")
    parser.add_argument("--save", dest="save", action="store_true")
    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--arena-path", dest="arena_path", type=str, default=None)
    args = parser.parse_args()

    if args.solve:
        if args.arena_path:
            with open(args.arena_path, "rb") as f:
                arena = pickle.load(f)
            result = run_solver(arena=arena, plot=args.plot, save=args.save, optimize=args.optimize)
        else:
            result = run_solver(num_nodes=args.num_nodes, 
                        edge_probability=args.edge_probability, 
                        seed=args.seed, 
                        plot=args.plot, 
                        save=args.save, 
                        optimize=args.optimize)
        print(result)
    elif args.generate:
        generate_multiple_arenas()
    else:
        raise ValueError("You must provide either --generate or --solve-all")
