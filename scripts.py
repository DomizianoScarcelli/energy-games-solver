import cProfile
import logging
import os
import pstats
import time
from typing import Optional

from tqdm import tqdm
from Graph import Arena
from plot_graph import plot_graph
import json


def run_solver(num_nodes: int = 30, edge_probability: float = 0.1, seed: int | None = None, plot: bool = False, save: bool = False):
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

if __name__ == "__main__":
    # generate_multiple_arenas()
    print(solve_game(num_nodes=10, edge_probability=0.3, seed=0, plot=True, save=False))
    # profile(plot=False, save=True, seed=0)    
    # run_multiple(n_runs=1, plot=False, save=True)
