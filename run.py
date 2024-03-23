import os
import pickle
import time
from typing import List, Optional
from tqdm import tqdm
from Graph import Arena, Player
from Solver import Solver
from plot_graph import plot_graph
import json
from argparse import ArgumentParser

def run_solver(num_nodes: Optional[int] = None,
            edge_probability: Optional[float] = None, 
            save_results: bool = False,
            save_arena: bool = False, 
            seed: int | None = None,
            plot: bool = False, 
            optimize: bool = False, 
            arena: Optional[Arena] = None):

    if arena and (num_nodes or edge_probability):
        raise ValueError("You must provide either an arena or the number of nodes and edge probability")

    if not arena:
        arena = Arena(num_nodes=num_nodes,
                    edge_probability=edge_probability, 
                    seed=seed) 
        arena.generate()
    if plot:
        plot_graph(arena)
    if save_arena:
        arena.save(f"arena_{arena.num_nodes}_{arena.edge_probability}.pkl")

    solver = Solver(arena)
    if optimize:
        start = time.time()
        num_steps = solver.optimized_value_iteration()
        end = time.time()
        
        time_in_ms = (end - start) * 1000
        min_energy_dict = solver.get_min_energy()
        if save_results:
            save_results_json(arena=arena, 
                         time_to_complete=time_in_ms, 
                         steps=num_steps, 
                         min_energy_dict=min_energy_dict,
                         file=f"{arena.num_nodes}_{arena.edge_probability}_optimized.json")
    else:
        start = time.time()
        num_steps = solver.value_iteration()
        end = time.time()

        time_in_ms = (end - start) * 1000
        min_energy_dict = solver.get_min_energy()
        if save_results:
            save_results_json(arena=arena, 
                         time_to_complete=time_in_ms, 
                         steps=num_steps, 
                         min_energy_dict=min_energy_dict,
                         file=f"{arena.num_nodes}_{arena.edge_probability}_naive.json")

    return min_energy_dict

def save_results_json(arena: Arena, 
                 min_energy_dict: dict,
                 time_to_complete: float, 
                 steps:int, 
                 file: str) -> None:


    min_energy_dict.update({"time_to_complete_ms": time_to_complete})
    min_energy_dict.update({"converged_in_steps": steps})
    min_energy_dict.update({"num_nodes": arena.num_nodes})
    min_energy_dict.update({"edge_probability": arena.edge_probability})
    min_energy_dict.update({"MAX_nodes": len([node for node, player in arena.player_mapping.items() if player == Player.MIN])}) 
    min_energy_dict.update({"MIN_nodes": len([node for node, player in arena.player_mapping.items() if player == Player.MAX])})
    min_energy_dict.update({"edges": len(arena.edges)})
    min_energy_dict.update({"sum_player_MAX_weights": sum([arena.value_mapping[node] for node, player in arena.player_mapping.items() if player == Player.MAX])})
    min_energy_dict.update({"sum_player_MIN_weights": sum([arena.value_mapping[node] for node, player in arena.player_mapping.items() if player == Player.MIN])})
    min_energy_dict.update({"min_energy_MAX": min_energy_dict[Player.MAX]})
    min_energy_dict.update({"min_energy_MIN": min_energy_dict[Player.MIN]})
    del min_energy_dict[Player.MAX]
    del min_energy_dict[Player.MIN]

    base_path = "results"    
    with open(os.path.join(base_path, file), "w") as f:
        json.dump(min_energy_dict, f)

# def profile(seed: int = 0, plot: bool = False, save: bool = False):
#     profiler = cProfile.Profile()
#     profiler.enable()

#     # Run your function
#     solution = run_solver(num_nodes=5000, edge_probability=0.2, seed=seed, plot=plot, save=save)    
#     logging.info(f"Solution: {solution}")
#     profiler.disable()
#     stats = pstats.Stats(profiler).sort_stats('cumtime')
#     stats.print_stats()

def generate_arenas(nodes_space: Optional[List[int]] = None, 
                             probability_space: Optional[List[float]] = None,
                             seed: int | None = None):
    if not nodes_space:
        nodes_space = [10, 100, 500, 1000, 5000, 10_000]
    if not probability_space:
        probability_space = [0.01, 0.05, 0.1, 0.2, 0.5]
    pbar = tqdm(total=len(nodes_space) * len(probability_space), desc="Generating arena")
    for n in nodes_space:
        for p in probability_space:
            pbar.set_description(f"Generating arena_{n}_{p}")
            if os.path.exists(f"arenas/arena_{n}_{p}.pkl"):
                pbar.update(1)
                continue
            arena = Arena(num_nodes=n, edge_probability=p, seed=seed)
            start = time.time()
            arena.generate()
            end = time.time()
            time_to_generate = (end - start) * 1000
            arena_name = f"arena_{n}_{p}"
            arena.save(f"arenas/{arena_name}.pkl")

            if not os.path.exists("arena_times.json"):
                print(f"Creating arena_times.json for the first time")
                with open("arena_times.json", "w") as f:
                    json.dump({arena_name: time_to_generate}, f)

            with open("arena_times.json", "r") as f:
                data = json.load(f) 
                data.update({arena_name: time_to_generate})
            with open("arena_times.json", "w") as f:
                json.dump(data, f)

            pbar.update(1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--node-space", dest="node_space", type=int, nargs="+") 
    parser.add_argument("--probability-space", dest="probability_space", type=float, nargs="+")

    parser.add_argument("--solve", action="store_true")
    parser.add_argument("--num-nodes", dest="num_nodes", type=int, default=100)
    parser.add_argument("--edge-probability", dest="edge_probability", type=float, default=0.4)
    parser.add_argument("--seed", dest="seed", type=int, default=0)
    parser.add_argument("--plot", dest="plot", action="store_true")
    parser.add_argument("--save-arena", dest="save_arena", action="store_true")
    parser.add_argument("--save-results", dest="save_results", action="store_true")
    parser.add_argument("--optimize", dest="optimize", action="store_true")
    parser.add_argument("--arena-path", dest="arena_path", type=str, default=None)
    args = parser.parse_args()

    if args.solve:
        if args.arena_path:
            with open(args.arena_path, "rb") as f:
                arena = pickle.load(f)
            result = run_solver(arena=arena, plot=args.plot, save_arena=args.save_arena, optimize=args.optimize, save_results=args.save_results)
        else:
            result = run_solver(num_nodes=args.num_nodes, 
                        edge_probability=args.edge_probability, 
                        save_results=args.save_results,
                        seed=args.seed, 
                        plot=args.plot, 
                        save_arena=args.save_arena, 
                        optimize=args.optimize)
        print(result)
    elif args.generate:
        generate_arenas(nodes_space=args.node_space, 
                        probability_space=args.probability_space, 
                        seed=args.seed)
    else:
        raise ValueError("You must provide either --generate or --solve")
