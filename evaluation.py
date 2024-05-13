import json

def parse_arena_times():
    path = 'arena_times.json'
    with open(path, 'r') as f:
        data = json.load(f)

    parsed_data = {}

    for arena_name in data:
        print(arena_name)
        parsed_data[arena_name] = {}
        for strategy, times in data[arena_name].items():
            avg_times = sum(item["time_to_generate"] for item in times) / len(times)
            parsed_data[arena_name][strategy] = avg_times

    print(parsed_data)
    # Compute percentage difference
    for arena_name in parsed_data.copy():
        if "none" not in parsed_data:
            continue
        baseline = parsed_data[arena_name]["none"]
        for strategy, avg_time in parsed_data[arena_name].copy().items():
            if strategy == "none":
                continue
            percentage = (avg_time - baseline) / baseline * 100
            parsed_data[arena_name]["percentage_diff"] = percentage

    print(parsed_data)
    with open('parsed_arena_times.json', 'w') as f:
        json.dump(parsed_data, f, indent=4)
    return parsed_data

def parse_result_times():
    parsed_data = {}
    with open ('solve_results.json', 'r') as f:
        data = json.load(f)

    for arena_name in data: 
        avg_optimized = 0
        avg_naive = 0
        # num_nodes = arena_name.split("_")[1]
        # edge_p = arena_name.split("_")[2]
        for run in data[arena_name]["not_applicable"]:
            if run["optimized"]:
                avg_optimized += run["time_to_complete"]
            else:
                avg_naive += run["time_to_complete"]
        parsed_data[arena_name] = {
            "naive": avg_naive / len(data[arena_name]["not_applicable"]),
            "optimized": avg_optimized / len(data[arena_name]["not_applicable"]),
        } 

    with open('parsed_solved_results.json', 'w') as f:
        json.dump(parsed_data, f, indent=4)
    return parsed_data

def generate_latex_table_result_times():
    path = "parsed_solved_results.json"
    result = ""
    with open(path, 'r') as f:
        data  = json.load(f)

    sorted_data = sorted(data.items(), key=lambda x: (int(x[0].split("_")[1]), float(x[0].split("_")[2])))
    prev_nodes = None
    for item, eval in sorted_data:
        num_nodes = item.split("_")[1]
        edge_p = item.split("_")[2]
        if prev_nodes is not None and prev_nodes != num_nodes:
            result += "\\hline \n"
        prev_nodes = num_nodes
        result += f"{num_nodes} & {edge_p} & {parse_time(eval['naive'])} & {parse_time(eval['optimized'])}"
        result += "\\\\ \n"
    with open('latex_table_results.txt', 'w') as f:
        f.write(result)
    return result

def parse_time(ms_time):
    if ms_time > 1000 * 60:
        return f"{ms_time / 1000 / 60:.2f} min" 
    if ms_time > 1000:
        return f"{ms_time / 1000:.2f} sec"
    return f"{ms_time:.2f} ms"

def generate_latex_table_arena_generation():
    path = "parsed_arena_times.json"
    result = ""

    with open(path, 'r') as f:
        data = json.load(f)
    
    sorted_data = sorted(data.items(), key=lambda x: (int(x[0].split("_")[1]), float(x[0].split("_")[2])))
    prev_nodes = None
    for item, eval in sorted_data:
        num_nodes = item.split("_")[1]
        edge_p = item.split("_")[2]
        if prev_nodes is not None and prev_nodes != num_nodes:
            result += "\\hline \n"
        prev_nodes = num_nodes
        result += f"{num_nodes} & {edge_p}" 
        if "none" in eval:
            result += f" & {parse_time(eval['none'])}"
        else:
            result += " & Nan"
        if "bellman_ford" in eval:
            result += f" & {parse_time(eval['bellman_ford'])}"
        else:
            result += " & Nan"
        if "incremental_bellman_ford" in eval:
            result += f" & {parse_time(eval['incremental_bellman_ford'])}"
        else:
            result += " & Nan"
        result += "\\\\ \n"
    with open('latex_table_arena_generation.txt', 'w') as f:
        f.write(result)
    return result

if __name__ == '__main__':
    parse_arena_times()
    generate_latex_table_arena_generation()
    # parse_result_times()
    # generate_latex_table_result_times()
