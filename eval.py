#!/usr/bin/env python3

import simexpal
import pandas as pd
import yaml
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


colors = ['b', 'g', 'r', (0.0, 0.7, 0.7), (0.7, 0.7, 0.), (0.7, 0., 0.7), (0.8, 0.4, 0.1), (0.1, 0.8, 0.4), (0.4, 0.1, 0.8)]


def parse(run, f):
    output = yaml.load(f, Loader=yaml.Loader)
    if not output:
        return {}
    exps = output['Runs']
    exp = exps[0]

    d = {}
    d = exp.copy()
    d['Instance'] = run.instance.shortname
    d["Experiment"] =  run.experiment.name

    if "JLT-Test" in exp:
        d["Experiment"] = "JLT-Test"
        d["JLT-Test"] = True

        # d = {
        #     "instance": run.instance.shortname,
        #     'experiment': "JLT-Test",
        #     'nodes': exp['Nodes'],
        #     'edges': exp['Edges'],
        #     'rel-errors': exp['Rel-Errors'],
        #     'jlt-test': True
        # }
    else:
        if 'gain' in d and d['gain'] < 0:
            d['gain'] *= -1.
        # d = {
        #     'experiment': run.experiment.name,
        #     'instance': run.instance.shortname,
        #     'nodes': exp['Nodes'],
        #     'edges': exp['Edges'],
        #     'k': exp['k'],
        #     'algorithm': exp['Algorithm'],
        #     'value': exp['Value'],
        #     'time': exp['Time'],
        #     'gain': exp['Gain'],
        # }



    return d



cfg = simexpal.config_for_dir()
df = pd.DataFrame(cfg.collect_successful_results(parse))

def print_df(df):
    print(df.to_string())



jlt_df = df[df['JLT-Test'] == True]
df = df[df['JLT-Test'].isnull()]

#print(df.groupby('Experiment').agg('mean'))


matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})



def project(df, col, value):
    cols = [c for c in df.columns if c != col]
    return df[df[col] == value].filter(items=cols)


def restrict_frame(df, restrictions):
    restricted_frame = df
    for k, v in restrictions.items():
        restricted_frame = project(restricted_frame, k, v)
    return restricted_frame

def output_file(fig, filename):
    fig.savefig(filename + ".pgf", transparent=True)
    fig.savefig(filename + ".png")
    print("Wrote " + filename + ".pgf, " + filename + ".png")


def draw_jlt(df):
    jlt_test_results = project(df, "JLT-Test", True)
    instances = set(jlt_test_results["Instance"].tolist())

    err_data = []
    err_data2 = []


    for instance_name in instances:
        inst_fr = project(jlt_test_results, "Instance", instance_name)
        if not "Rel-Errors" in inst_fr:
            print_df(inst_fr)
        err_data.append(np.array(inst_fr["Rel-Errors"].tolist()[0]))
        err_data2.append(np.array(inst_fr['Rel-Errors-2'].tolist()[0]))


    fig, ax = plt.subplots()
    ax.set_ylim(0., 1.)

    ax.set_title("Relative Errors of JLT ")
    ax.boxplot(err_data, labels=instances, whis=[2.5,97.5])
    plt.ylabel('Rel. Error')
    plt.xticks(rotation=90)
    fig.tight_layout()

    output_file(fig, "jlt-test-2")    
    
    plt.cla()
    plt.clf()

    ax.set_ylim(0., 1.)

    fig, ax = plt.subplots()

    ax.set_title("Relative Errors of JLT wrt. node subset. ")
    ax.boxplot(err_data2, labels=instances, whis=[2.5, 97.5])
    plt.ylabel('Rel. Error')
    fig.tight_layout()
    plt.xticks(rotation=90)
    fig.tight_layout()

    output_file(fig, "jlt-test")    

    plt.cla()
    plt.clf()


def draw_jlt_comparison(k):
    ns = np.arange(1000, 50000, 200)
    epsilon = 0.5
    col1 = 2 + 2 * 2 / (epsilon**2/2 - epsilon**3/3) * np.log(ns)
    fig, ax = plt.subplots()
    col3 = ns / (2**0.5) / k * math.log(1. / 0.9)
    epsilon = 0.75
    col2 = 2 + 2 * 2 / (epsilon**2/2 - epsilon**3/3) * np.log(col3)
    ax.plot(ns, col1)
    ax.plot(ns, col2)
    ax.plot(ns, col3)
    plt.legend(["JLT", "JLT subset", "Main"])
    plt.xlabel("n")
    plt.ylabel("count")
    ax.set_title("Solved linear eqns. ")

    output_file(fig, "jlt-cols"+str(k))

    plt.cla()
    plt.clf()




def plot_averaged(df, instance_names, experiment_restriction_list, experiment_names, filename=None):    
    ks = set()
    resistances_by_k = {}
    times_by_k = {}

    def analyze_experiment(restrictions):
        resistances = {}
        times = {}

        restricted_frame = restrict_frame(df, restrictions)
        for instance_name in instance_names:
            instance_frame = project(restricted_frame, "Instance", instance_name)

            def insert(d, k1, v):
                if k1 not in d:
                    d[k1] = {}
                d[k1][instance_name] = v

            for row in instance_frame.iterrows():
                row = row[1]

                k = row['k']
                ks.add(k)

                insert(resistances, k, row['Gain'])
                insert(times, k, row['Time'])

        return resistances, times

    result_resistances = []
    result_times = []
    
    for restrictions in experiment_restriction_list:
        res, times = analyze_experiment(restrictions)
        result_resistances.append(res)
        result_times.append(times)
    
    submodular_resistances, submodular_times = analyze_experiment({"Threads": 12, "Experiment": "submodular-greedy"})


    ks = sorted(list(ks))


    x_pos = np.arange(len(ks))


    fig, (ax1, ax2) = plt.subplots(1, 2)
    #fig.suptitle(instance_name)
    fig._set_dpi(400)
    fig.set_size_inches((6, 4))

    ax1.set_xlabel('k')
    ax1.set_ylabel('Gain rel. Submodular Greedy')
    ax2.set_xlabel('k')
    ax2.set_ylabel('Time rel. Submodular Greedy')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(ks)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(ks)
    ax1.set_ylim(0., 1.)
    ax2.set_ylim(0., 0.5)
    
    #ax2.legend(experiment_names)


    def mean(l):
        found_any = False
        length = 0
        p = 1.
        for v in l:
            if v or v == 0.:
                p *= v
                length += 1
        if l == 0:
            return 0
        return p ** (1. / length)
    
    def offset(l, k):
        width = 0.8 / k
        return width * l - width / 2 * k + width / 2

    num_experiments = len(result_resistances)

    for j, (algorithm_resistances, algorithm_times) in enumerate(zip(result_resistances, result_times)):
        resistance_means = []
        time_means = []
        for k in ks:
            if k not in submodular_resistances and k not in submodular_times or not k in algorithm_resistances or not k in algorithm_times:
                resistance_means.append(0)
                time_means.append(0)
                continue
            algorithm_k_resistances = algorithm_resistances[k]
            algorithm_k_times = algorithm_times[k]
            relative_resistances = []
            relative_times = []
            for instance_name in instance_names:
                if instance_name in algorithm_k_resistances and instance_name in submodular_resistances[k]:
                    relative_resistances.append(algorithm_k_resistances[instance_name] / submodular_resistances[k][instance_name])
                if instance_name in algorithm_k_times and instance_name in submodular_times[k]:
                    relative_times.append(algorithm_k_times[instance_name] / submodular_times[k][instance_name])

            res_mean = mean(relative_resistances)
            time_mean = mean(relative_times)
            resistance_means.append(res_mean)
            time_means.append(time_mean)
            print("Taking means of {} instances for k = {} and j = {}.".format(len(relative_resistances), k, j))
        
        ax1.bar(x_pos + offset(j, num_experiments), resistance_means, align='center', color=colors[j], alpha = 0.5, label=experiment_names[j], width=0.8 / num_experiments)
        ax2.bar(x_pos + offset(j, num_experiments), time_means, align='center', color=colors[j], alpha = 0.5, label=experiment_names[j], width=0.8 / num_experiments)

    
    #ax1.legend()
    #ax2.legend()
    plt.legend(ncol = 1, bbox_to_anchor=(1, 1), loc='lower right')

    fig.tight_layout()
    if filename == None:
        filename = "results_aggregated"
    
    output_file(fig, filename)

    plt.close(fig)



#draw_jlt(jlt_df)
#draw_jlt_comparison(5)
#draw_jlt_comparison(20)


large_instances = ["deezer_europe", "opsahl-powergrid", "arxiv-grqc", "facebook_ego_combined", "arxiv-hephth", "arxiv-heph"]

restr_lpinv_diag = {
    "Experiment": "sq-greedy",
    "Heuristic": "Lpinv Diagonal",
    "Linalg": "LU",
    "Threads": 12,
    "Epsilon": 0.9,
    "Epsilon2": 1.0
}

plot_averaged(df, large_instances, [restr_lpinv_diag], ["Main-Resistances-Approx"], "results_aggregated_1")

restr_stoch = {
    "Experiment": "stochastic-greedy",
    "Threads": 12
}

restr_similarity = {
    "Experiment": "sq-greedy",
    "Heuristic": "Similarity",
    "Linalg": "LU",
    "Threads": 12
}

plot_averaged(df, large_instances, [restr_lpinv_diag, restr_stoch, restr_similarity], ["Main-Resistances-Approx", "Stochastic-Submodular", "Main-Similarity"], "results_aggregated_3")

restr_random = {
    "Experiment": "sq-greedy",
    "Heuristic": "Random",
    "Linalg": "LU",
    "Threads": 12
}

restr_similarity_jlt = {
    "Experiment": "sq-greedy",
    "Heuristic": "Similarity",
    "Linalg": "JLT via Sparse LU",
    "Threads": 12
}

restr_lpinv_diag_large_eps2 = {
    "Experiment": "sq-greedy",
    "Heuristic": "Lpinv Diagonal",
    "Linalg": "LU",
    "Threads": 12,
    "Epsilon": 0.9,
    "Epsilon2": 10.

}
plot_averaged(df, large_instances, [ restr_lpinv_diag_large_eps2, restr_stoch, restr_similarity, restr_random, restr_similarity_jlt], ["Main-Resistances-Approx", "Stochastic-Submodular", "Main-Similarity", "Main-Random", "Main-Similarity-JLT"], "results_aggregated_5")








def plot_instance(df, instance_name, restrictions=[], filename=None):
    restricted_frame = project(df, "Instance", instance_name)
    restricted_frame = restrict_frame(restricted_frame, restrictions)
        
    result_t = []
    result_gain = []
    result_name = []
    result_k = next(restricted_frame.iterrows())

    for index, row in restricted_frame.iterrows():
        experiment = row['Experiment']
        algorithm_name = experiment
        heuristic = row['Heuristic']
        if heuristic and pd.notnull(heuristic):
            algorithm_name += heuristic
        if heuristic == "Lpinv Diagonal":
            if row['Linalg'] != 'LU':
                continue

        rename = {
            "greedy-3" : "main-resistances-exact",
            "sq-greedyLpinv Diagonal" : "main-resistances-approx",
            "sq-greedyRandom": "main-random",
            "sq-greedySimilarity": "main-similarity",
        }
        if algorithm_name in rename:
            result_name.append(rename[algorithm_name])
        else:
            result_name.append(algorithm_name)

        if algorithm_name == "Random-Averaged":
            result_t.append(0)
        else:
            result_t.append(row['Time'])

        gain = row['Gain']
        if gain < 0:
            gain *= -1
        result_gain.append(gain)

        if result_name[-1] == "main-random":
            result_name[-1] = "main-random-" + row["Linalg"]
        if result_name[-1] == "main-similarity":
            result_name[-1] = "main-similarity-" + row["Linalg"]


    len_results = len(result_t)
    x_pos = list(range(len(result_t)))

    # Sort by best gain
    x_pos.sort(key=lambda x: result_gain[x])
    x_pos_inv = [1]*len_results
    for i in range(len_results):
        x_pos_inv[x_pos[i]] = i
    x_pos = x_pos_inv


    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(instance_name)
    fig.set_size_inches((8,4))
    fig._set_dpi(200)

    #ax1.set_xlabel('k')
    #ax1.set_xscale('log')
    #ax1.set_yscale('log')
    #ax2.set_xscale('log')
    #ax2.set_yscale('log')
    ax1.set_ylabel('Total Effective Resistance Gain')
    #ax2.set_xlabel('k')
    ax2.set_ylabel('Time (s)')

    x_pos_shifted = [x - 0.5 for x in x_pos]
    ax1.bar(x_pos, result_gain, align='center', alpha = 0.5, color=colors)
    ax2.bar(x_pos, result_t, align='center', alpha = 0.5, color=colors)
    ax1.set_xticks(x_pos_shifted)
    ax1.set_xticklabels(result_name)
    ax2.set_xticks(x_pos_shifted)
    ax2.set_xticklabels(result_name)
    ax1.tick_params(axis='x', rotation=90)
    ax2.tick_params(axis='x', rotation=90)
    
    #ax1.legend()
    #ax2.legend()
    fig.tight_layout()
    if filename == None:
        filename = instance_name
    output_file(fig, filename)

    plt.close(fig)




    #for expname, r in results.items():
        # if expname.startswith("hillclimbing"):# or expname.startswith("random"):
        #     continue
        # r.sort(key=lambda x: x[0])
        # ks = np.array([x[0] for x in r])

        # def f(a, b):
        #     return (a-b)/b
        # values = np.array([f(r[i][1], subm_values[i]) for i in range(len(r))])
        # times = np.array([x[2] for x in r])

        # if (expname != "submodular-greedy"):
        #     ax1.plot(ks, values, 'o-', label=expname)
        # if expname != "random-averaged":
        #     ax2.plot(ks, times, 'o-', label=expname)
        # else:
        #     ax2.plot(np.array([]), np.array([]))

    #handles, labels = ax1.get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper center')
    #plt.show()

#eval.plot_instance(eval.df, "arxiv-hephth", [["Threads", 12], ["k", 20]])
def quick_plot(name, threads, k):
    plot_instance(df, name, [["Threads", threads], ["k", k]], name + "-" + str(k))

#plot_instance(df, "arxiv-heph", {"Threads": 12, "k": 20})


for k in [2, 5, 20, 50, 200]:
    for i in large_instances:
        plot_instance(df, i, {"k": k}, i+"-"+str(k))



#medium_instances = ["erdos_renyi_1000_0.02.nkb", "erdos_renyi_3000_0.01.nkb", "watts_strogatz_1000_7_0.3.nkb", "watts_strogatz_3000_7_0.3.nkb", "barabasi_albert_2_1000_2.nkb", "barabasi_albert_2_3000_2.nkb"]



