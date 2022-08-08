"""Quick script to parse cpu.txt files output by MP-Gadget and create scalability information."""

import glob
import os
import re
import collections
import matplotlib.pyplot as plt
import numpy as np

def parse_step_header(line, compiled_regex):
    """Parse a string describing a step into total simulation time and scalefactor of the step.
    Returns None if the line is not a step header, otherwise a tuple of Stepnum, scale factor, total elapsed time."""
    # This avoids the overhead of doing the regex for non-header lines.
    reg = re.match(compiled_regex, line)
    if reg is None:
        return None
    grps = reg.groups()
    summary = {"Step" : int(grps[0]), "Scale": float(grps[1]), "MPI": int(grps[2]), "Thread": int(grps[3]), "Time": float(grps[-1])}
    return summary

def Tree():
    """A tree structure: a dictionary whose default values are trees."""
    return collections.defaultdict(Tree)

def add(t, path, value):
    """Add a whole path to the tree."""
    for node in path[:-1]:
        t = t[node]
    t[path[-1]] = value

def TreeTime(SubTree):
    """Sum the total times in a subtree."""
    try:
        return sum(SubTree.values())
    except TypeError:
        return sum([TreeTime(ss) for ss in SubTree.values()])
    except AttributeError:
        return SubTree

def dicts(t):
    """Convert a defaultdict collection into true dicts, to prevent more expansion."""
    try:
        return {k: dicts(t[k]) for k in t}
    except TypeError:
        return t

def merge_trees(t1, t2):
    """Merge two trees together, by summing each value recursively."""
    if t1 is None:
        return t2
    if t2 is None:
        return t1
    ret = {}
    for k in t1.keys():
        try:
            #If we reached a leaf
            ret[k] = t1[k] + t2[k]
        #This happens if the line number changed
        except KeyError:
            #Try and find an equivalent key
            for k2 in t2.keys():
                if re.match(k[:-4], k2):
                    ret[k] = t1[k] + t2[k2]
                    break
            #If we didn't find an equivalent key,
            #we just drop this entry
            if not k in ret:
                print("Could not merge: ",k," in ",t2)
                ret[k] = t1[k]
        except TypeError:
            ret[k] = merge_trees(t1[k], t2[k])
    return ret

def parse_full_step(textdata):
    """Parse a full chunk of text into separate time entries."""
    stepd = Tree()
    #whitespace followed by a word, followed by whitespace, followed by a time.
    regex = r"(\W+)([a-zA-Z][_@\.a-zA-Z0-9\-:]+)\W+([\.0-9]+)"
    pattern = re.compile(regex)
    ppath = []
    for ll in textdata:
        reg = re.match(pattern,ll)
        if reg is None:
            continue
        grps = reg.groups()
        level = len(grps[0])
        node = grps[1]
        time = float(grps[2])
        #Walking up the tree as needed
        if len(ppath) >= level:
            #If we need to walk up, we are done with this branch.
            add(stepd, ppath, ptime)
            while len(ppath) >= level:
                ppath.pop()
        ppath += [node,]
        ptime = time
    #Add last path
    if len(ppath) > 0:
        add(stepd, ppath, ptime)
    return dicts(stepd)

def parse_file(fname, step = None, sf=None):
    """Parse a file from the end, looking for the last step before or equal to scalefactor.
    Returns a dictionary of total times."""
    regex = r"Step ([0-9]*), Time: ([\.0-9]*), MPIs: ([0-9]*) Threads: ([0-9]*) Elapsed: ([\.0-9]*)"
    compiled_regex = re.compile(regex)
    with open(fname) as fn:
        laststep = []
        for line in fn:
            lhead = None
            #Read lines, stopping when done.
            if r"Step" == line[:4]:
                lhead = parse_step_header(line, compiled_regex)
            if lhead is not None:
                #Stop reading if we finished the last step we wanted.
                if sf is not None and lhead["Scale"] > sf:
                    break
                if step is not None and lhead["Step"] > step:
                    break
                #New step
                laststep = []
                head = lhead
            else:
                #Accumulate non-header lines
                laststep += [line,]
    stepdata = parse_full_step(laststep)
    return head, stepdata

def get_cpu_time(directory, endsf=None):
    """Get the total cpu time required for a simulation,
        adding together all cpu.txt files."""
    cpus = sorted(glob.glob(os.path.join(directory, "cpu.tx*")))
    sf = endsf
    time = 0
    steps = 0
    for cpu in cpus[::-1]:
        try:
            head, _ = parse_file(cpu, sf=sf)
        #An empty file or one with no steps before sf
        except UnboundLocalError:
            continue
        #Add to totals
        time += head["Time"] * head["MPI"] * head["Thread"]
        steps += head["Step"]
        #Get starting position of this file for the end of the next one
        headstart, _ = parse_file(cpu, step = 0)
        sf = headstart["Scale"]
    return time, steps

def get_total_time(directory, endsf = None):
    """Get the total (wall) time required for a simulation,
        adding together all cpu.txt files."""
    cpus = sorted(glob.glob(os.path.join(directory, "cpu.tx*")))
    sf = endsf
    totals = {"MPI": 0, "Thread": 0, "Time": 0}
    steptot = None
    for cpu in cpus[::-1]:
        try:
            head, stepd = parse_file(cpu, sf=sf)
        except UnboundLocalError:
            continue
        #Add to totals
        totals["Time"] += head["Time"]
        # Check that the core counts are the same
        if totals["MPI"] == 0:
            totals["MPI"] = head["MPI"]
            totals["Thread"] = head["Thread"]
        assert totals["MPI"] == head["MPI"]
        assert totals["Thread"] == head["Thread"]
        steptot = merge_trees(steptot, stepd)
        #Get starting position of this file for the end of the next one
        headstart, _ = parse_file(cpu, step = 0)
        sf = headstart["Scale"]
    return totals, steptot

def plot_sim_cost(directory):
    """Make a plot showing how much time a simulation takes as a function of scale factor.
    """
    cpus = sorted(glob.glob(os.path.join(directory, "cpu.tx*")))
    #Find start and end scale factors
    headstart, _ = parse_file(cpus[0], step = 0)
    headfinal, _ = parse_file(cpus[-1])
    scales = np.linspace(headstart["Scale"], headfinal["Scale"], 100)
    total = np.zeros_like(scales)
    gravity = np.zeros_like(scales)
    treegrav = np.zeros_like(scales)
    domain = np.zeros_like(scales)
    hydro = np.zeros_like(scales)
    galaxy = np.zeros_like(scales)
    hrs = 60 * 60.
    for ii, aa in enumerate(scales[1:]):
        jj = ii +1
        for cpu in cpus:
            head, stepd = parse_file(cpu, sf = aa)
            #Did we find this scale factor in this file?
            #Has to be approximate because the steps
            #might not line up with the linspace() output
            final_file = abs(head["Scale"] - aa) < 5e-3*aa
            #print(cpu, head["Scale"], aa, final_file)
            if final_file:
                jj = ii + 1
            else:
                jj = np.arange(ii+1, np.size(total))
            total[jj] += head["Time"]*head["MPI"]/hrs
            gravity[jj] += head["MPI"]*TreeTime(stepd["PMgrav"])/hrs
            treegrav[jj] += head["MPI"]*TreeTime(stepd["Tree"])/hrs
            domain[jj] += head["MPI"]*TreeTime(stepd["Domain"])/hrs
            if "SPH" in stepd:
                hydro[jj] += head["MPI"]*TreeTime(stepd["SPH"])/hrs
            if "Cooling" in stepd:
                galaxy[jj] += head["MPI"]*TreeTime(stepd["Cooling"])/hrs
            if "BH" in stepd:
                galaxy[jj] += head["MPI"]*TreeTime(stepd["BH"])/hrs
            if "FOF" in stepd:
                galaxy[jj] += head["MPI"]*TreeTime(stepd["FOF"])/hrs
            #Did we find this scale factor in this file? If so, we are done.
            # If not then we already added this file to totals, don't need to do it again.
            if final_file:
                break
            cpus = cpus[1:]
    plt.plot(scales, np.array(total), ls="-", color="black", label="Total")
    plt.plot(scales, np.array(domain), ls="-", color="blue", label="Domain")
    plt.plot(scales, np.array(gravity), ls="-", color="red", label="PM Gravity")
    plt.plot(scales, np.array(treegrav), ls="-", color="brown", label="Tree Gravity")
    plt.plot(scales, np.array(hydro), ls="-", color="pink", label="Hydro")
    plt.plot(scales, np.array(galaxy), ls="-", color="green", label="Galaxy")
    plt.legend(loc="upper left", ncol=3)
    plt.ylabel("SUs")
    plt.xlabel("a")
