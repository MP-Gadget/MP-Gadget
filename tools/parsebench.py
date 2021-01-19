"""Quick script to parse cpu.txt files output by MP-Gadget and create scalability information."""

import glob
import os
import re
import collections

def parse_step_header(line):
    """Parse a string describing a step into total simulation time and scalefactor of the step.
    Returns None if the line is not a step header, otherwise a tuple of Stepnum, scale factor, total elapsed time."""
    regex = r"Step ([0-9]*), Time: ([\.0-9]*), MPIs: ([0-9]*) Threads: ([0-9]*) Elapsed: ([\.0-9]*)"
    reg = re.match(regex, line)
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
            if not k in ret.keys():
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
    ppath = []
    for ll in textdata:
        reg = re.match(regex,ll)
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
    with open(fname) as fn:
        line = fn.readline()
        #Read lines, stopping when done.
        laststep = []
        while line != "":
            lhead = parse_step_header(line)
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
            line = fn.readline()
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
        head, stepd = parse_file(cpu, sf=sf)
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
        head, stepd = parse_file(cpu, sf=sf)
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
