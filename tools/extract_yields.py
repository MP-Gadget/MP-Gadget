"""Parse a yield table in the format used by Karakas and collaborators and output it in the raw C format we prefer.

There is what I assume to be a typo in Karaks 2010 table A3: M = 2 is listed as M= 2.1"""
import re
import os.path

def parse_2010_file(fname):
    """Parse a yield file. Looks for the end of each mass/metallicity bin
    Returns a dictionary of total metal yields."""
    yields = {}
    with open(fname) as fn:
        #Get first header
        line = fn.readline()
        head = parse_step_header(line)
        line = fn.readline()
        #Read lines, stopping when done.
        laststep = []
        while line != "":
            lhead = parse_step_header(line)
            if lhead is not None:
                yielddata = parse_full_bin(laststep)
                (mass, metal, ejected) = head
                yielddata['ej'] = ejected
                yields[(mass, metal)] = yielddata
                #New step
                laststep = []
                head = lhead
            else:
                #Accumulate non-header lines
                laststep += [line,]
            line = fn.readline()
        #Process final bin
        yielddata = parse_full_bin(laststep)
        (mass, metal, ejected) = head
        yielddata['ej'] = ejected
        yields[(mass, metal)] = yielddata
    return yields

def parse_step_header(line):
    """Parse a string describing a mass, metallicity bin.
    Returns None if the line is not a bin header, otherwise a tuple of mass, metallicity and ejected mass."""
    regex = r"# Minitial =  ([\.0-9]+) msun, Z = ([\.0-9]+), Mfinal = ([\.0-9]+) msun"
    reg = re.match(regex, line)
    if reg is None:
        return None
    grps = reg.groups()
    initmass = float(grps[0])
    return initmass, float(grps[1]), initmass - float(grps[2])

def parse_full_bin(textdata):
    """Split a line of a yield table into a yield number and an elemental species"""
    metalnames = ('H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ca', 'Fe', 'Z')
    #Zero yield for this bin
    yielddata = {mm : 0 for mm in metalnames}
    for line in textdata:
        #Regex for a yield line. If this doesn't match, return nothing.
        regex = r"\s+([a-z0-9]+)\s+[0-9]+\s+([\.0-9E\-\+]+)"
        reg = re.match(regex, line)
        if reg is None:
            continue
        grps = reg.groups()
        species = parse_species(grps[0], metalnames)
        yy = float(grps[1])
        #Add individual tracked species
        if species is not None:
            yielddata[species] += yy
        #if species == 'H':
            #print(grps[1], reg)
        #Add non-H, non-He to total metals
        if species not in ('H', 'He'):
            yielddata['Z'] += yy
    return yielddata

def parse_species(string, metalnames):
    """Convert a species into a canonical name"""
    for mm in metalnames:
        if mm == 'H':
            if string in ('p', 'd'):
                return 'H'
        if re.match(mm.lower()+"[0-9]+", string) is not None:
            return mm
    #Default to returning nothing
    return None

def format_c_array(data, rowkeys, colkeys, key, fstr='%.3f'):
    """Format an array for C, including line breaks in the right places."""
    string = ""
    for rk in rowkeys:
        for ck in colkeys:
            string+= (fstr +",") % data[(rk,ck)][key]
        string += "\n"
    return string

def format_for_c(arrayname, yields):
    """Format the parsed yield data into a C style array."""
    #Masses and metallicities
    out = """#define %(uname)s_NMET %(zsize)d
#define %(uname)s_NMASS %(msize)d
double %(name)s_masses[%(uname)s_NMASS] = { %(masses)s };
double %(name)s_metallicities[%(uname)s_NMET] = { %(metals)s };
double %(name)s_total_mass[%(uname)s_NMET*%(uname)s_NMASS] = {
%(eject)s
};

double %(name)s_total_metals[%(uname)s_NMET*%(uname)s_NMASS] = {
%(totmet)s
};

double %(name)s_yield[NSPECIES][%(uname)s_NMET*%(uname)s_NMASS] = {
%(yields)s
};

    """
    (mass, metals) = zip(*yields.keys())
    mass = sorted(list(set(mass)))
    #Remove the M = 6.5 bin for now
    mass = [mm for mm in mass if mm < 6.5]
    metals = sorted(list(set(metals)))
    cmass = ','.join('%.3f' % m for m in mass)
    cmet = ','.join('%.3f' % m for m in metals)
    cejected = format_c_array(yields, mass, metals, 'ej')
    ctotmet = format_c_array(yields, mass, metals, 'Z', '%.3e')
    metalnames = ('H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'S', 'Ca', 'Fe')
    cyield = '{'+'\n},\n{\n'.join([format_c_array(yields, mass, metals, mm, '%.6e') for mm in metalnames])+'}'
    cstring = out % dict(name = arrayname, uname = arrayname.upper(),
                         msize = len(mass), zsize = len(metals),
                         masses = cmass, metals = cmet,
                         eject=cejected, totmet=ctotmet, yields=cyield)
    return cstring

def get_all_agb():
    """Get the tables for all AGB stars."""
    # First Karakas 2010. These tables are supplementary data on the journal archive.
    # Not committed because possible copyright.
    files = ("table_a2.txt", "table_a3.txt", "table_a4.txt", "table_a5.txt")
    yllist = [parse_2010_file(os.path.join("1048800_Supplementary_Data",ff)) for ff in files]
    yields = dict(yllist[0])
    [yields.update(yl) for yl in yllist[1:]]
    return format_for_c("agb", yields)
