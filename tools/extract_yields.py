"""Parse a yield table in the format used by Karakas and collaborators and output it in the raw C format we prefer.

There is what I assume to be a typo in Karaks 2010 table A3: M = 2 is listed as M= 2.1"""
import re
import os.path
import numpy as np

def parse_file(fname, karakas=True):
    """Parse a yield file. Looks for the end of each mass/metallicity bin
    Returns a dictionary of total metal yields."""
    yields = {}
    with open(fname) as fn:
        #Get first header
        line = fn.readline()
        head = parse_step_header(line, karakas=karakas)
        line = fn.readline()
        #Read lines, stopping when done.
        laststep = []
        while line != "":
            lhead = parse_step_header(line, karakas=karakas)
            if lhead is not None:
                yielddata = parse_full_bin(laststep, karakas=karakas)
                if head is None:
                    raise Exception
                yields[head] = yielddata
                #New step
                laststep = []
                head = lhead
            else:
                #Accumulate non-header lines
                laststep += [line,]
            line = fn.readline()
        #Process final bin
        yielddata = parse_full_bin(laststep, karakas=karakas)
        yields[head] = yielddata
    return yields

def parse_step_header(line, karakas=True):
    """Parse a string describing a mass, metallicity bin.
    Returns None if the line is not a bin header, otherwise a tuple of mass, metallicity and ejected mass."""
    #This flags whether we are looking in Karakas 2010 or Doherty 2014.
    if karakas:
        regex = r"# Minitial =  ([\.0-9]+) msun, Z = ([\.0-9]+), Mfinal = ([\.0-9]+) msun"
    else:
        regex = r"\s+([\.0-9]+)M Z=([\.0-9]+) VW93"
    reg = re.match(regex, line)
    if reg is None:
        return None
    grps = reg.groups()
    initmass = float(grps[0])
    #Read massloss from the table so we can reuse code for 2014 table
    #initmass - float(grps[2])
    return initmass, float(grps[1])

def parse_full_bin(textdata, karakas=True):
    """Split a line of a yield table into a yield number and an elemental species"""
    metalnames = ('H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'Fe', 'Z', 'ej')
    #Zero yield for this bin
    yielddata = {mm : 0 for mm in metalnames}
    for line in textdata:
        #Regex for a yield line. If this doesn't match, return nothing.
        if karakas:
            #Extra column with atomic number in Karakas.
            regex = r"\s+([a-z0-9]+)\s+[0-9]+\s+([\.0-9E\-\+]+)\s+([\.0-9E\-\+]+)"
        else:
            regex = r"\s+([a-z0-9]+)\s+([\.0-9E\-\+]+)\s+([\.0-9E\-\+]+)"
        reg = re.match(regex, line)
        if reg is None:
            continue
        grps = reg.groups()
        species = parse_species(grps[0], metalnames)
        yy = float(grps[2])
        #Add individual tracked species
        if species is not None:
            yielddata[species] += yy
        #if species == 'H':
            #print(grps[1], reg)
        #Add non-H, non-He to total metals
        if species not in ('H', 'He'):
            yielddata['Z'] += yy
        yielddata['ej'] += float(grps[2])
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

def format_for_c(arrayname, yields, agb=True):
    """Format the parsed yield data into a C style array. If agb is True, do some filtering."""
    #Masses and metallicities
    out = """#define %(uname)s_NMET %(zsize)d
#define %(uname)s_NMASS %(msize)d
static const double %(name)s_masses[%(uname)s_NMASS] = { %(masses)s };
static const double %(name)s_metallicities[%(uname)s_NMET] = { %(metals)s };
static const double %(name)s_total_mass[%(uname)s_NMET*%(uname)s_NMASS] = {
%(eject)s
};

static const double %(name)s_total_metals[%(uname)s_NMET*%(uname)s_NMASS] = {
%(totmet)s
};

static const double %(name)s_yield[NSPECIES][%(uname)s_NMET*%(uname)s_NMASS] = {
%(yields)s
};

    """
    (mass, metals) = zip(*yields.keys())
    mass = sorted(list(set(mass)))
    metals = sorted(list(set(metals)))
    if agb:
        #Stars with M >= 8 are in the SnII table
        mass = [mm for mm in mass if mm < 8.0]
        #Remove the Z=0.001 bin which is only available for large masses
        metals = [zz for zz in metals if (zz > 0.002)+(zz < 0.0009)]
    cmass = ','.join('%.2f' % m for m in mass)
    cmet = ','.join('%.4f' % m for m in metals)
    cejected = format_c_array(yields, mass, metals, 'ej')
    ctotmet = format_c_array(yields, mass, metals, 'Z', '%.3e')
    metalnames = ('H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'Fe')
    cyield = '{'+'\n},\n{\n'.join([format_c_array(yields, mass, metals, mm, '%.6e') for mm in metalnames])+'}'
    cstring = out % dict(name = arrayname, uname = arrayname.upper(),
                         msize = len(mass), zsize = len(metals),
                         masses = cmass, metals = cmet,
                         eject=cejected, totmet=ctotmet, yields=cyield)
    return cstring

def get_all_agb():
    """Get the tables for all AGB stars."""
    # First Karakas 2010. These tables are supplementary data from the journal archive.
    files = ("table_a2.txt", "table_a3.txt", "table_a4.txt", "table_a5.txt")
    yllist = [parse_file(os.path.join("../yield_data/agb",ff)) for ff in files]
    #Get the Doherty 2014 first table
    yields = parse_file("../yield_data/agb/TABLE1-VW93ML.txt", karakas=False)
    yields.update(parse_file("../yield_data/agb/P3Doh14b-table1.txt", karakas=False))
    [yields.update(yl) for yl in yllist]
    return format_for_c("agb", yields)

def parse_snii_species(string, metalnames):
    """Extract a species from the SnII string, which has the atomic number in front"""
    #string = str(string, 'utf-8')
    # Preserve a mass row
    if string == "M_cut_" or string == "M_final_":
        return string
    for mm in metalnames:
        if mm == 'H':
            if string in ('p', 'd'):
                return 'H'
        elif re.match('.+'+mm+'$', string) is not None:
            return mm
    #Default to returning nothing: means 'unknown element'
    return None

def parse_snii_file(filename):
    """Parse out the yields from the SnII file."""
    #Automagically does the parsing
    snii = np.genfromtxt(filename, dtype=None, encoding='utf-8')
    # Comes as a tuple.
    # Note that different metallicities have different mfinal,
    # accounting for mass loss by stellar winds. We assume that
    # all this lost mass is of the same yield as the input.
    masses = list(snii[0])[2:]
    metallicities = sorted(list({ii[0] for ii in snii}))
    metalnames = ('H', 'He', 'C', 'N', 'O', 'Ne', 'Mg', 'Si', 'Fe', 'Z', 'ej')
    # Init the yield structure
    yields = {}
    for ma in masses:
        for zz in metallicities:
            yields[(ma, zz)] = {mm : 0 for mm in metalnames}
    for row in snii:
        spec = parse_snii_species(row[1], metalnames)
        for i in range(len(masses)):
            #Ejected mass
            if row[1] == 'M_cut_':
                yields[(masses[i], row[0])]['ej'] -= row[i+2]
                continue
            # Final mass
            if row[1] == 'M_final_':
                yields[(masses[i], row[0])]['ej'] += row[i+2]
                continue
            # Yield for this species
            if spec is not None:
                yields[(masses[i], row[0])][spec] += row[i+2]
            # Total metal yield, including unknown elements
            if spec not in ('H', 'He'):
                yields[(masses[i], row[0])]['Z'] += row[i+2]
    return yields

def get_all_snii():
    """Get the tables for all SNII."""
    # First Kobayashi 2006, M = 13 - 40.
    yields = parse_snii_file("../yield_data/snii_kabayashi_2006.txt")
    return format_for_c("snii", yields, agb=False)
