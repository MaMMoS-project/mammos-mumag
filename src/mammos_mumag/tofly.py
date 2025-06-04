"""Convert unv mesh files to the fly format.

usage: tofly.py [-h] [-e DIMENSIONS] [UNV] [FLY]

Convert unv files to the fly format. Elements that belong to a group called
'contact' will be converted to their contact counterparts. First and secound
order meshes are supported.

positional arguments:
  UNV                   Path to the input file or '-' for stdin. It must
                        already exist and be stored in the unv format. If
                        ommited stdin will be used instead.
  FLY                   Path to the output file or '-' for stdout. Overridden
                        if it already exists. If ommited stdout will be used
                        instead.

optional arguments:
  -h, --help            show this help message and exit
  -e DIMENSIONS, --exclude DIMENSIONS
                        Comma separated list of dimensions that shall be
                        ignored while converting (e.g. '-e 1,2' only converts
                        3D elements).
"""

import functools
import pathlib

CONTACT_GRP = "contact"

UNV_DELIM = "    -1"

UNV_NODES = "2411"
UNV_ELEMS = "2412"
UNV_GROUPS = "2467"

UNV_BEAM = set(["11", "21", "22", "23", "24"])
UNV_index = {
    1: set(["11", "21", "22", "24"]),
    2: set(["41", "81", "91", "42", "82", "92", "44"]),
    3: set(["115", "116", "111", "118"]),
}
FLY_MESH_NAME = "Mesh"

FLY_NODES = "Nodes"
FLY_LINE2 = "Line2"
FLY_LINE3 = "Line3"
FLY_TRI3 = "Tri3"
FLY_TRI6 = "Tri6"
FLY_REC4 = "Rec4"
FLY_REC8 = "Rec8"
FLY_REC9 = "Rec9"
FLY_TET4 = "Tet4"
FLY_TET10 = "Tet10"
FLY_HEX8 = "Hex8"
FLY_HEX20 = "Hex20"
FLY_HEX27 = "Hex27"

FLY_LINE2_CONTACT = "Line2_Contact"
FLY_LINE3_CONTACT = "Line3_Contact"
FLY_TRI3_CONTACT = "Tri3_Contact"
FLY_TRI6_CONTACT = "Tri6_Contact"
FLY_REC4_CONTACT = "Rec4_Contact"
FLY_REC8_CONTACT = "Rec8_Contact"

MNORMAL = {
    "11": FLY_LINE2,
    "21": FLY_LINE2,
    "22": FLY_LINE3,
    "24": FLY_LINE3,
    "41": FLY_TRI3,
    "81": FLY_TRI3,
    "91": FLY_TRI3,
    "42": FLY_TRI6,
    "82": FLY_TRI6,
    "92": FLY_TRI6,
    "44": FLY_REC4,
    "115": FLY_HEX8,
    "116": FLY_HEX20,
    "111": FLY_TET4,
    "118": FLY_TET10,
}

MCONTACT = {
    "115": FLY_REC4_CONTACT,
    "116": FLY_REC8_CONTACT,
    "112": FLY_TRI3_CONTACT,
}


class ParseError(Exception):
    pass


class UnsupportedElementError(Exception):
    pass


class EndOfFileError(Exception):
    pass


class EndOfSectionError(Exception):
    pass


def scanUnv(file, exclude):
    index = {}
    groups = {}
    nodes = []
    contact = set()
    t = findSection(file)
    while t is not None:
        if t == UNV_NODES:
            indexNodes(nodes, file)
        elif t == UNV_ELEMS:
            indexElems(index, file, exclude)
        elif t == UNV_GROUPS:
            parseGroups(groups, contact, file)
        else:
            skipSection(file)
        t = findSection(file)
    return nodes, index, groups, contact


def findSection(file):
    secType = ""
    line = file.readline()
    while line and secType == "":
        if line.startswith(UNV_DELIM):
            secType = file.readline().strip()
        else:
            line = file.readline()
    if secType == "":
        return None
    return secType


def skipSection(file):
    line = file.readline()
    while not line.startswith(UNV_DELIM):
        line = file.readline()


def indexNodes(nodes, file):
    data = (file.tell(), countUnvNodes(file))
    nodes.append(data)


def countUnvNodes(file):
    num = 7
    cnt = 0
    data = parse(file, num, 0b1)
    while data:
        data = parse(file, num, 0b1)
        cnt += 1
    return cnt


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


@static_vars(rem=[])
def parse(f, num, pattern):
    cache = filter(parse.rem, 0, pattern)
    wCnt = len(parse.rem)
    parse.rem = []
    line = ""
    words = []
    while wCnt < num:
        line = f.readline()
        if line.startswith(UNV_DELIM):
            break
        words = line.split()
        new = filter(words, wCnt, pattern)
        cache.extend(new)
        wCnt += len(words)
    diff = wCnt - num
    if diff > 0:
        parse.rem = words[len(words) - diff :]
    if cache and wCnt < num:
        raise EndOfSectionError()
    return cache


def filter(words, i, pattern):
    ret = []
    for v in words:
        if ((1 << i) & pattern) > 0:
            ret.append(v)
        i += 1
    return ret


def indexElems(index, file, exclude):
    posPrev = file.tell()
    curr = nextType(file)
    prev = curr
    count = 0
    while curr is not None:
        count += 1
        pos = file.tell()
        curr = nextType(file)
        if curr != prev:
            if prev not in exclude:
                data = (posPrev, count)
                regIndex(prev, data, index)
            posPrev = pos
            prev = curr
            count = 0


def regIndex(t, data, index):
    list = index.get(t, [])
    list.append(data)
    index[t] = list


def nextType(file):
    data = parse(file, 6, 0b100010)
    if data:
        t = data[0]
        n = int(data[1])
        if t in UNV_BEAM:
            n += 3
        data = parse(file, n, 0b1)
        if not data:
            raise EndOfSectionError()
        return t
    return None


def parseGroups(groups, contact, file):
    data = parse(file, 9, 0b110000000)
    while data:
        num = int(data[0])
        group = data[1]
        elems = []
        while num > 0:
            data = parse(file, 4, 0b0010)
            if not data:
                raise EndOfSectionError()
            (entity,) = data
            elems.append(entity)
            num -= 1
        if group == CONTACT_GRP:
            contact |= set(elems)
        else:
            for e in elems:
                groups[e] = group
        data = parse(file, 9, 0b110000000)


def writeFly(nodes, groups, index, contact, unvFile, flyFile, exclude):
    writeHeader(flyFile)
    convertNodes(nodes, unvFile, flyFile)
    convertElemsContact(index, groups, contact, unvFile, flyFile)
    if UNV_index[2].issubset(exclude):  # if 2D is excluded
        writeFooter(flyFile)
    else:
        writeFooter2(flyFile)


def writeHeader(file):
    file.write(FLY_MESH_NAME + "\n")


def convertNodes(nodes, unvFile, flyFile):
    sum = 0
    for pos, num in nodes:
        sum += num
    flyFile.write("3D-nodes %d\n" % sum)
    for pos, num in nodes:
        unvFile.seek(pos)
        while num > 0:
            nId, x, y, z = parseNode(unvFile)
            flyFile.write(
                nId + " " + nId + " 0 " + str(x) + " " + str(y) + " " + str(z) + "\n"
            )
            num -= 1


def writeFooter(fly):
    fly.write("""Tri3 0
Tri3_Contact 0
Point1 0
Tags
""")


def writeFooter2(fly):
    fly.write("""Tri3_Contact 0
Point1 0
Tags
""")


def parseNode(file):
    return parse(file, 7, 0b1110001)


def convertElemsContact(index, groups, contact, unv, fly):
    contactBuff = {}
    eCnt = 0
    for t, data in index.items():
        buff = []
        for pos, num in data:
            unv.seek(pos)
            while num > 0:
                eId, ns = parseElem(t, unv)
                grp = groups.get(eId, "-1")
                line = grp + " " + " ".join(ns) + "\n"
                if eId in contact:
                    sif = MCONTACT[t]
                    addTo(contactBuff, sif, line)
                else:
                    buff.append(line)
                num -= 1
        if buff:
            eCnt = writeBuffer(fly, buff, t, MNORMAL)
    for sif, buff in contactBuff.items():
        writeBuffer(fly, buff, sif, i=eCnt)


def writeBuffer(f, b, t, m=None, i=1):
    if m is not None:
        t = m[t]
    f.write("%s %d\n" % (t, len(b)))
    for ll in b:
        f.write(str(i) + " " + ll)
        i += 1
    return i


def addTo(m, k, d):
    ls = m.get(k, [])
    ls.append(d)
    m[k] = ls


def parseElem(t, file):
    eId, t, nStr = parse(file, 6, 0b100011)
    n = int(nStr)
    p = ~0b0
    if t in UNV_BEAM:
        n += 3
        p = ~0b111
    data = parse(file, n, p)
    return eId, data


def get_exclude_set(exclude_list):
    return functools.reduce(
        lambda x, y: x.union(y),
        [UNV_index[i] for i in exclude_list],
        set(),
    )


def convert(
    unv_path: str | pathlib.Path,
    fly_path: str | pathlib.Path,
    exclude_list: list[int] = [1, 2],
) -> None:
    """Convert mesh file from `unv` to `fly`.

    Args:
        unv_path: Input `unv` file path.
        fly_path: Output `fly` file path.
        exclude_list: List of dimensions to be excluded. Defaults to [1,2], so it will
            exclude 1D and 2D elements.

    """
    infile = open(unv_path)
    pathlib.Path(fly_path).parent.mkdir(exist_ok=True, parents=True)
    outfile = open(fly_path, "w")
    exclude_set = get_exclude_set(exclude_list)
    nodes, index, groups, contact = scanUnv(infile, exclude_set)
    writeFly(nodes, groups, index, contact, infile, outfile, exclude_set)
    infile.close()
    outfile.close()
