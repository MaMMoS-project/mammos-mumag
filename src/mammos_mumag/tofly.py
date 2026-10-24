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


class _ParseError(Exception):
    pass


class _UnsupportedElementError(Exception):
    pass


class _EndOfFileError(Exception):
    pass


class _EndOfSectionError(Exception):
    pass


def _scanUnv(file, exclude):
    index = {}
    groups = {}
    nodes = []
    contact = set()
    t = _findSection(file)
    while t is not None:
        if t == UNV_NODES:
            _indexNodes(nodes, file)
        elif t == UNV_ELEMS:
            _indexElems(index, file, exclude)
        elif t == UNV_GROUPS:
            _parseGroups(groups, contact, file)
        else:
            _skipSection(file)
        t = _findSection(file)
    return nodes, index, groups, contact


def _findSection(file):
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


def _skipSection(file):
    line = file.readline()
    while not line.startswith(UNV_DELIM):
        line = file.readline()


def _indexNodes(nodes, file):
    data = (file.tell(), _countUnvNodes(file))
    nodes.append(data)


def _countUnvNodes(file):
    num = 7
    cnt = 0
    data = _parse(file, num, 0b1)
    while data:
        data = _parse(file, num, 0b1)
        cnt += 1
    return cnt


def _static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate


@_static_vars(rem=[])
def _parse(f, num, pattern):
    cache = _filter(_parse.rem, 0, pattern)
    wCnt = len(_parse.rem)
    _parse.rem = []
    line = ""
    words = []
    while wCnt < num:
        line = f.readline()
        if line.startswith(UNV_DELIM):
            break
        words = line.split()
        new = _filter(words, wCnt, pattern)
        cache.extend(new)
        wCnt += len(words)
    diff = wCnt - num
    if diff > 0:
        _parse.rem = words[len(words) - diff :]
    if cache and wCnt < num:
        raise _EndOfSectionError()
    return cache


def _filter(words, i, pattern):
    ret = []
    for v in words:
        if ((1 << i) & pattern) > 0:
            ret.append(v)
        i += 1
    return ret


def _indexElems(index, file, exclude):
    posPrev = file.tell()
    curr = _nextType(file)
    prev = curr
    count = 0
    while curr is not None:
        count += 1
        pos = file.tell()
        curr = _nextType(file)
        if curr != prev:
            if prev not in exclude:
                data = (posPrev, count)
                _regIndex(prev, data, index)
            posPrev = pos
            prev = curr
            count = 0


def _regIndex(t, data, index):
    list = index.get(t, [])
    list.append(data)
    index[t] = list


def _nextType(file):
    data = _parse(file, 6, 0b100010)
    if data:
        t = data[0]
        n = int(data[1])
        if t in UNV_BEAM:
            n += 3
        data = _parse(file, n, 0b1)
        if not data:
            raise _EndOfSectionError()
        return t
    return None


def _parseGroups(groups, contact, file):
    data = _parse(file, 9, 0b110000000)
    while data:
        num = int(data[0])
        group = data[1]
        elems = []
        while num > 0:
            data = _parse(file, 4, 0b0010)
            if not data:
                raise _EndOfSectionError()
            (entity,) = data
            elems.append(entity)
            num -= 1
        if group == CONTACT_GRP:
            contact |= set(elems)
        else:
            for e in elems:
                groups[e] = group
        data = _parse(file, 9, 0b110000000)


def _writeFly(nodes, groups, index, contact, unvFile, flyFile, exclude):
    _writeHeader(flyFile)
    _convertNodes(nodes, unvFile, flyFile)
    _convertElemsContact(index, groups, contact, unvFile, flyFile)
    if UNV_index[2].issubset(exclude):  # if 2D is excluded
        _writeFooter(flyFile)
    else:
        _writeFooter2(flyFile)


def _writeHeader(file):
    file.write(FLY_MESH_NAME + "\n")


def _convertNodes(nodes, unvFile, flyFile):
    sum = 0
    for pos, num in nodes:
        sum += num
    flyFile.write("3D-nodes %d\n" % sum)
    for pos, num in nodes:
        unvFile.seek(pos)
        while num > 0:
            nId, x, y, z = _parseNode(unvFile)
            flyFile.write(
                nId + " " + nId + " 0 " + str(x) + " " + str(y) + " " + str(z) + "\n"
            )
            num -= 1


def _writeFooter(fly):
    fly.write("""Tri3 0
Tri3_Contact 0
Point1 0
Tags
""")


def _writeFooter2(fly):
    fly.write("""Tri3_Contact 0
Point1 0
Tags
""")


def _parseNode(file):
    return _parse(file, 7, 0b1110001)


def _convertElemsContact(index, groups, contact, unv, fly):
    contactBuff = {}
    eCnt = 0
    for t, data in index.items():
        buff = []
        for pos, num in data:
            unv.seek(pos)
            while num > 0:
                eId, ns = _parseElem(t, unv)
                grp = groups.get(eId, "-1")
                line = grp + " " + " ".join(ns) + "\n"
                if eId in contact:
                    sif = MCONTACT[t]
                    _addTo(contactBuff, sif, line)
                else:
                    buff.append(line)
                num -= 1
        if buff:
            eCnt = _writeBuffer(fly, buff, t, MNORMAL)
    for sif, buff in contactBuff.items():
        _writeBuffer(fly, buff, sif, i=eCnt)


def _writeBuffer(f, b, t, m=None, i=1):
    if m is not None:
        t = m[t]
    f.write("%s %d\n" % (t, len(b)))
    for ll in b:
        f.write(str(i) + " " + ll)
        i += 1
    return i


def _addTo(m, k, d):
    ls = m.get(k, [])
    ls.append(d)
    m[k] = ls


def _parseElem(t, file):
    eId, t, nStr = _parse(file, 6, 0b100011)
    n = int(nStr)
    p = ~0b0
    if t in UNV_BEAM:
        n += 3
        p = ~0b111
    data = _parse(file, n, p)
    return eId, data


def _get_exclude_set(exclude_list):
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
    exclude_set = _get_exclude_set(exclude_list)
    nodes, index, groups, contact = _scanUnv(infile, exclude_set)
    _writeFly(nodes, groups, index, contact, infile, outfile, exclude_set)
    infile.close()
    outfile.close()
