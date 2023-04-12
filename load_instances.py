#!/usr/bin/env python3


import networkit as nk  
import os
import os.path


instance_str = """\
instdir: "instances"
instances:
  - repo: local
    items: """


def dl(url, name):
    return os.system(f"""\
if [ -f "/tmp/{name}" ]; then
    echo "{name} already downloaded."
    false
else
    wget {url} -O /tmp/{name}
fi""")

def graph_inst(g, name):
    global instance_str
    instance_str += "\n      - " + name + ".nkb"
    _g = nk.components.ConnectedComponents.extractLargestConnectedComponent(g, True)
    _g.removeMultiEdges()
    _g.sortEdges()
    nk.graphio.NetworkitBinaryWriter().write(_g, "instances/"+name+".nkb")

def txt_to_inst(name, sep=" ", first_node=0, comment_prefix="#", continuous=True, input_file_name=None):
    if input_file_name == None:
        input_file_name = "instances/" + name + ".txt"
    g = nk.graphio.EdgeListReader(sep, first_node, comment_prefix, continuous).read(input_file_name)
    graph_inst(g, name)

def dl_txt_gz(url, name):
    if os.path.isfile("instances/{0}.nkb".format(name)):
        return False
    dl(url, name)
    error = os.system(f"gunzip /tmp/{name} -c > instances/{name}")

def mtx_to_inst(in_path, name):
    # Remove comments and first line
    s = ""
    first_line = True
    with open(in_path, "r") as f:
        for line in f:
            if line[0] != "%":
                if first_line:
                    first_line = False
                else:
                    s += line + "\n"
    with open(f"instances/{name}.txt", "w") as f2:
        f2.write(s)
    txt_to_inst(name, ' ', 1, "%")

def dl_tar_bz2(url, archive_path_to_instance, name):
    if os.path.isfile("instances/{0}.nkb".format(name)):
        return False
    dl(url, name+".tar.bz2")
    os.system('tar -xf /tmp/{0}.tar.bz2 -C /tmp/'.format(name))
    os.system('mv /tmp/{0} instances/{1}'.format(archive_path_to_instance, name))

def dl_zip(url, archive_path_to_instance, name):
    if os.path.isfile(f"instances/{name}.nkb"):
        return False
    dl(url, name+".zip")
    os.system(f"unzip -o /tmp/{name}.zip -d /tmp/")

def gen_er_inst(n, p, seed=1):
    name = f"erdos_renyi_{n}_{p}"
    if seed != 1:
        name += f"_{seed}"
    if os.path.isfile(f"instances/{name}.nkb"):
        return

    nk.setSeed(seed, True)
    g = nk.generators.ErdosRenyiGenerator(n, p).generate()
    graph_inst(g, name)

def gen_ba_inst(k, nMax, n0, seed=1):
    name = "barabasi_albert_{0}_{1}_{2}".format(k, nMax, n0)
    if seed != 1:
        name += f"_{seed}"
    if os.path.isfile(f"instances/{name}.nkb"):
        return

    nk.setSeed(seed, True)
    g = nk.generators.BarabasiAlbertGenerator(k, nMax, n0).generate()
    graph_inst(g, name)

def gen_ws_inst(nNodes, nNeighbors, p, seed=1):
    name = "watts_strogatz_{0}_{1}_{2}".format(nNodes, nNeighbors, p)
    if seed != 1:
        name += f"_{seed}"
    if os.path.isfile(f"instances/{name}.nkb"):
        return

    nk.setSeed(seed, True)
    g = nk.generators.WattsStrogatzGenerator(nNodes, nNeighbors, p).generate()
    graph_inst(g, name)

def load_snap_stanford(name):
    dl_txt_gz(f"https://snap.stanford.edu/data/{name}.txt.gz", name)
    txt_to_inst(name, "\t", 0, "#", False, f"instances/{name}")

if __name__ == "__main__":
    if not os.path.isdir("instances"):
        os.system("mkdir instances")

    dl_txt_gz("https://snap.stanford.edu/data/facebook_combined.txt.gz", "facebook_ego_combined")
    txt_to_inst("facebook_ego_combined", "\t", 0, "#", False, "instances/facebook_ego_combined")

    dl_txt_gz("https://snap.stanford.edu/data/ca-AstroPh.txt.gz", "arxiv-astro-ph")
    txt_to_inst("arxiv-astro-ph", "\t", 0, "#", False, "instances/arxiv-astro-ph")

    dl_txt_gz("https://snap.stanford.edu/data/ca-HepPh.txt.gz", "arxiv-heph")
    txt_to_inst("arxiv-heph", "\t", 0, "#", False, "instances/arxiv-heph")

    load_snap_stanford("wiki-Vote")
    load_snap_stanford("p2p-Gnutella09")
    load_snap_stanford("p2p-Gnutella04")
    load_snap_stanford("as-caida20071105")
    load_snap_stanford("cit-HepTh")
    load_snap_stanford("loc-brightkite_edges")
    load_snap_stanford("soc-Slashdot0902")

    dl_zip("https://nrvis.com/download/data/inf/inf-power.zip", "inf-power.mtx", "inf-power")
    mtx_to_inst("/tmp/inf-power.mtx", "inf-power")

    dl_zip("https://nrvis.com/download/data/ia/ia-email-EU.zip", "ia-email-EU.mtx", "ia-email-EU")
    mtx_to_inst("/tmp/ia-email-EU.mtx", "ia-email-EU")

    dl_zip("https://nrvis.com/download/data/web/web-spam.zip", "web-spam.mtx", "web-spam")
    mtx_to_inst("/tmp/web-spam.mtx", "web-spam")

    dl_zip("https://nrvis.com/download/data/web/web-webbase-2001.zip", "web-webbase-2001.mtx", "web-webbase-2001")
    mtx_to_inst("/tmp/web-webbase-2001.mtx", "web-webbase-2001")

    dl_zip("https://nrvis.com/download/data/web/web-indochina-2004.zip", "web-indochina-2004.mtx", "web-indochina-2004")
    mtx_to_inst("/tmp/web-indochina-2004.mtx", "web-indochina-2004")

    dl_zip("https://nrvis.com/download/data/ia/ia-wiki-Talk.zip", "ia-wiki-Talk.mtx", "ia-wiki-Talk")
    mtx_to_inst("/tmp/ia-wiki-Talk.mtx", "ia-wiki-Talk")

    dl_zip("https://nrvis.com/download/data/soc/soc-LiveMocha.zip", "soc-LiveMocha.mtx", "soc-LiveMocha")
    mtx_to_inst("/tmp/soc-LiveMocha.mtx", "soc-LiveMocha")

    dl_zip("https://nrvis.com/download/data/road/road-usroads.zip", "road-usroads.mtx", "road-usroads")
    mtx_to_inst("/tmp/road-usroads.mtx", "road-usroads")
    
    # maria: konect.cc not connecting..
    # dl_tar_bz2("http://konect.cc/files/download.tsv.flickrEdges.tar.bz2", "flickrEdges/out.flickrEdges", "flickr")
    # txt_to_inst("flickr", " ", 1, "%")

    gen_er_inst(1000, 0.02)

    gen_ws_inst(1000, 7, 0.3)

    gen_ba_inst(2, 1000, 2)

    print(instance_str)