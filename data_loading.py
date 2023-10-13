import numpy as np


def load_lastfm():
    filename = "data/lasftm_asia/lastfm_asia_edges.csv"
    return np.genfromtxt(filename, delimiter=",", dtype=np.uint16)[1:]


def load_ca_hep_ph():
    filename = "data/CA-HepPh.txt"
    data = np.genfromtxt(filename, delimiter="\t", dtype=np.uint)
    return data[data[:, 0] < data[:, 1]]


def load_ca_hep_th():
    filename = "data/CA-HepTh.txt"
    data = np.genfromtxt(filename, delimiter="\t", dtype=np.uint)
    return data[data[:, 0] < data[:, 1]]


def load_youtube():
    filename = "data/com-youtube.ungraph.txt"
    data = np.genfromtxt(filename, delimiter="\t", dtype=np.uint)
    return data[data[:, 0] < data[:, 1]]


def load_facebook(graph_id: int):
    filename = f"data/facebook/{graph_id}.edges"
    data = np.genfromtxt(filename, delimiter=" ", dtype=np.uint16)[1:]
    return data[data[:, 0] < data[:, 1]]


def main():
    data = load_youtube()
    print(data)
    print(np.unique(data).size)
    print(data.shape)


if __name__ == "__main__":
    main()
