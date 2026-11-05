import os
import subprocess
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("CYBERFLY_DATA", Path.home() / ".cache" / "cyberfly" / "data"))
SOURCES = {
    "Drosophila_brain_model": "https://github.com/philshiu/Drosophila_brain_model",
    "flywire_annotations": "https://github.com/flyconnectome/flywire_annotations",
}


def download():
    DATA.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        if not (DATA / name).exists():
            subprocess.run(["git", "clone", "--depth", "1", url, str(DATA / name)], check=True)


@dataclass
class Connectome:
    ids: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    count: np.ndarray

    @classmethod
    def load(cls, version: str = "783") -> "Connectome":
        root = DATA / "Drosophila_brain_model"
        ids = pd.read_csv(root / f"Completeness_{version}.csv", index_col=0).index.values
        df = pd.read_parquet(root / f"Connectivity_{version}.parquet",
                             columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
        return cls(ids, df["Presynaptic_Index"].values, df["Postsynaptic_Index"].values,
                   df["Excitatory x Connectivity"].values)

    def __len__(self) -> int:
        return len(self.ids)

    @cached_property
    def annotations(self) -> pd.DataFrame:
        path = DATA / "flywire_annotations" / "supplemental_files" / "Supplemental_file1_neuron_annotations.tsv"
        a = pd.read_csv(path, sep="\t", low_memory=False)
        index = {r: i for i, r in enumerate(self.ids)}
        a = a[a.root_id.isin(index)].copy()
        a["index"] = a.root_id.map(index)
        return a

    def cells(self, types: list[str], side: str | None = None) -> list[int]:
        a = self.annotations
        m = a.cell_type.isin(types)
        if side:
            m &= a.side == side
        return a[m]["index"].tolist()


if __name__ == "__main__":
    download()
