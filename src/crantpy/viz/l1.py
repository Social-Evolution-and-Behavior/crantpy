# -*- coding: utf-8 -*-
"""L1 visualization module for CRANTBpy."""

import functools
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

import numpy as np
import navis
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from crantpy.utils.cave import get_cave_client
from crantpy.utils.config import CRANT_VALID_DATASETS
from crantpy.utils.decorators import inject_dataset, parse_neuroncriteria
from crantpy.queries.neurons import NeuronCriteria
from crantpy.utils.helpers import parse_root_ids, retry
from crantpy.viz.mesh import get_mesh_neuron


def dotprops_from_mesh(
    mesh: "navis.MeshNeuron",
    k: int = None,
    resample: float = None,
    units: str = "1 nm",
    **kwargs,
) -> "navis.Dotprops":
    """Generate Dotprops from a mesh.

    Parameters
    ----------
    mesh : navis.MeshNeuron
        Mesh of the neuron (coordinates in nm).
    k : int, optional
        k for Dotprops (size of neighbourhood; passed to `navis.make_dotprops`).
    resample : float, optional
        If given, resample skeleton (in nm) before making dotprops.
    units : str
        Units of coordinates. Default '1 nm'.
    **kwargs
        Passed to `navis.make_dotprops`.

    Returns
    -------
    navis.Dotprops
    """
    if mesh is None or not len(mesh.vertices):
        raise ValueError("Empty mesh, cannot make dotprops")

    # 1) skeletonize the mesh
    sk = navis.conversion.mesh2skeleton(mesh)   # uses navis built-in mesh skeletoniser

    # 2) optional resample
    if resample is not None:
        sk = navis.resample_skeleton(sk, resample)

    # 3) turn skeleton into dotprops
    dp = navis.make_dotprops(sk, k=k, **kwargs)
    dp.id = getattr(mesh, "id", None)
    dp.units = units

    return dp




def get_skeleton(root_id: Union[int, str], dataset: Optional[str] = None, level: int = 1) -> Optional["navis.TreeNeuron"]:
    """Fetch skeleton for a given neuron at specified level.

    Parameters
    ----------
    root_id : int or str
        Root ID of the neuron.
    dataset : str, optional
        Dataset to fetch from.
    level : int
        LOD level for the skeleton (1 for L1).

    Returns
    -------
    navis.TreeNeuron or None
        Skeleton neuron or None if not available.
    """
    client = get_cave_client(dataset=dataset)
    try:
        # Fetch skeleton with level of detail
        skeleton = client.chunkedgraph.get_skeleton(root_id, remove_duplicate_nodes=True)
        if skeleton is None or len(skeleton) == 0:
            return None
        # Convert to navis skeleton
        tn = navis.TreeNeuron(skeleton, id=root_id, units="1 nm")
        return tn
    except Exception as e:
        logging.warning(f"Failed to fetch skeleton for {root_id}: {e}")
        return None


@parse_neuroncriteria()
@inject_dataset()
def get_l1_dotprops(
    root_ids: Union[int, str, List[Union[int, str]], "NeuronCriteria"],
    resample: Optional[float] = None,
    omit_failures: Optional[bool] = None,
    progress: bool = True,
    max_threads: int = 10,
    dataset: Optional[str] = None,
    **kwargs,
) -> "navis.NeuronList":
    """Generate dotprops from L1 meshes for given neuron(s).

    Parameters
    ----------
    root_ids : int, str, list, np.ndarray, or NeuronCriteria
        Root ID(s) of the neuron(s) to generate dotprops for.
    resample : float, optional
        If given, resample skeleton (in nm) before computing dotprops.
    omit_failures : bool, optional
        Behaviour when dotprops generation fails (mesh missing/empty).
        None (default) raises, True skips, False returns empty Dotprops.
    progress : bool
        Whether to show a progress bar.
    max_threads : int
        Number of parallel requests to make when fetching meshes.
    dataset : str, optional
        Dataset to query. If None, will use the default dataset.
    **kwargs
        Additional keyword arguments passed to dotprops_from_mesh.

    Returns
    -------
    navis.NeuronList
        List of Dotprops in microns.
    """
    if omit_failures not in (None, True, False):
        raise ValueError(
            "`omit_failures` must be either None, True or False. "
            f'Got "{omit_failures}".'
        )

    # Normalize input
    root_ids = parse_root_ids(root_ids)

    # Fetch meshes in parallel
    def _get_mesh_with_error_handling(root):
        try:
            mesh = get_mesh_neuron(
                root,
                dataset=dataset,
                omit_failures=False,
                threads=1,
                progress=False,
            )
            return mesh
        except Exception as e:
            logging.warning(f"Failed to fetch mesh for {root}: {e}")
            return None

    meshes = []
    with ThreadPoolExecutor(max_workers=max_threads) as pool:
        futures = pool.map(_get_mesh_with_error_handling, root_ids)
        for mesh in navis.config.tqdm(
            futures,
            desc="Fetching L1 meshes",
            total=len(root_ids),
            disable=not progress or len(root_ids) == 1,
            leave=False,
        ):
            meshes.append(mesh)

    # Convert each mesh to Dotprops
    dps = []
    for root, mesh in navis.config.tqdm(
        zip(root_ids, meshes),
        desc="Creating L1 dotprops",
        total=len(root_ids),
        disable=not progress or len(root_ids) <= 1,
        leave=False,
    ):
        if mesh is None or not len(mesh.vertices):
            msg = f"Unable to create L1 dotprops: no mesh for root ID {root}."
            if omit_failures is None:
                raise ValueError(msg)
            if not omit_failures:
                dp = navis.Dotprops(None, k=None, id=root, units="1 um", **kwargs)
                dps.append(dp)
            continue

        try:
            # Generate dotprops from mesh
            dp = dotprops_from_mesh(mesh, resample=resample, units="1 nm", **kwargs)
            dps.append(dp)
        except Exception as e:
            msg = f"Unable to create L1 dotprops for root ID {root}: {e}"
            if omit_failures is None:
                raise ValueError(msg) from e
            if not omit_failures:
                dp = navis.Dotprops(None, k=None, id=root, units="1 um", **kwargs)
                dps.append(dp)

    # Convert dotprops from nanometers to microns
    for dp in dps:
        if dp.points is not None:
            dp.points = dp.points / 1000
            dp.units = "1 um"

    return navis.NeuronList(dps)