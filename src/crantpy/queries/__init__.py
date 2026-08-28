import lazy_loader

__getattr__, __dir__, __all__ = lazy_loader.attach(
    __name__,
    submodules={
        "connections",
        "nested_connectivity_matrices",
        "neurons",
        "neuropils",
    },
    submod_attrs={
        "connections": [
            "attach_synapses",
            "get_adjacency",
            "get_connectivity",
            "get_synapse_counts",
            "get_synapses",
            "logger",
        ],
        "nested_connectivity_matrices": [
            "All",
            "DirectedNestedMatrix",
            "NestedMatrix",
            "NeuropilCollection",
        ],
        "neurons": [
            "NeuronCriteria",
            "get_annotations",
            "is_proofread",
            "parse_neuroncriteria",
        ],
        "neuropils": [
            "CAVE_ROW_LIMIT",
            "count_synapses_in_mesh",
            "get_synapses_in_mesh",
            "get_synapses_in_neuropils",
            "logger",
        ],
    },
)

__all__ = [
    "All",
    "CAVE_ROW_LIMIT",
    "DirectedNestedMatrix",
    "NestedMatrix",
    "NeuronCriteria",
    "NeuropilCollection",
    "attach_synapses",
    "connections",
    "count_synapses_in_mesh",
    "get_adjacency",
    "get_annotations",
    "get_connectivity",
    "get_synapse_counts",
    "get_synapses",
    "get_synapses_in_mesh",
    "get_synapses_in_neuropils",
    "is_proofread",
    "logger",
    "nested_connectivity_matrices",
    "neurons",
    "neuropils",
    "parse_neuroncriteria",
]
