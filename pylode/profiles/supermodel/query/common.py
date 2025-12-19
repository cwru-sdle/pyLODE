import logging
import sys
from itertools import chain

from rdflib import URIRef, Graph, Dataset, Literal, RDFS, SKOS, SDO, DCTERMS

from pylode.profiles.supermodel.model import Class, Ontology

logger = logging.getLogger(__name__)

# =========================
# ===== DEBUG SWITCH ======
# =========================

# Set this to False to turn off the depth/class tracing.
DEBUG_RECURSION = True


def _dbg(msg: str) -> None:
    """Simple debug printer for recursion tracing."""
    if DEBUG_RECURSION:
        print(msg, file=sys.stderr)


# =========================
# ===== HELPER FUNCS ======
# =========================


def get_values(
    iri: URIRef, graph: Graph | Dataset, properties: list[URIRef]
) -> list[URIRef | Literal]:
    result = list(
        chain.from_iterable([graph.objects(iri, prop) for prop in properties])
    )

    for value in result:
        if not isinstance(value, (URIRef, Literal)):
            raise ValueError(
                f"Expected only IRIs or literals but found type {type(value)} "
                f"with value {value} for IRI {iri}"
            )

    return result


def get_name(iri: URIRef, graph: Graph, db: Dataset = None) -> str:
    """Get name for resource.

    If no name found in the profile graph, look in the union dataset.
    If still no name, fall back to a qname or full IRI.
    """
    name_predicates = [RDFS.label, SKOS.prefLabel, SDO.name]

    names = get_values(iri, graph, name_predicates)

    if not names and db is not None:
        names = get_values(iri, db, name_predicates)

    if not names:
        try:
            names.append(graph.qname(iri))
        except ValueError as err:
            logger.warning(
                f"Failed to create a qname for IRI {iri}. Reason: {err}. "
                "Adding full IRI as name instead."
            )

    return str(names[0]) if len(names) > 0 else str(iri)


def get_descriptions(iri: URIRef, graph: Graph) -> str | None:
    descriptions = get_values(
        iri, graph, [SKOS.definition, DCTERMS.description, SDO.description]
    )
    return (
        " ".join(sorted(str(i) for i in descriptions))
        if len(descriptions) > 0
        else None
    )


# =========================
# ===== CLASS QUERIES =====
# =========================


def get_class(
    iri: URIRef,
    graph: Graph,
    db: Dataset,
    ignored_classes: list[URIRef],
    depth: int = 0,
    trail: list[URIRef] | None = None,
) -> Class:
    """Get a Class model for an IRI, with subclasses.

    depth  – current depth in the subclass tree (for debug printing)
    trail  – list of IRIs visited on this recursion path (for cycle detection)
    """
    if trail is None:
        trail = []

    indent = "  " * depth
    _dbg(f"{indent}[get_class] depth={depth} iri={iri}")

    # Simple cycle detection: if we've seen this IRI on this path, log and stop.
    if iri in trail:
        _dbg(f"{indent}>>> CYCLE DETECTED at iri={iri}")
        for i, node in enumerate(trail + [iri]):
            _dbg(f"{indent}    {i:03d}: {node}")
        name = get_name(iri, graph, db)
        # Break the cycle: return class with no subclasses.
        return Class(iri, name, subclasses=[])

    new_trail = trail + [iri]

    name = get_name(iri, graph, db)
    subclasses = get_subclasses(
        iri,
        graph,
        db,
        ignored_classes,
        depth=depth,
        trail=new_trail,
    )
    return Class(iri, name, subclasses=subclasses)


def get_subclasses(
    iri: URIRef,
    graph: Graph,
    db: Dataset,
    ignored_classes: list[URIRef],
    depth: int = 0,
    trail: list[URIRef] | None = None,
) -> list[Class]:
    """Get subclasses of a class IRI, as Class models.

    depth  – depth of the *parent* iri in the tree (child depth = depth + 1)
    trail  – recursion trail passed down from get_class for debugging/cycles
    """
    if trail is None:
        trail = []

    indent = "  " * depth
    _dbg(f"{indent}[get_subclasses] depth={depth} parent={iri}")

    # Find subclass IRIs: ?sub rdfs:subClassOf iri
    subclass_iris = list(graph.subjects(RDFS.subClassOf, iri))

    # Filter out ignored classes and non-URIRefs
    subclass_iris = [
        x for x in subclass_iris if x not in ignored_classes and isinstance(x, URIRef)
    ]

    # Avoid trivial self-loop immediately
    subclass_iris = [s for s in subclass_iris if s != iri]

    results: list[Class] = []
    for subclass in subclass_iris:
        _dbg(f"{indent}  edge: {iri} -> {subclass}")
        results.append(
            get_class(
                subclass,
                graph,
                db,
                ignored_classes,
                depth=depth + 1,
                trail=trail,
            )
        )

    # Sort by name for stable output
    return sorted(results, key=lambda x: x.name)


# =========================
# ===== OTHER QUERIES =====
# =========================


def get_is_defined_by(iri: URIRef, graph: Graph, db: Dataset) -> Ontology | None:
    is_defined_by = get_values(iri, graph, [RDFS.isDefinedBy])
    ontology = is_defined_by[0] if len(is_defined_by) > 0 else None
    if ontology is not None:
        name = get_name(ontology, graph, db)
        return Ontology(iri=ontology, name=name)
    return None
