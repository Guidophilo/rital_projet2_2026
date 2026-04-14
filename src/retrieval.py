# fonctions pour initialiser PyTerrier, lancer BM25 et DPH
# récupérer les top-k document
import pyterrier as pt
import pandas as pd

def init_pt():
    """
    Initialise PyTerrier (nécessaire avant toute utilisation).
    Ne lance l'init qu'une seule fois.
    """
    if not pt.started():
        pt.init()


def get_dataset(name="irds:cord19/trec-covid"):
    """
    Charge un dataset PyTerrier.
    Exemple:
        "irds:vaswani" (petit dataset pour debug)
        plus tard → TREC
    """
    return pt.get_dataset(name)


def get_index(dataset):
    """
    Récupère l'index Terrier associé au dataset.
    """
    return dataset.get_index()


def get_topics(dataset):
    """
    Récupère les requêtes (topics).
    On garde uniquement qid et query.
    """
    return dataset.get_topics()[["qid", "query"]]


def build_retriever(index_ref, wmodel="BM25", num_results=1000):
    """
    Construit un moteur de recherche Terrier.

    Args:
        wmodel: "BM25" ou "DPH"
        num_results: nombre max de documents retournés
    """
    return pt.BatchRetrieve(
        index_ref,
        wmodel=wmodel,
        num_results=num_results
    )


def run_retrieval(index_ref, topics, wmodel="BM25", num_results=1000):
    """
    Lance la recherche sur une liste de requêtes.
    Retour:
        DataFrame avec:
        - qid
        - docno
        - score
        - rank
        - query
    """
    retriever = build_retriever(index_ref, wmodel, num_results)
    return retriever.transform(topics)