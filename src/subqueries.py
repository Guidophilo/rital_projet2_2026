# génération des sous requêtes
# ground-truth si disponible
# clustering/Bo 1
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

def generate_dummy_subqueries(query):
    """
    Version simple pour commencer.

    On simule des aspects.
    Plus tard → clustering ou TREC subtopics.
    """
    return [
        query + " aspect1",
        query + " aspect2",
        query + " aspect3",
    ]



def extract_document_texts(
    docs_df: pd.DataFrame,
    text_col: Optional[str] = None,
) -> List[str]:
    """
    Extrait la liste des textes d'un DataFrame de documents.

    Args:
        docs_df: DataFrame contenant les documents.
        text_col: nom explicite de la colonne texte si on la connaît déjà.

    Returns:
        Liste de textes, dans le même ordre que le DataFrame.

    Notes:
        - Si text_col est fourni, on l'utilise directement.
        - Sinon, on essaie quelques noms de colonnes fréquents.
        - On remplace les valeurs manquantes par une chaîne vide.
    """
    if text_col is not None:
        if text_col not in docs_df.columns:
            raise ValueError(f"La colonne texte '{text_col}' n'existe pas dans docs_df.")
        return docs_df[text_col].fillna("").astype(str).tolist()

    candidate_columns = ["text", "body", "contents", "abstract", "title"]
    for col in candidate_columns:
        if col in docs_df.columns:
            return docs_df[col].fillna("").astype(str).tolist()

    raise ValueError(
        "Impossible de trouver une colonne texte dans docs_df. "
        f"Colonnes disponibles: {list(docs_df.columns)}"
    )


def clean_generated_terms(terms: List[str]) -> List[str]:
    """
    Nettoyage léger des termes extraits.

    Objectif:
        - enlever doublons
        - enlever termes trop courts
        - conserver l'ordre

    Args:
        terms: liste brute de mots-clés

    Returns:
        Liste nettoyée.
    """
    seen = set()
    cleaned = []

    for term in terms:
        term = str(term).strip().lower()

        # On ignore les termes vides ou trop courts
        if len(term) < 2:
            continue

        if term not in seen:
            cleaned.append(term)
            seen.add(term)

    return cleaned


def extract_cluster_keywords_from_centroid(
    centroid_vector: np.ndarray,
    feature_names: np.ndarray,
    top_k_terms: int = 5,
) -> List[str]:
    """
    Extrait les mots-clés les plus représentatifs d'un centroïde de cluster.

    Args:
        centroid_vector: vecteur du centroïde
        feature_names: vocabulaire TF-IDF
        top_k_terms: nombre de termes à garder

    Returns:
        Liste de termes triés par importance décroissante.
    """
    # On trie les dimensions du centroïde par poids décroissant
    top_indices = np.argsort(centroid_vector)[::-1][:top_k_terms]
    raw_terms = [feature_names[i] for i in top_indices]
    return clean_generated_terms(raw_terms)


def build_subquery_from_terms(terms: List[str]) -> str:
    """
    Construit une sous-requête textuelle à partir d'une liste de mots-clés.

    Exemple:
        ["dielectric", "liquid", "microwave"] ->
        "dielectric liquid microwave"
    """
    if not terms:
        return ""
    return " ".join(terms)


def build_subqueries_from_kmeans(
    docs_df: pd.DataFrame,
    text_col: Optional[str] = None,
    n_clusters: int = 5,
    top_k_terms: int = 5,
    max_features: int = 5000,
    random_state: int = 42,
) -> List[str]:
    """
    Génère de vraies sub-queries à partir des documents via TF-IDF + KMeans.

    Pipeline:
        - extraction des textes
        - vectorisation TF-IDF
        - clustering KMeans
        - extraction de mots-clés par cluster
        - construction d'une sous-requête par cluster

    Args:
        docs_df: DataFrame contenant les documents et leur texte
        text_col: colonne contenant le texte, si connue
        n_clusters: nombre de clusters / aspects
        top_k_terms: nombre de mots-clés par cluster
        max_features: taille max du vocabulaire TF-IDF
        random_state: graine pour la reproductibilité

    Returns:
        Liste de sous-requêtes générées.
    """
    texts = extract_document_texts(docs_df, text_col=text_col)

    # On retire les documents vides pour éviter un TF-IDF dégénéré
    non_empty_texts = [t for t in texts if t and t.strip()]
    if len(non_empty_texts) == 0:
        return []

    # Si on a très peu de documents, on réduit automatiquement le nombre de clusters
    effective_k = min(n_clusters, len(non_empty_texts))
    if effective_k <= 0:
        return []

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
    )
    X = vectorizer.fit_transform(non_empty_texts)

    # Si le vocabulaire est vide après filtrage, on ne peut pas construire de sub-queries
    if X.shape[1] == 0:
        return []

    feature_names = vectorizer.get_feature_names_out()

    kmeans = KMeans(
        n_clusters=effective_k,
        random_state=random_state,
        n_init=10,
    )
    kmeans.fit(X)

    subqueries = []
    for cluster_id in range(effective_k):
        centroid = kmeans.cluster_centers_[cluster_id]
        keywords = extract_cluster_keywords_from_centroid(
            centroid_vector=centroid,
            feature_names=feature_names,
            top_k_terms=top_k_terms,
        )
        subquery = build_subquery_from_terms(keywords)

        if subquery:
            subqueries.append(subquery)

    # On retire les doublons éventuels tout en gardant l'ordre
    deduped = []
    seen = set()
    for sq in subqueries:
        if sq not in seen:
            deduped.append(sq)
            seen.add(sq)

    return deduped