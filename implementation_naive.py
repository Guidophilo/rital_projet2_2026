def xquad(docs, relevance_q, relevance_qi, subqueries, 
          importance=None, k=10, omega=0.5):
    """
    docs : liste des documents
    relevance_q : dict {doc: score de pertinence à la requête}
    relevance_qi : dict {qi: {doc: score}}
    subqueries : liste des sous-requêtes
    importance : dict {qi: poids} (optionnel)
    k : nombre de documents à retourner
    omega : importance de la diversification
    """

    # Si pas d'importance → uniforme
    if importance is None:
        importance = {qi: 1 / len(subqueries) for qi in subqueries}

    selected = []
    remaining = set(docs)

    # masse d'information déjà couverte
    m = {qi: 1e-6 for qi in subqueries}  # éviter division par 0

    while len(selected) < k and remaining:
        best_doc = None
        best_score = -float("inf")

        for d in remaining:
            # score de pertinence classique
            score_rel = relevance_q.get(d, 0)

            # score de diversité
            score_div = 0
            for qi in subqueries:
                r_d_qi = relevance_qi.get(qi, {}).get(d, 0)
                score_div += importance[qi] * r_d_qi / m[qi]

            # score final
            score = score_rel * (score_div ** omega)

            if score > best_score:
                best_score = score
                best_doc = d

        # ajouter le meilleur document
        selected.append(best_doc)
        remaining.remove(best_doc)

        # mise à jour de la "masse"
        for qi in subqueries:
            m[qi] += relevance_qi.get(qi, {}).get(best_doc, 0)

    return selected

docs = ["d1", "d2", "d3", "d4"]

# pertinence à la requête principale
relevance_q = {
    "d1": 0.9,
    "d2": 0.8,
    "d3": 0.7,
    "d4": 0.6
}

# pertinence aux sous-requêtes
relevance_qi = {
    "aspect1": {"d1": 0.9, "d2": 0.1, "d3": 0.2, "d4": 0.0},
    "aspect2": {"d1": 0.1, "d2": 0.8, "d3": 0.3, "d4": 0.2},
    "aspect3": {"d1": 0.0, "d2": 0.2, "d3": 0.9, "d4": 0.5},
}

subqueries = ["aspect1", "aspect2", "aspect3"]

result = xquad(docs, relevance_q, relevance_qi, subqueries, k=4)

print(result)
