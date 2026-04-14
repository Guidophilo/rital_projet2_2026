# implémentation des 4 fonctions d'importance pour ix
def normalize(weights):
    """
    Normalise les poids pour qu'ils somment à 1.

    Très important car on approxime une probabilité P(q_i | q).
    """
    total = sum(weights.values())

    if total == 0:
        # fallback: distribution uniforme
        n = len(weights)
        return {k: 1.0 / n for k in weights}

    return {k: v / total for k, v in weights.items()}


def importance_uniform(subqueries):
    """
    Tous les aspects sont considérés également importants.

    Cas baseline du papier.
    """
    k = len(subqueries)
    return {sq: 1.0 / k for sq in subqueries}


def importance_n(n_qi):
    """
    Importance basée sur le nombre de documents par subquery.

    Intuition:
        plus un aspect retourne de docs → plus il est important
    """
    return normalize(n_qi)


def importance_redde(docs, subqueries, r_query, r_sub, n_qi):
    """
    Version inspirée de ReDDE.

    Idée:
        un aspect est important si ses documents sont:
            - pertinents pour la requête
            - pertinents pour l'aspect
    """
    weights = {}

    for sq in subqueries:
        score = 0.0

        for d in docs:
            if sq in r_sub.get(d, {}):
                score += r_query[d] * r_sub[d][sq] * n_qi[sq]

        weights[sq] = score

    return normalize(weights)


def importance_crcs(docs_ranked, subqueries, r_sub, n_qi, tau=100):
    """
    Version inspirée de CRCS.

    Idée:
        un aspect est important si ses documents apparaissent haut dans le ranking.
    """
    weights = {}
    max_n = max(n_qi.values())

    # mapping doc → position dans ranking
    rank_map = {d: i+1 for i, d in enumerate(docs_ranked)}

    for sq in subqueries:
        score = 0.0
        count = 0

        # on regarde seulement les top tau documents
        for d in docs_ranked[:tau]:
            if sq in r_sub.get(d, {}):
                score += (tau - rank_map[d])  # plus haut = meilleur
                count += 1

        if count == 0:
            weights[sq] = 0
        else:
            weights[sq] = (n_qi[sq] / max_n) * (score / count)

    return normalize(weights)