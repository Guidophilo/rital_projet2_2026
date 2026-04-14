# l'algorithme principal de reclassement xQuAD
def xquad_rerank(
    docs_ranked,
    subqueries,
    r_query,
    r_sub,
    importance,
    tau=100,
    omega=0.5,
    init_mass=1.0, #non précisé dans l'article, ici 1 pour éviter division par 0
):
    """
    Implémentation de xQuAD.

    docs_ranked: liste des documents du baseline
    subqueries: liste des aspects
    r_query: score doc → query
    r_sub: score doc → subquery
    importance: poids des aspects
    """

    # documents restants à explorer
    remaining = list(docs_ranked)

    # ranking final
    selected = []

    # mémoire de couverture des aspects
    mass = {sq: init_mass for sq in subqueries}

    while remaining and len(selected) < tau:

        best_doc = None
        best_score = -1

        for d in remaining:
            diversity = 0.0

            # calcul de la diversité pour ce document
            for sq in subqueries:
                if sq in r_sub.get(d, {}):
                    diversity += (
                        importance[sq] #ix(qi,q)
                        * r_sub[d][sq] # r(d,qi)
                        / mass[sq] #1/m(qi)
                    )

            base = r_query.get(d, 0) # r(d,q)

            # combinaison relevance + diversité
            if diversity > 0: # la somme
                score = base * (diversity ** omega)# omega <=> w
            else: # cas expection, non explicité dans le papier
                score = 0

            if score > best_score: # <=> d^* = argmax r(d,q,Q(q)) sur d
                best_score = score
                best_doc = d

        if best_doc is None:
            break

        # mise à jour de la couverture des aspects
        # m(qi)=m(qi)+r(d*,qi)
        for sq in subqueries:
            #optimisation pour éviter de parcourir des score nuls
            if sq in r_sub.get(best_doc, {}):
                mass[sq] += r_sub[best_doc][sq]
        # S(q)=S(q)∪{d*}
        selected.append(best_doc)
        # R(q)=R(q)\{d*}
        remaining.remove(best_doc)

    return selected