# fonctions utilitaires pour convertir les dataframes PyTerrier
# en dictionnaires/listes à xQuAD 
import pandas as pd
from src.retrieval import *
from src.importance import *
from src.xquad import *
def results_to_doclist(df):
    """
    Convertit un DataFrame PyTerrier en liste de documents ordonnés.

    Pourquoi ?
    xQuAD travaille sur une liste ordonnée, pas un DataFrame.
    """
    df = df.sort_values("rank")
    return df["docno"].tolist()


def results_to_score_dict(df):
    """
    Convertit en dictionnaire:
        doc → score

    Très utile pour accéder rapidement à r(d, q).
    """
    return dict(zip(df["docno"], df["score"]))


def subquery_results_to_dict(subquery_results):
    """
    Transforme:
        {subquery: DataFrame}
    en:
        {doc: {subquery: score}}

    C'est la structure clé pour xQuAD.
    """
    result = {}

    for sq, df in subquery_results.items():
        for _, row in df.iterrows():
            d = row["docno"]
            s = row["score"]

            if d not in result:
                result[d] = {}

            result[d][sq] = s

    return result


def subquery_counts(subquery_results):
    """
    Compte combien de documents chaque subquery retourne.

    Sert pour:
        i_N
        i_R
        i_C
    """
    return {sq: len(df) for sq, df in subquery_results.items()}

def ranking_to_run_df(qid, query, ranking):
    """
    Convertit une liste ordonnée de documents en DataFrame de run.
    C'est pratique pour l'évaluation plus tard.
    """
    rows = []
    for rank, docno in enumerate(ranking, start=1):
        rows.append({
            "qid": qid,
            "query": query,
            "docno": docno,
            "rank": rank,
            "score": len(ranking) - rank + 1,  # score artificiel de sortie
        })
    return pd.DataFrame(rows)

def show_top_docs(run_df, docs_df, top_k=10, text_col="text"):
    merged = run_df.head(top_k).merge(docs_df, on="docno", how="left")
    
    for i, row in merged.iterrows():
        print(f"Rank {row['rank']} | docno={row['docno']} | score={row['score']}")
        if "title" in merged.columns:
            print("Title:", row.get("title", ""))
        text_value = row.get(text_col, "")
        print("Text preview:", str(text_value)[:300])
        print("-" * 80)




def compute_all_importances(subqueries, docs_ranked, r_query, r_sub, n_qi, tau=100):
    """
    Calcule les 4 variantes d'importance.
    """
    return {
        "uniform": importance_uniform(subqueries),
        "n": importance_n(n_qi),
        "redde": importance_redde(docs_ranked, subqueries, r_query, r_sub, n_qi),
        "crcs": importance_crcs(docs_ranked, subqueries, r_sub, n_qi, tau=tau),
    }


def run_xquad_variants(docs_ranked, subqueries, r_query, r_sub, importances, tau=100, omega=0.5):
    """
    Lance xQuAD avec les 4 variantes.
    """
    results = {}

    for name, imp in importances.items():
        results[name] = xquad_rerank(
            docs_ranked=docs_ranked,
            subqueries=subqueries,
            r_query=r_query,
            r_sub=r_sub,
            importance=imp,
            tau=tau,
            omega=omega,
        )

    return results


def print_top_comparison(label, baseline_ranking, xquad_rankings, top_k=10):
    """
    Affiche baseline + différentes variantes xQuAD.
    """
    print(f"\n===== {label} =====")
    print("Baseline :", baseline_ranking[:top_k])

    for name, ranking in xquad_rankings.items():
        print(f"xQuAD-{name} :", ranking[:top_k])


def check_importance_sums(importances):
    """
    Vérifie que les poids somment à 1.
    """
    for name, imp in importances.items():
        print(name, "-> somme =", sum(imp.values()))


def check_ranking_properties(rankings_dict):
    """
    Vérifie longueur et unicité des docs dans les rankings.
    """
    for name, ranking in rankings_dict.items():
        print(
            name,
            "| taille =", len(ranking),
            "| docs uniques =", len(set(ranking))
        )

def prepare_subquery_scores(index_ref, qid, subqueries, wmodel="BM25", num_results=1000):
    """
    Lance le retrieval pour chaque sous-requête et construit:
    - subquery_results : dict {subquery -> DataFrame}
    - r_sub : dict {doc -> {subquery -> score}}
    - n_qi : dict {subquery -> nombre de docs}
    """

    subquery_results = {}

    for sq in subqueries:
        sq_topics = pd.DataFrame([{"qid": qid, "query": sq}])

        sq_df = run_retrieval(
            index_ref,
            sq_topics,
            wmodel=wmodel,
            num_results=num_results
        )

        subquery_results[sq] = sq_df

    # transformation pour xQuAD
    r_sub = subquery_results_to_dict(subquery_results)

    # nombre de documents par subquery
    n_qi = subquery_counts(subquery_results)

    return subquery_results, r_sub, n_qi