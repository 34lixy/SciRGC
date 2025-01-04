from rankers import *
from utils import *
import os
import pickle
from glob import glob
import numpy as np
import argparse
import json
from networkx.readwrite import json_graph
from tqdm import tqdm
import time


def LOG(info, end="\n"):
    with open(args.log_folder + "/" + args.log_file_name, "a") as f:
        f.write(info + end)


def read_citation_network(file_path):
    with open(file_path, 'r') as infile:
        data = json.load(infile)
        citation_network = json_graph.node_link_graph(data, directed=True)
    return citation_network


def compute_similarity(G, paper1, paper2):
    common_citations = len(set(G.neighbors(paper1)).intersection(G.neighbors(paper2)))
    try:
        similarity = common_citations / (
                len(list(G.neighbors(paper1))) + len(list(G.neighbors(paper2))) - common_citations)
    except ZeroDivisionError:
        similarity = 0
    return similarity


def order_similarity_paper_score(G, paper, threshold=0.5, top_k=None):
    neighbors = list(G.nodes())
    neighbors.remove(paper)
    if top_k:
        neighbors = sorted(neighbors, key=lambda x: compute_similarity(G, paper, x), reverse=True)[:top_k]
    else:
        neighbors = [n for n in neighbors if compute_similarity(G, paper, n) > threshold]
    return [(n, compute_similarity(G, paper, n)) for n in neighbors]


def order_neighbors_score(G, paper, threshold=0.5, top_k=None):
    neighbors = list(G.neighbors(paper))
    if top_k:
        neighbors = sorted(neighbors, key=lambda x: compute_similarity(G, paper, x), reverse=True)[:top_k]
    else:
        neighbors = [n for n in neighbors if compute_similarity(G, paper, n) > threshold]
    return [(n, compute_similarity(G, paper, n)) for n in neighbors]


# method1
def cited_similarity(paper):
    neighbors = order_neighbors_score(citation_network, paper, threshold=0, top_k=None)

    itf_candidates = {}
    for neighbor, sim1 in neighbors:
        neighbor_similarity_paper = order_similarity_paper_score(citation_network, neighbor, threshold=0, top_k=None)

        for n, sim2 in neighbor_similarity_paper:
            if n in itf_candidates:
                itf_candidates[n] += sim1 * sim2
            else:
                itf_candidates[n] = sim1 * sim2
    if paper in itf_candidates:
        del itf_candidates[paper]
    itf_candidates = sorted(itf_candidates.items(), key=lambda x: x[1], reverse=True)

    return itf_candidates


# method2
def similarity_cited(paper):
    similarity_paper = order_similarity_paper_score(citation_network, paper, threshold=0, top_k=None)

    utf_candidates = {}
    for paper_x, sim1 in similarity_paper:
        similarity_paper_neighbors = order_neighbors_score(citation_network, paper_x, threshold=0, top_k=None)

        for n, sim2 in similarity_paper_neighbors:
            if n in utf_candidates:
                utf_candidates[n] += sim1 * sim2
            else:
                utf_candidates[n] = sim1 * sim2
    if paper in utf_candidates:
        del utf_candidates[paper]
    utf_candidates = sorted(utf_candidates.items(), key=lambda x: x[1], reverse=True)

    return utf_candidates


def fused_candidates(candidates_A, utf_candidates_B, weight1=1, weight2=1):
    fused_results = {}

    for paper, score in candidates_A:
        if paper in fused_results:
            fused_results[paper] += score * weight1
        else:
            fused_results[paper] = score * weight1
    for paper, score in utf_candidates_B:
        if paper in fused_results:
            fused_results[paper] += score * weight2
        else:
            fused_results[paper] = score * weight2

    fused_results = sorted(fused_results.items(), key=lambda x: x[1], reverse=True)
    fused_results = [paper_id for paper_id, score in fused_results]
    return fused_results


def fused_candidates_triple(candidates_A, utf_candidates_B, utf_candidates_C, weight1=1, weight2=1, weight3=1):
    fused_results = {}

    for paper, score in candidates_A:
        if paper in fused_results:
            fused_results[paper] += score * weight1
        else:
            fused_results[paper] = score * weight1

    for paper, score in utf_candidates_B:
        if paper in fused_results:
            fused_results[paper] += score * weight2
        else:
            fused_results[paper] = score * weight2

    for paper, score in utf_candidates_C:
        if paper in fused_results:
            fused_results[paper] += score * weight3
        else:
            fused_results[paper] = score * weight3

    fused_results = sorted(fused_results.items(), key=lambda x: x[1], reverse=True)
    fused_results = [paper_id for paper_id, score in fused_results]
    return fused_results


def mean_reciprocal_rank(ranks):
    reciprocal_ranks = [1.0 / rank for rank in ranks if rank is not None]  # 排除没有正确结果的查询
    if len(reciprocal_ranks) == 0:
        return 0
    else:
        return sum(reciprocal_ranks) / len(reciprocal_ranks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-log_folder")
    parser.add_argument("-log_file_name")
    parser.add_argument("-start", type=int, default=0)
    parser.add_argument("-size", type=int, default=0)
    parser.add_argument("-K_list", type=int, nargs="+", default=[10, 20, 50, 100, 200, 500, 1000, 2000])
    parser.add_argument("-encoder_gpu_list", type=int, nargs="+", default=None)
    parser.add_argument("-ranker_gpu_list", type=int, nargs="+", default=None)
    parser.add_argument("-input_corpus_path")
    parser.add_argument("-unigram_words_path")
    parser.add_argument("-prefetch_model_folder")
    parser.add_argument("-prefetch_model_path", default=None)
    parser.add_argument("-prefetch_embedding_path")
    parser.add_argument("-paper_database_path")
    parser.add_argument("-context_database_path")
    parser.add_argument("-embed_dim", type=int, default=200)
    parser.add_argument("-num_heads", type=int, default=8)
    parser.add_argument("-hidden_dim", type=int, default=1024)
    parser.add_argument("-max_seq_len", type=int, default=512)
    parser.add_argument("-max_doc_len", type=int, default=3)
    parser.add_argument("-n_para_types", type=int, default=100)
    parser.add_argument("-num_enc_layers", type=int, default=1)
    parser.add_argument("-citation_title_label", type=int, default=0)
    parser.add_argument("-citation_abstract_label", type=int, default=1)
    parser.add_argument("-citation_context_label", type=int, default=3)
    parser.add_argument("-fused_label", type=int, default=6)
    args = parser.parse_args()

    if not os.path.exists(args.log_folder):
        os.makedirs(args.log_folder)

    if args.prefetch_model_path is not None:
        ckpt_name = args.prefetch_model_path
    else:
        try:
            ckpt_list = glob(args.prefetch_model_folder + "/*.pt")
            if len(ckpt_list) > 0:
                ckpt_list.sort(key=os.path.getmtime)
                ckpt_name = ckpt_list[-1]
            else:
                ckpt_name = None
        except:
            ckpt_name = None

    assert ckpt_name is not None
    encoder = PrefetchEncoder(ckpt_name, args.unigram_words_path,
                              args.embed_dim, args.encoder_gpu_list,
                              args.num_heads, args.hidden_dim,
                              args.max_seq_len, args.max_doc_len,
                              args.n_para_types, args.num_enc_layers
                              )

    ranker = Ranker(args.prefetch_embedding_path, args.embed_dim, gpu_list=args.ranker_gpu_list)
    ranker.encoder = encoder

    paper_database = json.load(open(args.paper_database_path))
    corpus = json.load(open(args.input_corpus_path))
    if args.size == 0:
        corpus = corpus[args.start:]
    else:
        corpus = corpus[args.start:args.start + args.size]

    context_database = json.load(open(args.context_database_path))

    ###################################
    K_list = args.K_list
    max_K = np.max(K_list)
    fused_label = args.fused_label
    print(fused_label)
    positive_ids_list = []
    candidates_list = []
    query_time_list = []
    ranks_list = []
    citation_network = read_citation_network('../../data/peerread/citation_network.json')

    # for i in range(7):
    #     fused_label = i
    for count, example in enumerate(tqdm(corpus)):

        context_id = example["context_id"]
        citing_id = context_database[context_id]["citing_id"]
        citing_paper_info = paper_database.get(citing_id, {})
        
        if citing_id not in citation_network.nodes():
            citation_network.add_node(citing_id)
        if context_database[context_id]["refid"] not in citation_network.nodes():
            citation_network.add_node(context_database[context_id]["refid"])

        if citation_network.has_edge(citing_id, context_database[context_id]["refid"]):
            citation_network.remove_edge(citing_id, context_database[context_id]["refid"])
        query_text = [
            [
                [citing_paper_info.get("title", ""), args.citation_title_label],
                [citing_paper_info.get("abstract", ""), args.citation_abstract_label],
                [context_database[context_id]["masked_text"], args.citation_context_label]
            ]
        ]

        tic = time.time()
        if fused_label == 0:

            candidates = ranker.get_top_n(max_K + 1, query_text)

        else:
            candidates_score = ranker.get_top_n_score(max_K + 1, query_text)
            itf_candidates = cited_similarity(citing_id)
            utf_candidates = similarity_cited(citing_id)
            if fused_label == 1:
                candidates = fused_candidates(candidates_score, utf_candidates)
            if fused_label == 2:
                candidates = fused_candidates(candidates_score, itf_candidates)
            if fused_label == 3:
                candidates = fused_candidates_triple(candidates_score, itf_candidates, utf_candidates)
            if fused_label == 4:
                fused_results = {}
                for paper, score in utf_candidates:
                    if paper in fused_results:
                        fused_results[paper] += score
                    else:
                        fused_results[paper] = score
                fused_results = sorted(fused_results.items(), key=lambda x: x[1], reverse=True)
                fused_results = [paper_id for paper_id, score in fused_results]
                candidates = fused_results
            if fused_label == 5:
                fused_results = {}
                for paper, score in itf_candidates:
                    if paper in fused_results:
                        fused_results[paper] += score
                    else:
                        fused_results[paper] = score
                fused_results = sorted(fused_results.items(), key=lambda x: x[1], reverse=True)
                fused_results = [paper_id for paper_id, score in fused_results]
                candidates = fused_results
            if fused_label == 6:
                candidates = fused_candidates(utf_candidates, itf_candidates)
        citation_network.add_edge(citing_id, context_database[context_id]["refid"])
        tac = time.time()

        query_time_list.append(tac - tic)

        if citing_id in candidates and citing_id not in set(example["positive_ids"]):
            candidates.remove(citing_id)
        candidates = candidates[:max_K]

        positive_ids_list.append(example["positive_ids"])
        candidates_list.append(candidates)

        rank = next((i + 1 for i, candidate in enumerate(candidates) if candidate in example["positive_ids"]), None)
        ranks_list.append(rank)

    precision_at_K = {}
    recall_at_K = {}
    F_at_K = {}
    for K in tqdm(K_list):
        recall_list = []
        precision_list = []
        for positive_ids, candidates in zip(positive_ids_list, candidates_list):
            hit_num = len(set(positive_ids) & set(candidates[:K]))
            recall_list.append(hit_num / len(set(positive_ids)))
            precision_list.append(hit_num / K)
        recall_at_K[K] = np.mean(recall_list)
        precision_at_K[K] = np.mean(precision_list)
        F_at_K[K] = 2 / (1 / (recall_at_K[K] + 1e-12) + 1 / (precision_at_K[K] + 1e-12))
    ckpt_name = str(ckpt_name)
    mrr_score = mean_reciprocal_rank(ranks_list)
    print({"fused_label": fused_label, "ckpt_name": ckpt_name, "recall": recall_at_K, "MRR": mrr_score}, flush=True)
    LOG(json.dumps({"fused_label": fused_label, "ckpt_name": ckpt_name, "recall": recall_at_K, "MRR": mrr_score}))
    print("Finished!", flush=True)
    LOG("Finished!")
