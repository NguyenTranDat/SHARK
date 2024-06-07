from sentence_transformers import SentenceTransformer, util
import torch
import csv


embedder = SentenceTransformer("all-MiniLM-L6-v2")


atomic_data = []
corpus_o_react = []
corpus_x_react = []
with open("data/atomic.tsv", newline="") as file:
    reader = csv.reader(file, delimiter="\t")
    for row in reader:
        if row[1] == "oReact":
            atomic_data.append(row[2])
            corpus_o_react.append(row[0])
        elif row[1] == "xReact":
            atomic_data.append(row[2])
            corpus_x_react.append(row[0])


corpus_o_react_embeddings = embedder.encode(corpus_o_react, convert_to_tensor=True)
corpus_x_react_embeddings = embedder.encode(corpus_x_react, convert_to_tensor=True)


def read_file(data_path: str) -> list:
    data: list = []
    with open(data_path, newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        next(reader)
        for row in reader:
            data.append(
                {
                    "index": f"dia{row[0]}_utt{row[1]}",
                    "text": row[2],
                    "speakername": row[7],
                }
            )

    return data


def cal_sematic(data: list, data_path: str) -> None:
    result = []
    for d in data:
        query_embedding = embedder.encode(d["text"], convert_to_tensor=True)
        cos_scores_o = util.cos_sim(query_embedding, corpus_o_react_embeddings)[0]
        top_results_o = torch.topk(cos_scores_o, k=3)

        cos_scores_x = util.cos_sim(query_embedding, corpus_x_react_embeddings)[0]
        top_results_x = torch.topk(cos_scores_x, k=3)

        result.append(
            {
                "index": d["index"],
                "oReact_retrieval": f"Others feel {atomic_data[top_results_o[1][0]]}, {atomic_data[top_results_o[1][1]]}, {atomic_data[top_results_o[1][2]]}",
                "xReact_retrieval": f'{d["speakername"]} feels {atomic_data[top_results_o[1][0]]}, {atomic_data[top_results_o[1][1]]}, {atomic_data[top_results_o[1][2]]}',
            }
        )

    with open(data_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, delimiter="\t", fieldnames=["index", "oReact_retrieval", "xReact_retrieval"])
        writer.writeheader()
        for row in result:
            writer.writerow(row)


import threading


def process_file(input_path: str, output_path: str):
    data = read_file(input_path)
    cal_sematic(data, output_path)


def process():
    files = [
        ("data/test.tsv", "data/output_test.tsv"),
        ("data/train.tsv", "data/output_train.tsv"),
        ("data/dev.tsv", "data/output_dev.tsv"),
    ]

    threads = []
    for input_path, output_path in files:
        thread = threading.Thread(target=process_file, args=(input_path, output_path))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


process()
