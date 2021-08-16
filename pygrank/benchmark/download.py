import gzip, tarfile
import wget
import shutil
import sys
import os

datasets = {
    "dblp": {"url": "https://snap.stanford.edu/data/com-DBLP.html",
             "pairs": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz",
             "groups": "https://snap.stanford.edu/data/bigdata/communities/com-dblp.all.cmty.txt.gz"},
    "eucore": {"url": "https://snap.stanford.edu/data/email-Eu-core.html",
               "pairs": "https://snap.stanford.edu/data/email-Eu-core.txt.gz",
               "labels": "https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz"},
    "citeseer": {"url": "https://linqs.soe.ucsc.edu/data",
                 "all": "https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz",
                 "pairs": "citeseer/citeseer.cites",
                 "features": "citeseer/citeseer.content",
                 "remove": "citeseer/"}
}


def downloadable_datasets():
    return list(datasets.keys())


def downloadable_small_datasets():
    return ["citeseer", "eucore"]


def download_dataset(dataset, path: str = "data"):   # pragma: no cover
    dataset = dataset.lower()
    if dataset not in datasets:
        return
    source = datasets[dataset] if isinstance(dataset, str) else dataset
    credentials = "REQUIRED CITATION: Please visit the url "+source["url"]+" for instructions on how to cite the dataset "+dataset+" in your research"
    print(credentials, file=sys.stderr)
    if not os.path.isdir(path):
        os.mkdir(path)
    download_path = path+"/"+dataset
    if not os.path.isdir(download_path):
        os.mkdir(download_path)
        if "all" in source:
            all_path = download_path+"/all."+source["all"].split(".")[-1]
            wget.download(source["all"], all_path)
            tarfile.open(all_path, 'r').extractall(download_path+"/")
            os.remove(all_path)

        if "pairs" in source:
            if source["pairs"].startswith("http"):
                pairs_path = download_path+"/pairs."+source["pairs"].split(".")[-1]
                wget.download(source["pairs"], pairs_path)
                with gzip.open(pairs_path, 'rb') as f_in:
                    with open(download_path+"/pairs.txt", 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(pairs_path)
            else:
                shutil.move(download_path+"/"+source["pairs"], download_path+"/pairs.txt")

        if "features" in source:
            features_path = download_path+"/"+source["features"]
            groups = dict()
            features = dict()
            with open(features_path) as features_file:
                for line in features_file:
                    line = line[:-1].split()
                    node_id = line[0]
                    group = line[-1]
                    if group not in groups:
                        groups[group] = list()
                    groups[group].append(node_id)
                    features[node_id] = [val.strip() for val in line[1:-1]]
            groups = {group: nodes for group, nodes in groups.items() if len(nodes) > 1}
            with open(download_path+'/groups.txt', 'w', encoding='utf-8') as file:
                for g in groups.values():
                    for uid in g:
                        file.write(str(uid) + '\t')
                    file.write('\n')
            with open(download_path+'/features.txt', 'w', encoding='utf-8') as file:
                for p in features:
                    file.write(str(p) + '\t' + '\t'.join(features[p]) + '\n')
        elif "groups" in source:
            groups_path = download_path+"/groups."+source["groups"].split(".")[-1]
            wget.download(source["groups"], groups_path)
            with gzip.open(groups_path, 'rb') as f_in:
                with open(download_path+"/groups.txt", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(groups_path)
        elif "labels" in source:
            labels_path = download_path+"/labels."+source["labels"].split(".")[-1]
            wget.download(source["labels"], labels_path)
            with gzip.open(labels_path, 'rb') as f_in:
                with open(download_path+"/labels.txt", 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(labels_path)
            groups = dict()
            with open(download_path+"/labels.txt", 'r', encoding='utf-8') as file:
                for line in file:
                    if line[0] != '#':
                        splt = line[:-1].split()
                        if len(splt) >= 2:
                            if splt[1] not in groups:
                                groups[splt[1]] = list()
                            groups[splt[1]].append(splt[0])
            with open(download_path+"/groups.txt", 'w', encoding='utf-8') as file:
                for group in groups.values():
                    file.write((" ".join(group))+"\n")

        if "remove" in source:
            shutil.rmtree(download_path+"/"+source["remove"])
    return credentials