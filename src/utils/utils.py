from models.structure_model.tree import Tree
import logging

# hierar.txt 파일에서 sibling, parent를 얻는 함수들임. 

def get_hierarchy_relations(hierar_taxonomy, label_map, root=None, fortree=False):
    label_tree = dict()
    label_tree[0] = root
    hierar_relations = {}
    with open(hierar_taxonomy, "r") as f:
        for line in f:
            line_split = line.rstrip().split('\t')
            parent_label, children_label = line_split[0], line_split[1:]
            if parent_label not in label_map:
                if fortree and parent_label == 'Root':
                    parent_label_id = -1
                else:
                    continue
            else:
                parent_label_id = label_map[parent_label]
            children_label_ids = [label_map[child_label] \
                                  for child_label in children_label if child_label in label_map]
            hierar_relations[parent_label_id] = children_label_ids
            if fortree:
                assert (parent_label_id + 1) in label_tree
                parent_tree = label_tree[parent_label_id + 1]

                for child in children_label_ids:
                    assert (child + 1) not in label_tree
                    child_tree = Tree(child)
                    parent_tree.add_child(child_tree)
                    label_tree[child + 1] = child_tree
    if fortree:
        return hierar_relations, label_tree
    else:
        return hierar_relations

def get_parent_sibling(hierar_taxonomy, label_map):
    hierar_relations = {}
    hierar_relations_sibling = {}
    with open(hierar_taxonomy, "r") as f:
        for line in f:
            line_split = line.rstrip().split('\t')
            parent_label, children_label = line_split[0], line_split[1:]
            if parent_label not in label_map:
                if parent_label != 'Root':
                    continue
            parent_label_id = parent_label
            children_label_ids = [child_label \
                                  for child_label in children_label if child_label in label_map]
            for child in children_label_ids:
                hierar_relations[child] = parent_label_id
                children_label_ids.remove(child)
                hierar_relations_sibling[child] = children_label_ids
                children_label_ids.append(child)
    return hierar_relations, hierar_relations_sibling

def get_parent(get_child, config):
    hierarchy_prob_child_parent_id = {}
    for (k, v) in get_child.items():
        for child in v:
            if child not in hierarchy_prob_child_parent_id:
                hierarchy_prob_child_parent_id[child] = [k]
            else:
                hierarchy_prob_child_parent_id[child].append(k)
    return hierarchy_prob_child_parent_id

def get_sibling(hierar_taxonomy, get_child, config, label_v2i):
    hierarchy_prob_sibling_id = {}
    for (k, v) in get_child.items():
        if len(v) == 1:
            continue
        for child in v:
            hierarchy_prob_sibling_id[child] = []
            for c in v:
                if c != child:
                    hierarchy_prob_sibling_id[child].append(c)
    first_layer = []
    with open(hierar_taxonomy, "r") as f:
        for line in f:
            line_split = line.rstrip().split('\t')
            parent_label, children_label = line_split[0], line_split[1:]
            if parent_label != "Root":
                continue
            children_label = [label_v2i[c] for c in children_label]
            first_layer = children_label
            if len(first_layer) == 1:
                break
            for child in first_layer:
                if child not in hierarchy_prob_sibling_id:
                    hierarchy_prob_sibling_id[child] = []
                    for c in first_layer:
                        if c != child:
                            hierarchy_prob_sibling_id[child].append(c)
            break
    return hierarchy_prob_sibling_id, first_layer

def get_root_logger(logger_name='basicsr',
                    log_level=logging.INFO,
                    log_file=None):
    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(format=format_str, level=log_level)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger