import networkx as nx
import matplotlib.pyplot as plt

# 准备数据
triples = [
    {"subject": "2445", "predicate": "film.actor.film", "object": "2446"},
    {"subject": "2447", "predicate": "film.person_or_entity_appearing_in_film.film", "object": "2448"}
]

# 创建有向图对象
graph = nx.DiGraph()

# 添加节点和边
for triple in triples:
    subject = triple["subject"]
    predicate = triple["predicate"]
    obj = triple["object"]

    graph.add_node(subject)
    graph.add_node(obj)
    graph.add_edge(subject, obj, label=predicate)

# 可视化图形
pos = nx.spring_layout(graph)  # 指定节点布局算法
labels = nx.get_edge_attributes(graph, 'label')  # 获取边标签

nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=5000)
nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

plt.show()
