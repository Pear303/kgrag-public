"""知识图谱可视化：pyvis 力导向图（文章 6.2 节）。

_build_pyvis_graph 共用生成器：
- 微观查询：展示多跳路径，不同路径不同颜色，节点圆点，边带关系标签。
- 宏观查询：展示命中社区内部节点与边，同一社区节点同色。
使用 forceAtlas2Based 物理布局（注：pyvis 0.3.2 的 force_atlas_2based
不接受 seed 参数，稳定性靠布局算法本身保证）。
"""
from pyvis.network import Network

PALETTE = [
    "#E57373", "#64B5F6", "#81C784", "#FFB74D",
    "#BA68C8", "#4DB6AC", "#FFF176", "#A1887F",
]


def _build_pyvis_graph(paths, same_color: bool = False, color: str = "#64B5F6"):
    """从路径列表构建 pyvis 网络。

    paths: 微观=[(node_names, rel_types), ...]；宏观=[(nodes, edges)] 可扩展。
    返回 Network 实例，调用方再 write_html。
    """
    net = Network(
        notebook=False, directed=True,
        height="750px", width="100%",
        bgcolor="#ffffff", font_color="black",
    )
    seen_nodes, seen_edges = set(), set()
    for idx, item in enumerate(paths):
        if same_color:
            c = color
        else:
            c = PALETTE[idx % len(PALETTE)]
        names, rels = item[0], item[1]
        for i, rel in enumerate(rels):
            s, o = names[i], names[i + 1]
            if s not in seen_nodes:
                net.add_node(s, label=s, shape="dot", size=18, color=c)
                seen_nodes.add(s)
            if o not in seen_nodes:
                net.add_node(o, label=o, shape="dot", size=18, color=c)
                seen_nodes.add(o)
            ek = (s, o, rel)
            if ek not in seen_edges:
                net.add_edge(s, o, label=rel, title=rel, color=c, arrows="to")
                seen_edges.add(ek)
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=120)
    net.show_buttons(filter_=["physics"])
    return net


def render_micro(paths, out_html: str):
    """渲染微观多跳路径并写出 HTML。"""
    net = _build_pyvis_graph(paths, same_color=False)
    net.write_html(out_html)
    return out_html


def render_community(community_nodes_edges, out_html: str):
    """渲染某个社区内部子图（同色）。"""
    # community_nodes_edges: [(nodes, edges)]，edges 为 (s, rel, o)
    paths = []
    for nodes, edges in community_nodes_edges:
        # 转成 (node_names, rel_types) 便于复用生成器
        names = list(nodes)
        rels = [e[1] for e in edges]
        # 用链式展开：nodes[0]-rel0->nodes[1]-... 近似
        names_chain = []
        rels_chain = []
        for k, (s, r, o) in enumerate(edges):
            if k == 0:
                names_chain.append(s)
            names_chain.append(o)
            rels_chain.append(r)
        paths.append((names_chain, rels_chain))
    net = _build_pyvis_graph(paths, same_color=True, color="#4DB6AC")
    net.write_html(out_html)
    return out_html
