#!/usr/bin/env python

import math
import numpy as np
import pathlib

# 注意：
# シルエットを構成する多角形は反時計回りで、
# 自己交差がないことを前提とする


class SilhouetteEvaluator:
    __slots__ = []

    def eval(self, sil):
        ...


def is_on_line(edge, vertex, epsilon2):
    ((ay, ax), (by, bx)) = edge
    (vy, vx) = vertex

    # Bounding Box判定
    if vy < min(ay, by):
        return False
    if vy > max(ay, by):
        return False
    if vx < min(ax, bx):
        return False
    if vx > max(ax, bx):
        return False

    # vertex bを基準に相対座標へ
    ay = ay - by
    vy = vy - by
    ax = ax - bx
    vx = vx - bx

    # 点と直線の距離の式の分子の二乗を計算
    top = ay * vx - ax * vy
    top2 = top * top
    bottom2 = ax * ax + ay * ay

    return top2 <= epsilon2 * bottom2


def is_same(vertex0, vertex1, epsilon):
    ay, ax = vertex0
    by, bx = vertex1
    if abs(by - ay) > epsilon:
        return False
    if abs(bx - ax) > epsilon:
        return False
    return True


def dist2(vertex0, vertex1):
    ay, ax = vertex0
    by, bx = vertex1
    dy = by - ay
    dx = bx - ax
    return (dy * dy + dx * dx)


def dist(vertex0, vertex1):
    return math.sqrt(dist2(vertex0, vertex1))


def edge_atan2(edge):
    (sy, sx), (ty, tx) = edge
    angle = math.atan2(ty - sy, tx - sx)
    return angle


def edges_atan2(edge0, edge1):
    angle_a = edge_atan2(edge0)
    angle_b = edge_atan2(edge1)
    change_angle = angle_b - angle_a
    return change_angle


def vector_angle(edge0, edge1):
    change_angle = edges_atan2(edge0, edge1)
    change_angle = change_angle + math.pi  # 内角(左側の角度)にする
    if change_angle < 0:
        change_angle = change_angle + 2 * math.pi
    elif change_angle > 2 * math.pi:
        change_angle = change_angle - 2 * math.pi
    return change_angle


def vector_degree(edge0, edge1):
    return 180 / math.pi * vector_angle(edge0, edge1)


def get_graph_repr(puzzle):
    """シルエットをグラフ表現へと変換."""

    # 全ての頂点を取得
    raw_vertex_list = [
        _v for _poly in puzzle.polygons for _v in _poly.xys]
    raw_y_coord_list = [_v[0] for _v in raw_vertex_list]
    raw_x_coord_list = [_v[1] for _v in raw_vertex_list]

    def _calc_scale(data):
        return np.max(data) - np.min(data)

    def _calc_rms(data):
        return np.sqrt(np.mean(np.square(data)))

    # バウンディングボックスの対角線の長さを基準の長さとする
    yscale = _calc_scale(raw_y_coord_list)
    xscale = _calc_scale(raw_x_coord_list)
    base_length = _calc_rms([yscale, xscale])

    # 座標値の絶対誤差を基準長さの1e-5倍に設定する(倍精度を想定)
    epsilon = base_length * 1e-5
    epsilon2 = epsilon * epsilon

    # グラフ表現への変換

    # unique list of nodes [(y0, x0), ... (yN, xN)]
    # , where N is the number of identical nodes
    full_node_list = []

    # unique list of edges [(s0, t0), ... (sM, tM)]
    # , where each s_j (or t_j) is node id
    # , where M is the number of of identical edges
    #   (note that swapping SOURCE <-> TARGET nodes is ignored here)
    full_edge_list = []

    # list of polygon [(e{0,0}, ... e{0,L_0}), ... (e{K,0}, ... e{K,L_K})]
    # , where e{i, j} represents the edge id of j'th edge of i'th polygon
    # , where K is the number of polygons
    # , where L_k represents the number of edges of k'th polygon
    full_polygon_list = []

    # Vertex Insertion
    new_polys = dict()
    # 辺上の頂点を追加する処理のためにまず最初に頂点リストを作成する
    for poly in puzzle.polygons:
        name = poly.name
        current_poly = []
        node_current_list = poly.xys
        node_next_list = np.concatenate([poly.xys[1:], poly.xys[:1]])

        for edge in zip(node_current_list, node_next_list):
            src, tgt = edge
            current_poly.append(src)

            # エッジの上にある頂点のリストを取得
            vertices = []
            for n in raw_vertex_list:
                if is_same(n, src, epsilon):
                    continue
                if is_same(n, tgt, epsilon):
                    continue
                if is_on_line(edge, n, epsilon2):
                    vertices.append(n)
            if len(vertices) == 0:
                ...
            elif len(vertices) == 1:
                current_poly.append(vertices[0])
            else:
                # 複数ある場合は重複を除外しながらsrcに近い順に追加
                vert_dist_list = [
                    (*v, dist(src, v)) for v in vertices]
                vert_dist_list.sort(key=lambda v: v[2])
                d_found = 0
                for i in range(len(vert_dist_list)):
                    d = vert_dist_list[i][2]
                    if d > d_found + epsilon2:
                        d_found = d
                        current_poly.append(vert_dist_list[i][:2])

        new_polys[name] = current_poly

    # Vertex Insertion 終了

    # 頂点の追加によってエッジが変化している

    # 逐次検査して都度登録
    for name, poly in new_polys.items():
        # print(name)
        # print(poly)
        current_poly = []
        node_current_list = poly
        node_next_list = np.concatenate([poly[1:], poly[:1]])
        for src, tgt in zip(node_current_list, node_next_list):

            found_src = False
            for i_node, n in enumerate(full_node_list):
                if is_same(src, n, epsilon):
                    found_src = i_node
                    break
            if found_src is False:
                full_node_list.append(src)
                found_src = len(full_node_list) - 1

            found_tgt = False
            for i_node, n in enumerate(full_node_list):
                if is_same(tgt, n, epsilon):
                    found_tgt = i_node
                    break
            if found_tgt is False:
                full_node_list.append(tgt)
                found_tgt = len(full_node_list) - 1

            found_edge = False
            for i_edge, e in enumerate(full_edge_list):
                if (e[0] == found_src) and (e[1] == found_tgt):
                    found_edge = i_edge
                    break
            if found_edge is False:
                full_edge_list.append((found_src, found_tgt))
                found_edge = len(full_edge_list) - 1

            current_poly.append(found_edge)

        full_polygon_list.append(current_poly)

    # print("-- Nodes --", len(full_node_list))
    # print(full_node_list)
    # print("-- Edges --", len(full_edge_list))
    # print(full_edge_list)
    # print("-- Polygons --")
    # print(full_polygon_list)
    return full_node_list, full_edge_list, full_polygon_list


def get_border(full_node_list, full_edge_list, full_polygon_list):
    # 外周の取得
    # エッジのリストに一度だけ現れるエッジを、外周のエッジと判定
    full_border_list = []
    for id_edge, edge in enumerate(full_edge_list):
        reversed_edge = (edge[1], edge[0])
        found = False
        for e in full_edge_list:
            if (e[0] == reversed_edge[0]) and (e[1] == reversed_edge[1]):
                found = True
                break
        if not found:
            full_border_list.append(id_edge)

    # 外周エッジリストを、適切な順番に並び変える
    # [FIXME] 頂点を共有しているシルエットが正しい経路にならない
    sorted_border_list_list = []
    sorted_border_list = []
    num_border = len(full_border_list)
    registered = np.zeros(num_border, dtype=bool)
    for _ in range(2 * num_border):
        if len(sorted_border_list) != 0:
            prev_tgt_id = full_edge_list[sorted_border_list[-1]][1]
            found_count = 0
            found_list = []
            for i in range(num_border):
                if registered[i]:
                    continue
                src_id = full_edge_list[full_border_list[i]][0]
                if src_id == prev_tgt_id:
                    found_count += 1
                    found_list.append(i)
            if found_count == 0:
                # ループになっているはず
                # 閉じたボーダーとして登録して、次の連結要素の処理へ
                sorted_border_list_list.append(sorted_border_list)
                sorted_border_list = []
            elif found_count == 1:
                new_id = found_list[0]
                sorted_border_list.append(full_border_list[new_id])
                registered[new_id] = True
            else:
                # 頂点接触の場合、候補OUTエッジの内、atan2がINエッジを
                # 反転したエッジのatan2の次に大きいエッジを選択する
                prev_bord = full_edge_list[sorted_border_list[-1]]
                prev_reversed_edge = [
                    full_node_list[prev_bord[1]],
                    full_node_list[prev_bord[0]]
                    ]
                base_atan2 = edge_atan2(prev_reversed_edge)
                most_small_diff = float('inf')
                most_small_id = None
                for i in found_list:
                    src_id, tgt_id = full_edge_list[full_border_list[i]]
                    src = full_node_list[src_id]
                    tgt = full_node_list[tgt_id]
                    e = (src, tgt)
                    current_atan2 = edge_atan2(e)
                    diff = current_atan2 - base_atan2
                    if diff < 0:
                        diff += 2 * math.pi
                    if diff < most_small_diff:
                        most_small_diff = diff
                        most_small_id = i
                new_id = most_small_id
                sorted_border_list.append(full_border_list[new_id])
                registered[new_id] = True

        if len(sorted_border_list) == 0:
            # 未登録のIDを取得
            found = False
            for i in range(num_border):
                if not registered[i]:
                    new_id = i
                    found = True
                    break
            if not found:
                break
            else:
                sorted_border_list.append(full_border_list[new_id])
                registered[new_id] = True
    if len(sorted_border_list) != 0:
        sorted_border_list_list.append(sorted_border_list)

    # print("-- Borders --")
    # print(sorted_border_list_list)
    # for sbl in sorted_border_list_list:
    #     for e in sbl:
    #         s, t = full_edge_list[e]
    #         print(s, t)

    return sorted_border_list_list


class DegreeCollector:
    __slots__ = ['degrees']

    DEGREE_EPSILON = 0.1

    def __init__(self):
        self.degrees = []

    def append(self, degrees):
        for d in degrees:
            found = False
            for _d in self.degrees:
                if abs(d - _d) < self.DEGREE_EPSILON:
                    found = True
                    break
            if not found:
                self.degrees.append(d)
        self.degrees.sort()


# 評価指標
# - シルエット頂点の評価指標
# - 頂点(シルエット内部も含む)の評価指標
# - シルエットエッジの評価指標
# - エッジ(シルエット内部も含む)の評価指標
class SilhouetteSurfaceNodeEvaluator(SilhouetteEvaluator):
    __slots__ = [
        'num_borders',
        'num_nodes',
        'num_valid_nodes',
        'num_inner_nodes',
        'degree_list',
        'degree_word',
        'degree_bag',
        ]

    DEGREE_EPSILON = 0.1

    def set_degree_word(self, words):
        self.degree_word = words

    def __init__(self):
        self.degree_word = None
        self.reset()

    def reset(self):
        self.num_borders = 0
        self.num_nodes = 0
        self.num_valid_nodes = 0
        self.num_inner_nodes = 0
        self.degree_list = []
        self.degree_bag = None
        if self.degree_word is not None:
            self.degree_bag = np.zeros(len(self.degree_word), dtype=np.int32)

    def eval(self, puzzle):
        self.reset()
        nodes, edges, polygons = get_graph_repr(puzzle)
        borders = get_border(nodes, edges, polygons)
        for border in borders:
            # 明らかにおかしいborderがある場合は後段の評価処理は無意味なので
            # 事前にraiseしておく
            if len(border) == 1:
                raise ValueError("get_border Failed")
        number_of_borders = len(borders)
        self.num_borders = number_of_borders
        # print('number of borders:', number_of_borders)

        total_num_nodes = 0
        total_num_valid_nodes = 0
        total_num_inner_nodes = 0

        for borders_id in range(len(borders)):
            b0 = borders[borders_id]
            number_of_nodes_on_border = len(b0)
            num_nodes = number_of_nodes_on_border
            total_num_nodes += num_nodes

            # print(f'{number_of_nodes_on_border=}')
            b0_next_list = np.concatenate([b0[1:], b0[:1]])
            degrees = []
            for b, b_next in zip(b0, b0_next_list):
                e = edges[b]
                e_next = edges[b_next]
                e_repr = [nodes[e[0]], nodes[e[1]]]
                e_next_repr = [nodes[e_next[0]], nodes[e_next[1]]]
                deg = vector_degree(e_repr, e_next_repr)
                degrees.append(deg)
            # print(degrees)
            self.degree_list.append(degrees)
            num_sufficient_small_deg_nodes = len([
                d for d in degrees if abs(d - 180) < 1])
            number_of_valid_nodes = number_of_nodes_on_border - num_sufficient_small_deg_nodes
            num_valid_nodes = number_of_valid_nodes
            total_num_valid_nodes += num_valid_nodes
            # print(f'{number_of_valid_nodes=}')

        if self.degree_word is not None:
            for degrees in self.degree_list:
                for deg in degrees:
                    for i, w in enumerate(self.degree_word):
                        if abs(w - deg) < self.__class__.DEGREE_EPSILON:
                            self.degree_bag[i] += 1
                            break

        def find_node(i_node):
            for border in borders:
                for edge_ref in border:
                    edge = edges[edge_ref]
                    if i_node == edge[0]:
                        return True
                    if i_node == edge[1]:
                        return True
            return False

        for i_node in range(len(nodes)):
            if not find_node(i_node):
                total_num_inner_nodes += 1

        self.num_nodes = total_num_nodes
        self.num_valid_nodes = total_num_valid_nodes
        self.num_inner_nodes = total_num_inner_nodes
