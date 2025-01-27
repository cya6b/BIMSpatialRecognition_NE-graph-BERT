'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2024 Yian Chen <cya187508866962021@163.com>

# This part is an improved algorithm of the original WL algorithm: the WLE

from code.base_class.method import method
import hashlib


class MethodWLENodeColoring(method):
    data = None
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    def setting_init(self, node_list, link_list):
        # 首先，创建一个集合存储所有存在的边，存储每条边的无序对
        existing_edges = set()
        for u1, u2 in link_list:
            existing_edges.add((u1, u2))
            existing_edges.add((u2, u1))  # 如果是无向图，确保双向都被添加

        for node in node_list:
            self.node_color_dict[node] = 1
            self.node_neighbor_dict[node] = {}

        # 初始化每个节点的邻居特征
        for node in node_list:
            for other_node in node_list:
                if node != other_node:
                    if (node, other_node) in existing_edges:
                        self.node_neighbor_dict[node][other_node] = 1
                    else:
                        self.node_neighbor_dict[node][other_node] = 0

    def WLE_recursion(self, node_list):
        iteration_count = 1
        while True:
            new_color_dict = {}
            for node in node_list:
                neighbors = self.node_neighbor_dict[node]
                # 现在考虑每个邻居的颜色和相应边的特征
                neighbor_color_list = [
                    f"{self.node_color_dict[neb]}_{self.node_neighbor_dict[node][neb]}" for neb in neighbors
                ]
                color_string_list = [str(self.node_color_dict[node])] + sorted(neighbor_color_list)
                color_string = "_".join(color_string_list)
                hash_object = hashlib.md5(color_string.encode())
                hashing = hash_object.hexdigest()
                new_color_dict[node] = hashing

            color_index_dict = {k: v + 1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
            for node in new_color_dict:
                new_color_dict[node] = color_index_dict[new_color_dict[node]]

            if self.node_color_dict == new_color_dict or iteration_count == self.max_iter:
                return
            else:
                self.node_color_dict = new_color_dict
            iteration_count += 1

    def run(self):
        node_list = self.data['idx']
        link_list = self.data['edges']
        self.setting_init(node_list, link_list)
        self.WLE_recursion(node_list)
        return self.node_color_dict
