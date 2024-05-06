import numpy as np
import json

class Node:
    def __init__(self,id):
        self.id=id
        self.used=False
        self.state=0
        self.parent=[]
        self.children=[]
class Tree:
    def __init__(self):
        self.tree=dict({})
    def add_node(self,node=Node(None)):
        id=node.id
        self.tree[id]=node
    def fit(self,graph):
        for g in graph:
            id=g['id_obj']
            state=g['state']
            node=Node(id)
            node.state=state
            for n in g['pred_obj']:
                node.parent.append(n)
            self.add_node(node)
    def flow(self,node_in,node_out):
        def go(node_id):
            count=0
            if node_id==node_in:
                print('connected')
                return 1
            node=self.tree[node_id]
            if node.used:
                return count
            node.used=True
            for n in node.parent:
                nid=n['id_node_pred']
                state=n['state']
                state_=self.tree[nid].state
                if state!=state_:
                    print("nid ",nid,"; childe ",node_id)
                if state*state_==0:
                    continue
                i=go(nid)
                count+=i
            return count
        n=go(node_out)
        return n

