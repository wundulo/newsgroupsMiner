"""
tree.py
"""

# global variable

from article import *
class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
        
    def __repr__(self):
        return self.value


def assertCreateNode(obj):
    ''' create a node for input object if it is not a node already '''
    
    if isinstance(obj, tree.Node):
        return obj
    else:
        return tree.Node(str(obj))


def printChildren(Node, level=0, s=None):
    ''' print out all decendents of a given parent node '''
    
    if Node is None: return
    print Node.value,
    print "/", 
    printChildren(Node.left)
    print "\\", 
    printChildren(Node.right)


if __name__ == "__main__":
    
    e1 = Node("e1")
    e2 = Node("e2")
    e3 = Node("e3")
    e4 = Node("e4")
    
    e1e2 = Node("e1e2", e2, e1)
    e1e2e3 = Node("e1e2e3", e3, e1e2)
    
    e1e2e3e4 = Node("e1e2e3e4", e4, e1e2e3)
    
    printChildren(e1e2e3e4)
