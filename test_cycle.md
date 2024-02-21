self= Node(1)
edge = (1,2)
reaches = {(1,2), (2,3), (3,4), (4,1)}
parents = {}
---
self = Node(2)
edge = (2,3)
reaches = {(2,3), (3,4), (4,1)}
parents = {(1,2)}
---
self = Node(3)
edge = (3,4)
reaches = {(3,4), (4,1)}
parents = {(2,3)}
---
self = Node(4)
edge = (4,1)
reaches = {((4,1))}
parents = {(3,4)}