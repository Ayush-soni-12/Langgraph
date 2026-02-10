from langgraph.store.memory import InMemoryStore

#  creating a store 

store = InMemoryStore()

#  creating a namespace

namespace1 = ("user", "u1")

# adding memories using put method

store.put(namespace1, "memory1", {"data":"user like pizza"})
store.put(namespace1, "memory2", {"data":"user prefer dark mode"})


# creating a new 
namespace2 = ("user", "u2")

#  adding memories using put method

store.put(namespace2, "memory1", {"data":"user like pasta"})
store.put(namespace2, "memory2", {"data":"user prefer light mode"})


#  for retrieving the memories using get method

store.get(namespace1, "memory1")  # returns {'data': 'user like pizza'}

#  to get all memories from a 

items = store.search(namespace1)  # returns all memories in the namespace

for item in items:
    print(item.key, item.value)
