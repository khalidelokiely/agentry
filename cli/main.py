from rich import print

from shopassist.memory.memory import Memory

if __name__ == "__main__":
    m = Memory(max_window=4)
    print(m.append('user', 'hi how are u'))
    print(m.append('user', 'hi how are u'))
    print(m.append('user', 'hi how are u'))
    print(m.append('user', 'hi how are u'))
    print(m.needs_compaction())
    print(m.append('user', 'hi how are u'))
    print(m.needs_compaction())
