import pygenten

entry = pygenten.Entry()
entry.iteration = 10
history = pygenten.PerfHistory()
history.addEntry(entry)
entry_last = history.lastEntry()
print(entry_last.iteration)
entry_0 = history[0]
print(entry_0)
print(history)
