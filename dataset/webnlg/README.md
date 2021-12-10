# WebNLG Corpus XML Reader

XML reader for the [WebNLG corpus](https://gitlab.com/shimorina/webnlg-dataset).

## Requirements
* Tested with Python3.6
* No dependencies

## Some usage commands
Download data, adjust your path, and load corpus to memory
```
>>> from benchmark_reader import Benchmark
>>> from benchmark_reader import select_files
>>> b = Benchmark()
>>> files = select_files('./challenge_split/ru/train')
>>> b.fill_benchmark(files)
```

Output some corpus statistics
```
>>> print("Number of entries: ", b.entry_count())
Number of entries:  5585
>>> print("Number of texts: ", b.total_lexcount())
Number of texts:  29224
>>> print("Number of distinct properties: ", len(list(b.unique_p_mtriples())))
Number of distinct properties:  229
```

Access entries
```
>>> entry = b.entries[2156]
>>> entry.size
'2'
>>> entry.category
'SportsTeam'
>>> entry.shape
'(X (X (X)))'
>>> entry.shape_type
'chain'
>>> entry.list_triples()
['A.C._Lumezzane | manager | Michele_Marcolini', 'Michele_Marcolini | club | F.C._Bari_1908']
```

Access texts
```
>>> texts = entry.lexs
>>> texts[4].lex
'Michele Marcolini manages the A.C. Lumezzane, he played for F.C. Bari 1908.'
>>> texts[5].lex
'Мишель Марколини управляет ФК "Лумеццане", он играл за ФК "Бари".'
```

Access links
```
>>> entry.links[0].s
'Michele Marcolini'
>>> entry.links[0].o
'Мишель Марколини'
>>> entry.links[0].p
'sameAs'
```

Access DBpedia links
```
>>> entry.dbpedialinks[1].s
'F.C._Bari_1908'
>>> entry.dbpedialinks[1].o
'Бари_(футбольный_клуб)'
```

Convert to JSON and write to file
```
>>> b.b2json('your_path', 'your_filename.json')
```

See also `example.py`.

## Generating references for the [RDF-to-text evaluation](https://github.com/WebNLG/GenerationEval)
See `generate_references.py`.