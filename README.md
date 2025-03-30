# bible-similarity

a quick and dirty proof of concept repo for experimenting with text reuse detetection of biblical references using chroma vector database

## data source

bible data is taken from https://github.com/LukeSmithxyz/vul

## scripts

### Indexing

```shell
uv run index_bible.py
```

takes about 15 minutes to populate a chroma-db index with the vulgata bible text

### Qurying
```shell
uv run query.py "Et factum est vespere et mane, dies quintus."
```
returns
```
############

best matches for query:
Et factum est vespere et mane, dies quintus.

Gen, 1, 1, 23: Et factum est vespere et mane, dies quintus.
Gen, 1, 1, 19: Et factum est vespere et mane, dies quartus.
Gen, 1, 1, 13: Et factum est vespere et mane, dies tertius.
############
```