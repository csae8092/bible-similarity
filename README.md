# bible-similarity

A quick and dirty proof of concept repo for experimenting with text reuse detection of biblical references using [Chroma - the open-source embedding database](https://github.com/chroma-core/chroma)

## Data source

Bible data is taken from https://github.com/LukeSmithxyz/vul

## Scripts

### Indexing

Needs to be run one time only (I guess it will throw errors running it the second time, because collection and documents already exist).
Takes about 15 minutes to populate a chroma-db index with the Vulgata bible text

```shell
uv run index_bible.py
```

### Querying
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

### JAD

For matching parts of passages from the JAD project which have already been identified by LLMs beeing a bible quote, run 
```shell
uv run find_bible_refs.py
```

The result is written into [result.json](result.json)