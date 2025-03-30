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

