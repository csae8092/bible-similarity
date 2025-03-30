import os
import chromadb
import argparse


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Query the Vulgate database")
    parser.add_argument(
        "query",
        type=str,
        nargs="?",  # Makes the argument optional
        default="in principio creavit",  # Default query
        help="The Latin text to search for",
    )

    args = parser.parse_args()

    chroma_db = os.path.join("chroma_data")
    chroma_client = chromadb.PersistentClient(path=chroma_db)
    collection = chroma_client.get_collection(name="vulgata")

    results = collection.query(query_texts=[args.query], n_results=3)

    print("############")
    print("best matches for query:")
    print(f"{args.query}\n")
    for i, x in enumerate(results["ids"][0]):
        print(f'{x.replace("_", ", ")}: {results["documents"][0][i]}')
    print("###########")


if __name__ == "__main__":
    main()
