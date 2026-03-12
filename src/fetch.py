import requests
import json
import time
import re


def fetch_papers(max_results=500):
    base_url = "http://export.arxiv.org/api/query"
    papers = []
    batch_size = 100

    for start in range(0, max_results, batch_size):
        params = {
            "search_query": "cat:cs.CL",
            "start": start,
            "max_results": batch_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }

        print(f"Fetching papers {start+1} to {start+batch_size}...")
        response = requests.get(base_url, params=params)

        if response.status_code != 200:
            print(f"Something went wrong: {response.status_code}")
            break

        entries = response.text.split("<entry>")[1:]

        for entry in entries:
            def get_tag(tag, text):
                s = text.find(f"<{tag}")
                e = text.find(f"</{tag}>")
                if s == -1 or e == -1:
                    return ""
                s = text.find(">", s) + 1
                return text[s:e].strip()

            title    = get_tag("title", entry).replace("\n", " ").strip()
            abstract = get_tag("summary", entry).replace("\n", " ").strip()
            authors  = [get_tag("name", b) for b in entry.split("<author>")[1:] if get_tag("name", b)]
            year     = get_tag("published", entry)[:4]
            cats     = re.findall(r'term="([^"]+)"', entry)
            arxiv_id = get_tag("id", entry).split("/abs/")[-1].strip()

            if title and abstract:
                papers.append({
                    "id": arxiv_id,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "year": year,
                    "categories": cats
                })

        time.sleep(3)

    print(f"Done! Got {len(papers)} papers.")
    return papers


if __name__ == "__main__":
    papers = fetch_papers(max_results=500)
    with open("data/arxiv_papers.json", "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    print("Saved to data/arxiv_papers.json")