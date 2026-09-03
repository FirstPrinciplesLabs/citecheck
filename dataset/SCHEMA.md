# Dataset schema

`corruption_metadata.json` is the single source of truth for the
hallucination-detection benchmark in this repository. Every CLI in this
package (`citecheck-evaluate`, `citecheck-baseline`, `citecheck-compare`)
loads citations and ground-truth labels from this file.

The dataset contains 42 collections (one per scientific subtopic) with a
total of 982 citations, balanced across three classes:

| ground-truth label    | meaning                                                      |
|-----------------------|--------------------------------------------------------------|
| `valid`               | original citation, kept unchanged                            |
| `hallucinated_minor`  | small metadata error (year ±1-2, partial title, initials …)  |
| `hallucinated_major`  | citation refers to a different / fabricated paper            |

Loaders in `citecheck.eval.ground_truth` map these to the
three classification labels used by the detector:

```
valid              -> exact_match
hallucinated_minor -> minor_hallucination
hallucinated_major -> major_hallucination
```

## Top-level structure

```jsonc
{
  "generation_timestamp": "<unix-seconds-as-string>",
  "configuration": {
    "random_seed": 42,
    "unchanged_percent": 34.0,
    "hallucinated_percent": 66.0,
    "hallucinated_minor_percent": 50.0,
    "hallucinated_major_percent": 50.0
  },
  "collections": [ /* see below */ ]
}
```

## Per-collection structure

Each entry in `collections` groups the citations drawn from one source
paper within a given topic:

```jsonc
{
  "collection_id":     "1-astrophysics",          // unique key, "<N>-<topic>"
  "topic":             "astrophysics",            // broad subject area
  "subtopic":          "Observational constraints on the dark energy equation of state (w): ...",
  "total_citations":   14,
  "corruption_summary": {
    "unchanged":           5,
    "hallucinated_minor":  4,
    "hallucinated_major":  5
  },
  "citations": [ /* see below */ ]
}
```

## Per-citation structure

Each entry in `citations` is one labelled citation:

```jsonc
{
  "citation_number":      1,
  "original_citation":    "[Bean & Melchiorri, 2009, Current constraints on the dark energy equation of state](https://arxiv.org/pdf/astro-ph/0110472.pdf)",
  "error_type":           "hallucination",          // "none" | "hallucination"
  "error_severity":       "major",                  // "minor" | "major"  (absent when error_type = "none")
  "corrupted_citation":   "[Farnsworth & Qanzar, 2016, Temporal Dynamics ...](https://arxiv.org/pdf/astro-ph/1602.8745.pdf)",
  "ground_truth_label":   "hallucinated_major",     // "valid" | "hallucinated_minor" | "hallucinated_major"
  "changes":              "Title changed from ... to ...; year changed to 2016; URL fabricated."
}
```

The detector's input is the `corrupted_citation` string; everything else
is metadata for evaluation and analysis.

## License

The benchmark dataset is released under the [MIT License](../LICENSE),
same as the source code.
