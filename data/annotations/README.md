# Annotations

The EgoStream question annotations (2,250 curated questions expanded into 8,528
recall-conditioned evaluations) will be released here as
`egostream_annotations.json`. Until then the run configs reference this path and the
benchmark scripts will stop with a clear error if the file is missing.

Each entry contains the question, the correct answer, three distractors, the source
dataset and video id, the evidence moments, the Answer Validity Window and the sampled
query time of every recall regime. The field reference will be published together with
the file.
