# M1 modality transfer

`run_m1_transfer.py` trains on all modalities with the cue visible, then
evaluates with one modality masked at a time. This isolates dependence on a
single input path; it is not a multimodal generalization benchmark.

Smoke run (`seed=0`, 20 episodes) remained close to chance for most variants;
the Engram variant reached `1.0` with all views and with image/symbolic masks,
but fell to `0.5` when its byte view was masked. Because the run is short and
the task is tiny, this is a routing diagnostic only, not evidence that Engram
is intrinsically superior.
