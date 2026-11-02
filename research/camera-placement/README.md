# Historical camera-placement evaluations

These folders contain preserved OptiTrack camera-placement reports and figures,
formerly stored as `Flex_*` directories at the repository root and under
`archive/`. They are research results, not runtime caches or calibration inputs.
No report or figure was regenerated or modified during the move.

The folder names are experiment/version labels. Each report below describes a
four-camera configuration; the `3` and `13` labels do not count active cameras.
The original meaning of those labels is not encoded in the report metadata.

| Evaluation | Cameras | Reported image size |
| --- | --- | --- |
| [Flex_13_Placement_Evaluation](Flex_13_Placement_Evaluation/Flex_13_Placement_Evaluation.txt) | 4 | 1280 x 1024 px |
| [Flex_13_Placement_Evaluation_v2](Flex_13_Placement_Evaluation_v2/placement_evaluation.txt) | 4 | 1280 x 1024 px |
| [Flex_13_Placement_Evaluation_v3](Flex_13_Placement_Evaluation_v3/placement_evaluation.txt) | 4 | 1280 x 1024 px |
| [Flex_13_Placement_Evaluation_v4](Flex_13_Placement_Evaluation_v4/placement_evaluation.txt) | 4 | 1280 x 1024 px |
| [Flex_13_Placement_Evaluation_v5](Flex_13_Placement_Evaluation_v5/placement_evaluation.txt) | 4 | 1280 x 1024 px |
| [Flex_13_Placement_Evaluation_v6](Flex_13_Placement_Evaluation_v6/placement_evaluation.txt) | 4 | 1280 x 1024 px |
| [Flex_13_Placement_Evaluation_v7](Flex_13_Placement_Evaluation_v7/placement_evaluation.txt) | 4 | 1280 x 1024 px |
| [Flex_13_Placement_Evaluation_v8](Flex_13_Placement_Evaluation_v8/placement_evaluation.txt) | 4 | 1280 x 1024 px |
| [Flex_13_Placement_Evaluation_v9](Flex_13_Placement_Evaluation_v9/placement_evaluation.txt) | 4 | 1280 x 1024 px |
| [Flex_3_Placement_Evaluation](Flex_3_Placement_Evaluation/Flex_3_Placement_Evaluation.txt) | 4 | 640 x 480 px |

Reports record an original machine-local `multi_camera_calib_latest.npz` path,
analysis settings, coverage, convergence, and model-based uncertainty. They do
not embed a calibration hash or source revision. They were committed before the
October 2026 camera-geometry corrections, so retain them as historical evidence;
do not treat their numerical uncertainty as a newly validated accuracy result.

For a new evaluation, write to an ignored directory such as `outputs/placement/`
and retain the calibration snapshot, its hash, the source commit, and exact CLI
arguments alongside the outputs. Keep participant recordings in their established
local data locations. The live tracker does not read these historical reports.
