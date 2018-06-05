The example provided in [pipeline.json](./pipeline.json) demonstrates using
`filters.assign` to reset all Classifications to 0 and `filters.smrf` to
classify points as either ground (Classification = 2) or unclassified
(Classifiation = 1).

The same processing steps can be performed via `pdal translate` on the command
line, as shown.

```console
pdal translate <input> <output> assign smrf \
    --filters.assign.assignment="Classification[:]=0"
```
