# Path convention

Scripts reference data through the placeholder `${UPDOWN_SC_ROOT}` (originally
the experiment machine's home directory). Set it to the directory where you
unpack the data package, preserving the relative layout listed in
`data/README.md`, e.g.:

```bash
export UPDOWN_SC_ROOT=/path/to/unpacked
```

Shell wrappers expand the variable normally. Current Python statistics and PR
curve entry points also expand it from the environment. A few archived
protocol scripts preserve the literal placeholder for provenance; materialize
those scripts in a working copy with `envsubst` before running them, for
example:

```bash
envsubst < archived_script.py > /tmp/archived_script.py
python /tmp/archived_script.py
```
