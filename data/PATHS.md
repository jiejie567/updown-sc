# Path convention

Scripts reference data through the placeholder `${UPDOWN_SC_ROOT}` (originally
the experiment machine's home directory). Set it to the directory where you
unpack the data package, preserving the relative layout listed in
`data/README.md`, e.g.:

```bash
export UPDOWN_SC_ROOT=/path/to/unpacked
```

Python scripts that embed the placeholder read it literally; substitute it
(`envsubst` or a one-line sed) or export the variable before running the shell
wrappers.
