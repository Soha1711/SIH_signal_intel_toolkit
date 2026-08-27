# GUI + Bitstream Correlation — Owner: Hrutu & Rahul

PyQt5 application shell: file picker, tabs for spectrum/waterfall/constellation
plots, parameter read-outs, and results export. Also implements bitstream
correlation (matching decoded bits against known sync-word/preamble patterns
to flag header vs. payload).

Starts after the other three modules have a basic working output.
See `docs/interface_contract.md` (Step 6) for exact input/output format.
