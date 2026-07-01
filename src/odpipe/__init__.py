"""General, config-driven OD-estimation pipeline (any zone count / demand range).

All modules read configs/odpipe.yaml via CFG below; the zone count is taken from
the generated TAZ file, never hard-coded. See README_odpipe.md.
"""

CFG = "odpipe"
