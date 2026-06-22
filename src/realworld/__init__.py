"""Real-data (Neshan travel-time) OD calibration for SUMO.

Pipeline (see configs/realworld.yaml and the plan):
  diagnose_traveltime  -> how much OD signal travel time carries (ridge ceiling)
  neshan_ingest        -> load the Neshan export
  match_edges          -> map Neshan segments to SUMO edges (+ coverage mask)
  build_observation    -> assemble the per-edge observed travel time
  calibrate            -> warm-start OD + SPSA SUMO-in-the-loop matching
  validate / whatif    -> re-simulate, score vs Neshan, and run network edits
"""
