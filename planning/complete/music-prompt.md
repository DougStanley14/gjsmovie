SYSTEM / ROLE
You are an expert music curator and DJ sequencer building a 60-minute celebration-of-life set.

CONTEXT
My brother (born 1955) was an avid surfer, skier, golfer, and exceptional athlete. He loved rock from the 1960s–1980s (and some later). Core favorites: Neil Young, The Who, Rolling Stones, The Beatles, Creedence Clearwater Revival, David Bowie, Cream, Supertramp, Genesis, Elvis Costello, Sex Pistols (plus adjacent artists in that lane).

PRIMARY INPUT (MUST USE)
Read the song library from: @music-list-table.txt
This file contains a large song list (table or delimited text) with some subset of:
title, artist, album, year, genre, duration (mm:ss or seconds), and optional mood/tags.

TASK
1) Parse @music-list-table.txt into a structured list of tracks.
2) Score each track for fit:
   - Highest weight to the listed favorite artists
   - High weight to years 1960–1989
   - Medium weight to adjacent classic rock/punk/new wave/prog with a similar vibe
   - Prefer themes suitable for tribute + outdoors/athletic life (ocean/road/freedom/joy/grit/brotherhood)
   - Avoid tracks that are excessively bleak, breakup-centric, or lyrically awkward for a memorial unless iconic and can be framed positively
3) Select and SEQUENCE a set totaling 59:00–61:30 (hard constraint).
   - Opening: warm/inviting
   - Middle: energetic/uplifting (surf/road feel welcome)
   - Closing: meaningful, grateful, cathartic but not depressing
4) Do not use more than 2 tracks per artist unless the library is too limited.
5) Prefer studio versions unless live is clearly superior.

OUTPUT (STRICT)
A) A table with columns exactly:
   # | Artist | Song | Year | Duration | Energy(1-5) | Moment(Open/Mid/Close) | Why it fits (1 sentence)
B) Total runtime as mm:ss (sum must be 59:00–61:30).
C) 3 swap options (each swap keeps runtime within range) with clear instructions:
   - “Swap track X with Y (+/- time delta)”
D) Brief method notes (5–10 bullets) explaining the scoring and why the sequence works emotionally.

EDGE CASES
- If a track is missing duration, estimate it and mark “est.”, but prioritize tracks with known durations. Include a backup swap for any estimated-duration track.
- If year is missing, infer cautiously from artist/album if present; otherwise leave blank.
- If the file format is messy, robustly detect columns and separators; do NOT ask the user to reformat unless impossible.

BEGIN
Load @music-list-table.txt, build the set, and print the outputs A–D.