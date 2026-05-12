# Blog and Bluesky Ideas

Parking lot for short-form writeups: blog posts, bluesky threads,
methodological notes, conference-aside material. Anything analytically
interesting but too narrow or off-topic for the thesis itself.

Working-notebook style: each entry is a sketch, not a finished outline.
Add liberally, prune later, promote to actual drafts when they ripen.

Format per entry:

- Title
- Status (parked / drafting / published)
- Source (which session or notebook surfaced it)
- The hook
- The argument
- Data or work needed
- Target length and format

---

## 1. Mechanism discrimination for the off-vs-def four-factor cross-block

**Status:** Parked.

**Source:** EDA 03 §3 (cross-factor relationships notebook, Session 12,
2026-05).

### The hook

Within a single NBA team-game, the correlation between this team's
offensive FT rate and the opponent's offensive FT rate is +0.18 across
68k+ team-game rows spanning 1997-98 to 2025-26. The same diagonal
pattern holds for eFG (+0.14), TOV (+0.12), and offensive rebounding
(+0.09). Some per-game scalar is lifting both teams' rate stats
together.

Quick: what's causing this? Most readers will say "pace" or
"officiating" within five seconds. Both are wrong, or rather both are
unproven. The data has at least six observationally equivalent
explanations.

### The argument

The six candidates:

1. **Pace coupling.** Both teams play in the same possession-paced
   game. High pace lifts both teams' transition makes (eFG up) and
   shortens possessions (TOV down per possession).
2. **Officiating crew.** A whistle-happy ref crew lifts both teams' FT
   rates regardless of how the teams play.
3. **Behavioral coupling.** Physical play begets physical play, and
   matched aggression lifts both teams' FT rates regardless of who's
   calling fouls.
4. **Score-state coupling.** Blowouts produce garbage-time minutes
   that distort rate stats for both sides asymmetrically.
5. **Opponent-quality coupling.** Strong defenses on both sides
   produce simultaneous low-eFG games without any pace or ref
   involvement.
6. **Style-matchup coupling.** Slow-vs-fast or finesse-vs-physical
   matchups pull both teams away from their season averages in joint
   ways the framework can't see.

Each predicts the same correlation pattern in box-score data. None
can be discriminated using box scores alone.

What would discriminate them:

- **Pace test.** Residualize both teams' rate stats against game pace
  (possessions per 48). If the cross-block correlations shrink toward
  zero, pace was doing the work.
- **Officiating test.** Include ref crew as a fixed effect. Variance
  absorbed by crew identity is officiating's share.
- **Behavioral test.** Subset to high-foul games vs low-foul games
  within matched ref crews. If the FT-rate diagonal persists at
  matched officiating, behavior is contributing on top.
- **Score-state test.** Subset to within-10-point games or
  first-three-quarters minutes only. Coupling shrinks = score state
  was a factor.
- **Opponent-quality test.** Residualize each team's factors against
  the opponent's season-average defensive rating. Coupling shrinks =
  opponent quality was the scalar.
- **Style-matchup test.** Cluster team-seasons by style; test whether
  cross-block correlations vary systematically by style-pair.

The honest answer is probably some weighted combination of all six,
and the weights probably differ across the four factors. FT rate
might be mostly officiating + behavior. eFG might be mostly pace +
opponent quality. TOV might be mostly pace + score state. ORB might
be the most genuinely zero-sum and thus the most surprising as a
positive (+0.09) coupling.

### The methodological punchline

Most public "we know why X happens in NBA stats" claims dissolve
under this kind of scrutiny. The four-factor framework is a
beautifully clean abstraction, but the joint distribution of factors
across teams in a game has structure that the framework doesn't
model and that the framework's standard regressions can't see.
Honest writeups should note where mechanism is unidentified.

There's a broader point worth making about how analytical-chemistry
training transfers cleanly here: you don't say "this peak is compound
X" because the retention time matches; you run a standard, do MS, or
co-inject. The four-factor box-score data has no analogous
discriminating tool for these six mechanisms, and saying so out loud
is the strongest form of the analysis, not the weakest.

### Data or work needed

- Ref crew assignments per game (NBA officiating reports are public
  going back ~20 years, would need scraping; not in current pipeline)
- Pace (possessions per 48) per team per game (computable from
  current pipeline, just not stored as a column)
- Score-state minutes splits (would need play-by-play; `nba_api` has
  this; not in current pipeline)
- Team-season style clusters (would need a clustering step on
  team-season factor profiles; doable in the current pipeline)

Three of the six tests can be done within the current pipeline plus
play-by-play. Two need ref crew scraping. One needs an extra
clustering step.

### Target length and format

**Short version (500-800 words, bluesky thread or short post).**
Hook + six candidates with one-sentence descriptions + the
discrimination-tests list + methodological punchline. One chart: the
cross-block heatmap from EDA 03 §3b. Link to thesis once published.

**Long version (2000-2500 words, full blog post).** Actually runs
the easy two tests (pace residualization and a partial officiating
fixed-effect using available data). Presents preliminary numbers,
shows which mechanisms gain or lose support. Becomes a "here's
what's identified and what isn't" piece rather than a pure
methodological note. More work but lands harder with the Sloan
crowd.

Recommended: short version first, see if it gets traction, expand
to long version if it does. The short version is the kind of post
that gets quoted in better-known analytics writers' newsletters,
which is the highest-value distribution channel pre-thesis.

---
