import numpy as np
import pandas as pd

from nba_four_factors.preseason import inputs as io

rows = []
for s in [
    x for x in io.processed_seasons() if x not in ("1998_99", "2011_12", "2019_20", "2020_21")
]:
    g = io.load_games(s).sort_values("game_date")
    g["game_no"] = g.groupby("team_id").cumcount() + 1
    ng = g.groupby("team_id").game_no.transform("max")
    early = g[g.game_no <= 55].groupby("team_id").margin.mean().rename("m_early")
    late = g[g.game_no > 60].groupby("team_id").margin.mean().rename("m_late")
    wins_early = (
        g[g.game_no <= 55]
        .assign(w=lambda d: d.margin > 0)
        .groupby("team_id")
        .w.mean()
        .rename("wp_early")
    )
    t = pd.concat([early, late, wins_early], axis=1).reset_index()
    t["rank_early"] = t.m_early.rank()
    t["season"] = s
    rows.append(t)
T = pd.concat(rows)
T["dip"] = T.m_late - T.m_early
T["era"] = np.where(T.season >= "2018_19", "2019+ flatter odds", "pre-2019 steep odds")
T["tier"] = pd.cut(
    T.rank_early, [0, 3, 6, 10, 20, 30], labels=["bottom 3", "4-6", "7-10", "11-20", "top 10"]
)
print("Margin change, games 61-82 minus games 1-55 (points per game); negative = got worse late")
for era, ge in T.groupby("era"):
    print(" ", era)
    for tier, gt in ge.groupby("tier", observed=True):
        se = gt.dip.std(ddof=1) / np.sqrt(len(gt))
        print(
            f"    {tier:8s} n={len(gt):3d}  dip {gt.dip.mean():+.2f} (SE {se:.2f})  -> about {gt.dip.mean() * 22 * 0.033:+.1f} wins over 22 games"
        )
mid = T[T.tier.isin(["11-20"])].groupby("era").dip.mean()
print("mid-table dip by era (regression to mean baseline):", mid.round(2).to_dict())

print()
print(
    "Regression-adjusted: late margin minus what early margin predicts (fit on teams ranked 11-30 in each era)"
)
for era, ge in T.groupby("era"):
    ref = ge[ge.rank_early > 10]
    b = np.polyfit(ref.m_early, ref.m_late, 1)
    ge = ge.assign(res=ge.m_late - np.polyval(b, ge.m_early))
    print(f"  {era}: slope {b[0]:.2f}")
    for tier, gt in ge.groupby("tier", observed=True):
        se = gt.res.std(ddof=1) / np.sqrt(len(gt))
        print(
            f"    {tier:8s} n={len(gt):3d}  late shortfall {gt.res.mean():+.2f} pts/game (SE {se:.2f})  ~ {gt.res.mean() * 22 * 0.033:+.1f} wins"
        )
