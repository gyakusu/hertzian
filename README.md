# hertzian

**FFT-accelerated elastic half-space normal contact solver — Rust core with PyO3 bindings.**

<p align="center">
  <img src="docs/img/hero.png" width="100%"
       alt="Converged contact-pressure fields for the four problems the solver handles: circular Hertz, elliptic Hertz, Sneddon's cone, and a fragmented rough contact.">
</p>

<p align="center"><sub>The four contact problems the core solves today, each as its converged contact-pressure field — free-space DC-FFT + Polonsky–Keer BCCG, every case checked against an analytic reference. See the <a href="#gallery--可視化">gallery</a> for the side-by-side validation.</sub></p>

> **Status: P0–P4 complete (Draft 0.1).**
> The Rust core solves circular (sphere–plane / sphere–sphere) and elliptic
> (sphere on a torus outer equator) Hertz contact via zero-padded free-space
> DC-FFT and a Polonsky–Keer BCCG solver, each validated against its analytic
> solution. P4 adds **arbitrary height-field shapes and additive roughness** (any
> `Gap` plus a roughness layer), validated against Sneddon's non-Hertzian cone,
> an independent dense projected-Gauss–Seidel solver, and — for the rough
> contacts that have no closed form — the external [Tamaas](https://gitlab.com/tamaas/tamaas)
> code run with its free-space operator. **Python bindings** (PyO3 + `maturin`,
> zero-copy NumPy, GIL released during the solve, single `abi3` wheel for
> CPython 3.11+) expose the solver and reproduce the benchmarks from Python.
> Periodic boundaries and multi-body contact remain on the roadmap.

---

## 概要 / Overview

二つの弾性体の**法線・無摩擦接触**を、両者を**弾性半空間**で近似し、接触界面を
**共通平面上の一様格子**で離散化して解くソルバです。圧力分布と表面変位の関係は
**畳み込み** `u = K * p` となり、畳み込み定理 `û = K̂ · p̂` により **FFT** で
O(N²) → O(N log N) に高速化できます。非貫入・非引張の拘束は **Polonsky–Keer 型の
制約付き共役勾配法 (BCCG)** で解きます。自由空間（非周期）の Hertz 接触を正しく
扱うため、**ゼロパディング DC-FFT** を用います。

A solver for **normal, frictionless contact** between two elastic bodies. Both
bodies are approximated as **elastic half-spaces** and the interface is
discretised on a **single shared uniform 2D grid**. Because the half-space is
homogeneous, the pressure→displacement influence function is translation
invariant, so the relation becomes a **convolution** `u = K * p`; by the
convolution theorem `û = K̂ · p̂`, this is evaluated with the **FFT**
(O(N²) → O(N log N)). Non-penetration / non-adhesion constraints are handled by
a **constrained conjugate gradient (BCCG, Polonsky–Keer)** scheme. Free-space
(non-periodic) Hertz contact requires **zero-padded DC-FFT**.

> A uniform grid is **mandatory**: the convolution structure (and therefore the
> FFT speed-up) breaks on non-uniform grids.

### Design priority

Extensibility toward **arbitrary geometry, surface roughness, and multi-body
contact** is prioritised over the raw speed of a single contact.

### Validation roadmap

1. **Circular contact** — sphere–plane / sphere–sphere, validated against the
   analytic Hertz solution.
2. **Elliptic contact** — sphere against a torus outer race (convex–convex), to
   exercise the full non-axisymmetric machinery.
3. **Arbitrary height-field shapes & roughness** — any sampled gap, plus an
   additive roughness layer, within the half-space approximation. Cross-validated
   against Sneddon's cone (analytic, non-Hertzian), an independent dense solver,
   and Tamaas (see [Cross-validation](#cross-validation--相互検証) below).

### Out of scope for v1

Friction / tangential contact, elasto-plasticity & visco-elasticity, coatings,
adhesion (JKR/Maugis), strongly conformal contact, and GPU execution. These are
not implemented in v1 but the architecture reserves trait boundaries for them.

### Prior art

[Tamaas](https://gitlab.com/tamaas/tamaas) (EPFL, C++/Python, FFTW + OpenMP) is
the closest mature library, but is periodic-boundary by default; a Rust + PyO3
implementation distributable as native `pip` wheels is the differentiator here.
Tamaas does expose a non-periodic operator, which P4 uses as a free-space
cross-validation reference (see [Cross-validation](#cross-validation--相互検証)).

---

## Technology stack

| Layer            | Tooling                                                        |
| ---------------- | ------------------------------------------------------------- |
| Numerical core   | Rust — `ndarray`, `rustfft` / `realfft`, `rayon`              |
| Python bindings  | `PyO3` + `maturin` + `rust-numpy` (zero-copy NumPy interop)   |
| Python env / dev | [`uv`](https://docs.astral.sh/uv/) (required — no raw Python) |
| Static analysis  | `ruff` (lint+format), `mypy --strict`, `clippy -D warnings`   |

---

## Usage (Python)

```python
import numpy as np
import hertzian

# Analytic shortcut: circular Hertz (sphere on a flat). `domain` is the physical
# width of the (origin-centred) square interface grid, in metres.
sol = hertzian.solve_sphere_on_flat(
    radius=10e-3, load=50.0, e_star=70e9, grid=(256, 256), domain=1.2e-3
)
print(sol.contact_radius, sol.max_pressure, sol.approach)
print(sol.diagnostics)            # iterations, residual, converged
pressure = sol.pressure           # (nx, ny) float64 NumPy array (axis 0 = x)

# Elliptic Hertz: a sphere on a torus outer equator (convex–convex, P2).
sol = hertzian.solve_sphere_on_torus(
    sphere_radius=12e-3, tube_radius=4e-3, centre_radius=20e-3,
    load=60.0, e_star=100e9, grid=(256, 256), domain=1.2e-3,
)
print(sol.contact_half_widths, sol.ellipticity)

# Applied example: a ball pressed into a Gothic-arch (ogival) bearing groove —
# two arcs (two tori) overlaid, conformity r/Rs = 1.04. A non-zero centre_offset
# (the arc-centre shim) makes the ball ride two flanks: the contact splits in
# two. centre_offset=0 recovers a single conformal elliptic contact. Tall domain
# along the split (meridional y) axis.
sol = hertzian.solve_sphere_in_gothic_arch(
    sphere_radius=4e-3, tube_radius=4.16e-3, centre_radius=15e-3,
    centre_offset=65e-6, load=800.0, e_star=100e9,
    grid=(96, 846), domain=(0.65e-3, 5.74e-3),
)
print(sol.max_pressure)  # two flank patches, each an elliptic Hertz contact at P/2

# General entry point (P4): an arbitrary undeformed-gap height field h(x, y) —
# any shape, optionally with roughness added on top. Build the gap on a centred
# uniform grid and hand it to the solver.
nx, ny = 256, 256
dx = dy = 1.2e-3 / nx
x = (np.arange(nx) - (nx - 1) / 2) * dx
y = (np.arange(ny) - (ny - 1) / 2) * dy
sphere = (x[:, None] ** 2 + y[None, :] ** 2) / (2 * 10e-3)          # smooth base
roughness = (                                                       # added waviness
    0.2e-6
    * np.cos(2 * np.pi * x[:, None] / 1e-4)
    * np.cos(2 * np.pi * y[None, :] / 1e-4)
)
sol = hertzian.solve_height_field(
    gap=np.ascontiguousarray(sphere + roughness), load=50.0, e_star=70e9, dx=dx, dy=dy
)
print(sol.contact_area, sol.max_pressure)
```

`e_star` is the equivalent modulus `1/E* = (1−ν₁²)/E₁ + (1−ν₂²)/E₂`. The solver
runs with the GIL released, so calls parallelise across Python threads. Only the
free-space boundary is implemented in v1; `boundary="periodic"` is reserved and
raises `NotImplementedError`.

---

## Gallery / 可視化

ソルバが**現在解いている問題**を、収束した**圧力場**と、それを裏づける**解析解**の
両方で示します。各図の左が圧力場、右が解析解との比較で、滑らかな Hertz 接触は閉じた
形と、コーンは Sneddon の閉形式と、粗い接触はスムーズ基準に対する**接触の分裂とピーク
圧の上昇**で確認できます。

Every contact problem the core solves today, shown as the converged pressure
field beside the analytic reference it is validated against. The closed forms
plotted on the right are re-derived in [`scripts/render_gallery.py`](./scripts/render_gallery.py)
independently of the Rust core, so each panel shows the solver landing on its
reference rather than on itself.

### Circular Hertz — sphere on flat (P1)

![Circular Hertz contact: the pressure field fills the analytic contact circle, and the per-cell radial pressure collapses onto the Hertz ellipsoid.](docs/img/circular.png)

The axisymmetric benchmark. The pressure field (left) fills the analytic contact
circle (dashed); plotting **every** grid cell's pressure against `r/a` (right)
collapses the whole field onto the Hertz ellipsoid `p₀·√(1 − (r/a)²)` — here
`a ≈ 0.175 mm`, `p₀ ≈ 780 MPa`, matched to ~0.2 %.

### Elliptic Hertz — sphere on a torus outer equator (P2)

![Elliptic Hertz contact: an elongated pressure ellipse matching the analytic contact ellipse, with principal-axis cuts on the analytic profiles.](docs/img/elliptic.png)

The convex–convex contact is elliptic — longer circumferentially (`x`) than
meridionally (`y`). The measured patch tracks the analytic contact ellipse
(dashed, `aₓ/a_y ≈ 1.92`), and cuts along each principal axis sit on the
analytic semi-ellipsoidal profiles (`p₀ ≈ 1.74 GPa`). The eccentricity comes from
a transcendental curvature relation solved with complete elliptic integrals.

### Sneddon's cone — a non-Hertzian, singular-apex punch (P4)

![Sneddon cone contact: a sharply peaked pressure field, with the radial profile following Sneddon's arccosh law and its logarithmic apex singularity.](docs/img/cone.png)

An **arbitrary** (non-paraboloidal) gap `h = m·r`, fed through the same
height-field path as any measured surface. Unlike Hertz the pressure diverges
logarithmically at the apex, so the (mesh-set) peak is *not* compared — but the
radial profile follows Sneddon's `(E*m/2)·arccosh(a/r)` and the contact radius
`a ≈ 0.138 mm` lands within ~0.2 % of the closed form.

### Rough contact — sphere + roughness, fragmented (P4)

![Rough contact: the smooth single Hertz patch beside a sphere-plus-roughness contact that has fragmented into a grid of high-pressure asperities.](docs/img/roughness.png)

Layering a cosine roughness onto the smooth sphere (plain height-field addition)
breaks the single Hertz disc into a grid of **asperity contacts**. At the *same*
applied load the real contact area drops to ~¼ of the smooth disc and the peak
pressure rises ~5.6×, the physical signature of rough contact. Rough patches have
no closed form, so they are cross-validated against an independent dense solver
and against Tamaas (next section).

> Regenerate the gallery with `make gallery` (or
> `uv run --with matplotlib python scripts/render_gallery.py`). matplotlib is a
> render-only dependency, deliberately kept out of the locked environment — like
> the Tamaas cross-validation below — so its release cadence can never break the
> core pipeline.

---

## Applied example — a Gothic-arch bearing groove / ゴシックアーチ溝

ボールベアリングの軌道溝は、単一円弧ではなく**中心をずらした2つの円弧**（=2トーラスを
重ねた凹面）で研削されることが多く、尖頭のオージー形＝**ゴシックアーチ**になります。
玉はこの溝の底ではなく**2つのフランクに乗り**、接触は2点に**分裂**します。これは新しい
ソルバ機能ではなく、検証済みの**楕円接触プリミティブの応用**で、`r/Rs = 1.04`（玉径に
対する溝半径52%という教科書的な保形度）の保形接触です。

A ball-bearing race is often ground not as one arc but as **two arcs with
offset centres** — two tori overlaid into one concave groove, giving the pointed,
ogival **Gothic arch**. A ball rides the **two flanks** rather than the bottom, so
the contact **splits in two**. This is not a new solver capability but an
**application of the validated elliptic primitive**, at a conformal `r/Rs = 1.04`
(groove radius 52 % of the ball diameter — a textbook bearing conformity).

![Gothic-arch groove: a contact pressure field split into two elliptic flank patches at y = ±y0 with a contact-free Gothic point between them, and a meridional cut showing the solver's two flank peaks landing on the analytic half-load elliptic-Hertz semi-ellipses below the single-arc full-load peak.](docs/img/gothic.png)

The groove gap reduces to the double-welled
`h(x, y) = x²/(2 R_x) + (|y| − y₀)²/(2 R_y)` — two offset elliptic paraboloids,
the surface closest to the ball winning — with a **conformal** meridional radius
`R_y = 1/(1/Rs − 1/r)` (concave groove), a convex circumferential radius
`R_x = 1/(1/Rs + 1/R₀)`, and a flank offset `y₀ = centre_offset · Rs/(r − Rs)`:
the tiny arc-centre shim is **amplified ~25×** by the conformity, so a 65 µm shim
throws the flanks ±1.6 mm apart. At the *same* total load the single arc
(`centre_offset = 0`) makes one elliptic patch; the Gothic shim splits it into two.

The split is **load-conserving and self-validating**: each flank is an elliptic
Hertz contact carrying **half** the load, so its peak lands on the **same closed
form the P2 benchmark uses** — here `p₀ ≈ 1.74 GPa`, matching the elliptic-Hertz
gallery panel, and exactly `(1/2)^{1/3} ≈ 0.79×` the single-arc peak (`≈ 2.19 GPa`).
The Gothic point at `y = 0` carries no load. The numbers above (`Rs = 4 mm`,
`r = 4.16 mm`, `R₀ = 15 mm`, `E* = 100 GPa`, `P = 800 N`) are tuned so the flank
pressure sits in the gallery's GPa range; the per-flank equivalence to elliptic
Hertz at `P/2` and the contact-free ridge are pinned in the Rust scenario tests
and the Python binding tests.

### Tuning the shim — flanks that overlap by half / 半分だけ重なる2つのフランク

同じ溝のまま**シムを詰める**と、2つのフランク接触は離れたままではなく、**接触楕円が
半分ずつ重なり合う**ところまで近づきます。設計目標は子午線方向のフランクオフセット
`y₀ = b/2`（`b` は半荷重の孤立フランク楕円の子午線半軸）。半軸 `b` の2つの楕円の中心が
`b` だけ離れていれば、互いに**半分ずつ**を共有します。重なりは**ゴシック点を埋め**、接触は
一続き（連結）になります——分離アーチの非接触リッジとは対照的です。`|y|` の折り返しは
そのままなので、**左右対称**も変わりません。

Keep the same groove but **tighten the shim**, and the two flank contacts stop
being separated: their **contact ellipses overlap by half**. The design target is
a meridional flank offset `y₀ = b/2`, where `b` is the meridional semi-axis of one
isolated half-load flank ellipse — two ellipses of semi-axis `b` whose centres sit
`b` apart share exactly **half** their extent. The overlap **fills in the Gothic
point**, so the contact is now a single **connected** patch — the contrast with the
separated arch's contact-free ridge — and stays **mirror-symmetric** (the `|y|` fold).

![Half-overlapping Gothic-arch groove: a single connected pressure patch with the two flank contact ellipses (centres y = ±y0 = ±b/2) overlaid overlapping by half, and a meridional cut showing the solver's two peaks joined by an in-contact saddle that rides above the isolated half-load flank semi-ellipses through the shaded overlap band, capped below the single-arc full-load peak.](docs/img/gothic_overlap.png)

重なり領域には**閉形式がありません**——2つのフランクは弾性場を通じて**相互作用**し、荷重は
もはやきれいに `P/2` ずつには分かれません。重なりはピーク圧を分離時の `(1/2)^{1/3}` 値より
**押し上げ**ますが、単一アーチ（`y₀ = 0`）のピークより**下**に留まります（ここでは
`≈ 1.85 GPa`、分離フランクの `1.74 GPa` と単一アーチの `2.19 GPa` の間）。`20 µm` のシムが
`y₀ ≈ 0.51 mm` を与えます。解析的な拠り所がないので、検証は **P4 方式**——同じ格子上の独立な
密行列・射影 Gauss–Seidel 参照解との相互検証——で行います。

The overlapping regime has **no closed form**: the two flanks interact through the
elastic field, so the load no longer splits cleanly into two `P/2` Hertz patches.
The overlap **raises** the peak above the separated `(1/2)^{1/3}` value yet keeps it
**below** the single-arc (`y₀ = 0`) peak — here `≈ 1.85 GPa`, between the `1.74 GPa`
separated flank and the `2.19 GPa` single arc; a `20 µm` shim places the flanks at
`y₀ ≈ 0.51 mm`. Having no analytic anchor, it is cross-validated the **P4 way** —
against the independent dense projected-Gauss–Seidel reference on the same grid — with
the overlap's signatures (a **connected**, load-carrying Gothic point and two
symmetric flanks joined by a **saddle**) pinned in the Rust scenario and Python
binding tests.

---

## 縮約接触則 — マルチボディ動力学のための軽量 `F(δ)` / A reduced contact law for multibody dynamics

単一の滑らかな Hertz 接触は、`F = k δ^{3/2}` と力の向き `e ∥ (x − o)` という
**2入力2出力**の代数式で表せます。しかしゴシックアーチ溝のように玉が**2つのフランク**に
乗る複雑形状では、もはや単一の代数式には収まりません。一方でマルチボディ動力学の
ような繰り返し計算では、毎ステップ FFT ソルバを回す余裕はありません。そこで検証済みの
場のソルバを**軽量な力則 `F(δ)`** に蒸留し、形状に合わせて**回帰でフィッティング**します。

A single smooth Hertz contact is the **two-input/two-output** algebraic law
`F = k δ^{3/2}` with the force `e ∥ (x − o)` along the line of centres. A
Gothic-arch groove, where the ball rides **two flanks**, no longer fits one
algebraic form — yet a multibody inner loop cannot afford an FFT solve per step.
So we distil the validated field solver into a **lightweight force law `F(δ)`**,
**fit to the shape** by regression.

### The model / モデル

溝の子午断面で、玉中心の変位 `δ = (δ_t, δ_n)`（横断 `t̂` ・法線 `n̂`）が、接触半角 `±α`
だけ傾いた2つのフランク法線 `n̂_± = (±sin α, cos α)` 方向に各フランクを押し込みます。各
フランクは（引張なし＝正の部分 `⌊·⌋₊` のみ）Hertz 荷重を担い、合力はそのベクトル和です:

In the groove's meridional plane, a ball-centre displacement `δ = (δ_t, δ_n)`
compresses two flanks whose contact normals are tilted by the half-angle `±α`,
`n̂_± = (±sin α, cos α)`. Each carries a Hertzian load along its own normal (no
adhesion, positive part `⌊·⌋₊` only), and the net force is the vector sum:

```text
s_± = δ · n̂_±  = δ_n cos α ± δ_t sin α        (per-flank approach)
Q_± = K ⌊s_±⌋₊^{3/2}                           (per-flank Hertz load)
F(δ) = Q_+ n̂_+ + Q_- n̂_-
     →  F_t = (Q_+ − Q_-) sin α,   F_n = (Q_+ + Q_-) cos α
```

`K` は1フランクの楕円 Hertz 荷重–変位定数、`α` は幾何学的な接触角です。これは単一 Hertz
接触の `F = k δ^{3/2}` を2フランクに重ね合わせた、2入力2出力の閉形式そのものです。

`K` is one flank's elliptic-Hertz load–deflection constant and `α` the geometric
contact angle (here `α ≈ 24°`). It is the two-in/two-out closed form that superposes
two copies of the single Hertz law.

### The boundary condition — a `C¹` two-to-one transition / 境界条件：2溝→1溝の微分連続

荷重が傾くと内側フランクが除荷し、`δ_t = δ_n cot α` で**離れ**ます。接触は**2フランクから
1フランク**へ移り、力則は単一 Hertz 接触 `F = K s_+^{3/2} n̂_+`（README の1溝の
`F = k δ^{3/2}`、`e ∥ (x − o)`）に collapse します。**この遷移は `C¹`** ——力**と**その
ヤコビアン（接線剛性）がともに連続です。理由は Hertz 指数 `3/2 > 1`：除荷するフランクは
荷重**も**剛性**も**ゼロで噛み合うため、`Q_- ∝ s_-^{3/2} → 0` かつ
`dQ_-/ds_- ∝ s_-^{1/2} → 0`。**1.5乗こそが、滑らかな2→1の受け渡しを保証する構造**です。
ただし `C²` ではありません——接線剛性は `√` のカスプ（`d²Q/ds² ∝ s^{-1/2} → ∞`）を持ち、
これは Hertz 接触が無限初期勾配で硬化する、おなじみの符牒です。

As the load tilts, the inner flank unloads and at `δ_t = δ_n cot α` it **lifts
off**: the contact passes from **two flanks to one** and the law collapses onto the
single Hertz contact `F = K s_+^{3/2} n̂_+` — the README's one-groove
`F = k δ^{3/2}`, `e ∥ (x − o)`. **The transition is `C¹`**: the force *and* its
Jacobian (tangent stiffness) are continuous, because the Hertzian exponent
`3/2 > 1` makes a flank engage with zero load *and* zero stiffness
(`Q_- ∝ s_-^{3/2} → 0`, `dQ_-/ds_- ∝ s_-^{1/2} → 0`). **The `3/2` power is exactly
what guarantees the smooth two-to-one handover.** It is `C¹` but not `C²` — the
tangent stiffness has a `√` cusp (`d²Q/ds² ∝ s^{-1/2} → ∞`), the familiar signature
of a Hertz contact stiffening with an infinite initial rate.

### Fitting & verification / フィッティングと検証

![A four-panel validation of the reduced two-flank law: (A) a log-log calibration with the single-arc solver points on the K·δ^1.5 line and the two-flank points on the 2K line; (B) the calibrated force vector F_t, F_n and |F| swept transversely through the lift-off onto the single-Hertz asymptote; (C) the unloading flank following the universal (1−ξ)^1.5 curve tangent to zero, with the solver's asymmetric-well markers landing on it; (D) the effective flank count η running from ~2 (separated) toward 1 (merged) as the shim tightens.](docs/img/reduced_law.png)

`K` は単一アーチの荷重スイープから較正します：自由勾配の回帰が Hertz 指数 **1.500**
（理論値 1.5）と `K = P/δ^{3/2}` を **`R² = 1.000000`** で復元し（パネルA、解析 `K` と
**0.2 %** 一致）、2フランクは `2K` 線に重なって**重ね合わせ**を確認します。パネルB は較正済み
の `F(δ_t, δ_n)` を横断方向にスイープし、離れの先で単一 Hertz 漸近線に滑らかに乗る様子
（`C¹`）を示します。パネルC は除荷フランクが普遍曲線 `Q_-/Q_-(0) = (1−ξ)^{3/2}` に従い
**接線的にゼロへ**接する様子で、ソルバの非対称ウェル実験のマーカーが（**3 %** 以内で）その上に
乗ります——**3/2乗が `C¹` そのもの**。パネルD はシムを詰めると有効フランク数
`η = P/(K δ^{3/2})` が **1.95**（分離した2フランク）から 1（合体した単一アーチ）へ動く、
幾何駆動の2→1で、`η` が 2 に満たない分が単一 `K` モデルの残差に畳み込む弾性カップリングです。

`K` is calibrated from a single-arc load sweep: a free-slope regression recovers the
Hertz exponent **1.500** (theory 1.5) and `K = P/δ^{3/2}` at **`R² = 1.000000`**
(panel A; matching the analytic `K` to **0.2 %**), and two flanks land on the `2K`
line — **superposition** confirmed. Panel B sweeps the calibrated `F(δ_t, δ_n)`
transversely, riding smoothly onto the single-Hertz asymptote past lift-off (`C¹`).
Panel C shows the unloading flank following the universal
`Q_-/Q_-(0) = (1 − ξ)^{3/2}`, **tangent to zero**, with the solver's asymmetric-well
markers landing on it to within **3 %** — **the `3/2` power is the `C¹`**. Panel D
traces the geometry-driven two-to-one: tightening the shim runs the effective flank
count `η = P/(K δ^{3/2})` from **1.95** (two separated flanks) down toward 1 (one
merged arc), the shortfall below 2 being the elastic coupling the single-`K` model
folds into its residual.

これは新しいソルバ機能ではなく、**検証済みプリミティブの蒸留**です：閉形式の `F(δ)` は
FFT を一切呼ばず `powf` 数回で評価でき、マルチボディの内側ループに直接置けます。Rust コア
（`hertzian::GothicArchLaw`、解析ヤコビアン付き）と Python バインディング
（`hertzian.GothicArchLaw`）の両方で公開し、`C¹` と Hertz 極限、ソルバとの一致を Rust／
Python テストに固定しています。図は `make gallery`（または
`uv run --with matplotlib python scripts/fit_reduced_law.py`）で再生成します。

This is not a new solver capability but a **distillation of the validated
primitive**: the closed-form `F(δ)` calls no FFT, evaluates in a couple of `powf`s,
and drops straight into a multibody inner loop. It is exposed in both the Rust core
(`hertzian::GothicArchLaw`, with the analytic Jacobian) and the Python bindings
(`hertzian.GothicArchLaw`); the `C¹` property, the Hertz limit, and the agreement
with the field solver are pinned in the Rust and Python tests. Regenerate the figure
with `make gallery` (or
`uv run --with matplotlib python scripts/fit_reduced_law.py`).

```python
import hertzian

# Calibrate the law once from the flank geometry (no FFT solve at runtime).
law = hertzian.GothicArchLaw.from_elliptic_flank(
    radius_x=3.31e-3,  # circumferential relative radius of one flank
    radius_y=0.104,  # meridional (conformal) relative radius
    e_star=100e9,
    contact_angle=hertzian.contact_half_angle(offset=1.6e-3, ball_radius=4e-3),
)

# Then, in the multibody inner loop, evaluate F(δ) and its tangent stiffness:
f_t, f_n = law.force(2e-6, 6e-6)  # contact force vector (N)
stiffness = law.jacobian(2e-6, 6e-6)  # 2x2 tangent stiffness dF/dδ (N/m)
```

---

## Cross-validation / 相互検証

Smooth Hertz contacts are checked against their closed form, but arbitrary
shapes — and especially **rough** contacts — have no analytic reference. P4
validates them three independent ways:

| Check | What it pins | Where |
| ----- | ------------ | ----- |
| **Sneddon's cone** | the half-space *model* on a non-Hertzian, singular-apex shape (exact contact radius / approach / load) | `cone_on_flat`, `SneddonCone` (Rust); `test_cone_matches_sneddon` (Python) |
| **Dense projected-Gauss–Seidel** | the *iterative solver*, by an unrelated algorithm on the same kernel — agreement to ~10 digits on a fragmented rough patch | `DenseReference` (Rust); `rough_sphere_cross_validates_against_the_dense_reference` |
| **Tamaas (free-space)** | the *implementation*, against the mature external [Tamaas](https://gitlab.com/tamaas/tamaas) boundary-element code run with its non-periodic operator — machine-precision agreement on smooth and rough gaps | `tests/test_cross_validation.py` |

A continuum **FEM** comparison would additionally probe regimes the half-space
model excludes (finite-thickness or conformal geometry); the `InfluenceOperator`
and `Gap` trait boundaries leave room to plug one in, while the exact-elasticity
analytic references above already pin the model within its stated scope.

Tamaas is an optional, validation-only dependency, deliberately kept out of the
locked project environment so its release cadence cannot break the core
pipeline. Run the comparison with:

```sh
uv run --with tamaas pytest tests/test_cross_validation.py
```

---

## Development

### Prerequisites

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) — **the only
  supported way to run Python in this project** (see *No raw Python* below).
- A Rust toolchain via `rustup`. The exact toolchain (incl. `clippy` and
  `rustfmt`) is pinned in [`rust-toolchain.toml`](./rust-toolchain.toml) and is
  installed automatically on first `cargo`/`rustup show`.

### Quick start

```sh
make setup    # uv sync + install git hooks + Rust toolchain
make build    # build the native extension into the uv venv (maturin develop)
make test     # cargo test + pytest
make lint     # run ALL static analysis exactly as CI does (pre-commit)
make fmt      # auto-format Python (ruff) and Rust (cargo fmt)
make help     # list all targets
```

> `make` is just a convenience wrapper. The authoritative checks live in
> [`.pre-commit-config.yaml`](./.pre-commit-config.yaml), and CI runs those same
> hooks — so if `make lint` is green locally, CI's static-analysis job is too.

### No raw Python

This project **forbids invoking Python directly** (`python …`, `pip …`,
`requirements.txt`, `setup.py`, conda, etc.). Everything goes through `uv`:

```sh
uv run python ...     # ✅ instead of `python ...`
uv run pytest         # ✅
uv add <pkg>          # ✅ instead of `pip install <pkg>`
uvx <tool>            # ✅ one-off tools
```

The policy is enforced by [`scripts/check-no-raw-python.sh`](./scripts/check-no-raw-python.sh),
which runs in pre-commit and CI. Rationale and details are in
[`CONTRIBUTING.md`](./CONTRIBUTING.md).

---

## License

Dual-licensed under either [MIT](./LICENSE-MIT) or
[Apache-2.0](./LICENSE-APACHE) at your option.
