# hertzian

**FFT-accelerated elastic half-space normal contact solver — Rust core with PyO3 bindings.**

> ⚠️ **Status: provisional scaffolding (Draft 0.1).**
> This repository currently contains only the project *environment*: CI, shared
> pre-commit hooks, tooling configuration, and a buildable skeleton. The
> numerical solver is **not implemented yet** — it lands in the next milestone.
> This README will be expanded as the implementation progresses.

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
3. **Arbitrary height-field shapes** — convex, non-conformal contacts within the
   half-space approximation; roughness and multi-body extensions.

### Out of scope for v1

Friction / tangential contact, elasto-plasticity & visco-elasticity, coatings,
adhesion (JKR/Maugis), strongly conformal contact, and GPU execution. These are
not implemented in v1 but the architecture reserves trait boundaries for them.

### Prior art

[Tamaas](https://gitlab.com/tamaas/tamaas) (EPFL, C++/Python, FFTW + OpenMP) is
the closest mature library, but is periodic-boundary by default; a Rust + PyO3
implementation distributable as native `pip` wheels is the differentiator here.

---

## Planned technology stack

| Layer            | Tooling                                                        |
| ---------------- | ------------------------------------------------------------- |
| Numerical core   | Rust — `ndarray`, `rustfft` / `realfft`, `rayon`              |
| Python bindings  | `PyO3` + `maturin` + `rust-numpy` (zero-copy NumPy interop)   |
| Python env / dev | [`uv`](https://docs.astral.sh/uv/) (required — no raw Python) |
| Static analysis  | `ruff` (lint+format), `mypy --strict`, `clippy -D warnings`   |

Only `pyo3` is wired up today; the numerical crates are added with the solver.

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
