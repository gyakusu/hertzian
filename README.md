# hertzian

**弾性半空間の法線接触を FFT で高速に解くソルバ。Rust コア + PyO3 バインディング。**

<p align="center">
  <img src="docs/img/hero.png" width="100%"
       alt="ソルバが扱う4つの問題の収束した接触圧力場：円形 Hertz、楕円 Hertz、Sneddon のコーン、分裂した粗面接触。">
</p>

<p align="center"><sub>コアがいま解いている4つの接触問題を、収束した接触圧力場として示します。どれも自由空間 DC-FFT と Polonsky–Keer BCCG で解いたものです。図は<a href="#ギャラリー--可視化">ギャラリー</a>に、解析解との詳しい照合は<a href="docs/verification.md">検証ドキュメント</a>にまとめています。</sub></p>

> **状態：P0–P4 完了（Draft 0.1）。**
> Rust コアは、ゼロパディング自由空間 DC-FFT と Polonsky–Keer BCCG ソルバを使って、
> 円形（球–平面 / 球–球）と楕円（トーラス外赤道上の球）の Hertz 接触、それに任意形状と
> 加算的な粗さ（任意の `Gap` に粗さ層を重ねたもの、P4）を解きます。結果はすべて解析解か
> 外部コードと照合済みです（[検証ドキュメント](docs/verification.md)）。Python バインディングは
> ソルバをそのまま公開し、ベンチマークを Python から再現できます（PyO3 と `maturin`、
> ゼロコピー NumPy、ソルブ中は GIL を解放、CPython 3.11+ 向けの単一 `abi3` ホイール）。
> 周期境界とマルチボディ接触は、今後の課題として残しています。

---

## 概要

二つの弾性体の法線・無摩擦接触を解くソルバです。両者を弾性半空間で近似し、接触界面を
共通平面上の一様格子で離散化します。圧力分布 $p$ と表面変位 $u$ の関係は畳み込み
$u = K * p$ になり、畳み込み定理を使えば FFT で $O(N^2) \to O(N\log N)$ に
高速化できます：

$$u = K * p \qquad \overset{\text{FFT}}{\Longrightarrow} \qquad \hat{u} = \hat{K}\cdot\hat{p}$$

非貫入・非引張の拘束 $\bigl(p \ge 0,\ g \ge 0,\ p\,g = 0\bigr)$ は、Polonsky–Keer 型の
制約付き共役勾配法（BCCG）で解きます。自由空間（非周期）の Hertz 接触を正しく
扱うために、ゼロパディング DC-FFT を使います。

```mermaid
flowchart TB
    A["2 つの弾性体<br/>法線・無摩擦接触"] --> B["等価な弾性半空間<br/>一様格子上の未変形ギャップ h(x, y)"]
    B --> C{"Polonsky–Keer BCCG 反復"}
    C --> D["影響演算子を適用 u = K * p<br/>ゼロパディング DC-FFT・O(N log N)"]
    D --> E["非負拘束へ射影<br/>p ≥ 0, g ≥ 0, p · g = 0"]
    E --> F{"収束?"}
    F -->|"未収束"| C
    F -->|"収束"| G["接触圧力場 p と剛体接近量 δ"]
```

各反復のコストは影響演算子の適用 $u = K * p$ に集まり、これがゼロパディング DC-FFT で
$O(N\log N)$ に下がります。単純に FFT を使うと巡回畳み込みになりますが、これは接触が
周期的に並んだ状態に当たるので、孤立した Hertz 接触には合いません。そこで、圧力とカーネルを
どちらも格子の 2 倍にゼロパディングし、カーネルをラップアラウンドの順に並べます。こうすると、
巡回畳み込みが元の領域の上では線形（自由空間）の畳み込みと一致します：

```mermaid
flowchart LR
    subgraph real["実空間（一様格子）"]
        P["圧力 p"]
        U["変位 u = K * p"]
    end
    subgraph freq["周波数空間"]
        PH["p̂"]
        KH["K̂（カーネル・事前計算）"]
        UH["û = K̂ · p̂"]
    end
    P -->|"2倍にゼロパディング + 実FFT"| PH
    PH --> UH
    KH --> UH
    UH -->|"逆FFT + 有効領域をクロップ"| U
```

> 一様格子は欠かせません。非一様格子では畳み込みの構造が崩れ、FFT による
> 高速化も成り立たなくなるからです。

### 設計方針

単一の接触をとにかく速く解くことよりも、任意形状や表面粗さ、マルチボディ接触へ
広げられることを優先しています。

### 検証ロードマップ

ソルバは、解析解を持つ問題から順に積み上げています。一致の度合いや検証方法の詳細は
[検証ドキュメント](docs/verification.md)にまとめています。

1. **円形接触**（球–平面 / 球–球）。解析的な Hertz 解で検証します。
2. **楕円接触**（トーラス外赤道上の球、凸–凸）。非軸対称の計算経路をひととおり確認します。
3. **任意の高さ場形状と粗さ**。半空間近似の範囲で、格子上に与えた任意のギャップに
   加算的な粗さ層を重ねます。Sneddon のコーン（解析解・非 Hertz）、独立な密行列ソルバ、
   Tamaas と相互検証します。

```mermaid
flowchart LR
    P1["P1 · 円形 Hertz<br/>球–平面 / 球–球"]
    P2["P2 · 楕円 Hertz<br/>トーラス外赤道上の球"]
    P4["P4 · 任意形状 + 加算的粗さ"]
    P1 -->|"解析 Hertz で検証"| P2
    P2 -->|"完全楕円積分 + 解析 Hertz で検証"| P4
    P4 --> C1["Sneddon コーン<br/>解析・非 Hertz"]
    P4 --> C2["密行列 Gauss–Seidel<br/>独立アルゴリズム"]
    P4 --> C3["Tamaas 自由空間<br/>外部 BEM コード"]
```

### v1 のスコープ外

摩擦・接線接触、弾塑性・粘弾性、コーティング、凝着（JKR/Maugis）、強保形接触、
GPU 実行。これらは v1 では実装しません。ただし、後から差し込めるように、アーキテクチャ側に
トレイト境界だけは用意しています。

### 先行研究

近いライブラリのなかで最も成熟しているのは [Tamaas](https://gitlab.com/tamaas/tamaas)
（EPFL、C++/Python、FFTW + OpenMP）ですが、こちらは既定で周期境界を前提にします。本
プロジェクトは、ネイティブな `pip` ホイールとして配布できる Rust + PyO3 実装である点が
違います。なお Tamaas は非周期の演算子も持っているので、P4 ではこれを自由空間での相互検証の
基準として使います（[検証ドキュメント](docs/verification.md#相互検証)を参照）。

---

## 技術スタック

| レイヤ              | ツール                                                         |
| ------------------- | ------------------------------------------------------------- |
| 数値コア            | Rust — `ndarray`, `rustfft` / `realfft`, `rayon`              |
| Python バインディング | `PyO3` + `maturin` + `rust-numpy`（ゼロコピー NumPy 連携）      |
| Python 環境 / 開発   | [`uv`](https://docs.astral.sh/uv/)（必須。生の Python は使えません） |
| 静的解析            | `ruff`（lint+format）、`mypy --strict`、`clippy -D warnings`    |

ソルバは「関数的なコアと命令的なシェル」という構成で、ジオメトリ（`Gap`）と弾性応答
（`InfluenceOperator`）はトレイト境界の裏に隠しています。新しい形状やカーネル
（周期・層状など）は、ソルバ本体に手を入れず、impl を 1 つ追加するだけで差し込めます。

```mermaid
flowchart TB
    subgraph py["Python 層"]
        API["hertzian パッケージ<br/>solve_sphere_on_flat / _on_torus / _in_gothic_arch / solve_height_field"]
        LAW["hertzian.GothicArchLaw<br/>縮約接触則 F(δ)"]
    end
    subgraph bind["バインディング層"]
        PYO3["PyO3 + maturin + rust-numpy<br/>ゼロコピー NumPy・GIL 解放・単一 abi3 ホイール"]
    end
    subgraph core["Rust コア"]
        SOLVER["Bccg ソルバ<br/>Polonsky–Keer BCCG"]
        INFL["InfluenceOperator<br/>FreeSpaceBoussinesq（DC-FFT）"]
        GAP["Gap トレイト<br/>Paraboloid / Torus / Cone / Waviness / HeightField"]
        REF["DenseReference<br/>独立な密行列・射影 Gauss–Seidel"]
    end
    API --> PYO3
    LAW --> PYO3
    PYO3 --> SOLVER
    SOLVER --> INFL
    SOLVER --> GAP
    SOLVER -. 相互検証 .-> REF
```

---

## 使い方（Python）

```python
import numpy as np
import hertzian

# 解析解による簡易版：円形 Hertz（平面上の球）。`domain` は界面格子（原点中心の
# 正方形）の物理的な一辺の長さ（メートル）。
sol = hertzian.solve_sphere_on_flat(
    radius=10e-3, load=50.0, e_star=70e9, grid=(256, 256), domain=1.2e-3
)
print(sol.contact_radius, sol.max_pressure, sol.approach)
print(sol.diagnostics)            # 反復回数、残差、収束フラグ
pressure = sol.pressure           # (nx, ny) float64 NumPy 配列（軸 0 = x）

# 楕円 Hertz：トーラス外赤道上の球（凸–凸、P2）。
sol = hertzian.solve_sphere_on_torus(
    sphere_radius=12e-3, tube_radius=4e-3, centre_radius=20e-3,
    load=60.0, e_star=100e9, grid=(256, 256), domain=1.2e-3,
)
print(sol.contact_half_widths, sol.ellipticity)

# 応用例：ゴシックアーチ（尖頭）軸受溝に押し込まれた玉。2つの円弧（2トーラス）を
# 重ねた形で、保形度は r/Rs = 1.04。centre_offset を非ゼロにする（円弧中心のシム）と
# 玉は2つのフランクに乗り、接触が2つに分裂する。centre_offset=0 なら単一の保形楕円
# 接触に戻る。ドメインは分裂方向（子午線 y 軸）に沿って縦長にとる。
sol = hertzian.solve_sphere_in_gothic_arch(
    sphere_radius=4e-3, tube_radius=4.16e-3, centre_radius=15e-3,
    centre_offset=65e-6, load=800.0, e_star=100e9,
    grid=(96, 846), domain=(0.65e-3, 5.74e-3),
)
print(sol.max_pressure)  # 2つのフランクパッチ、各々 P/2 での楕円 Hertz 接触

# 汎用エントリポイント（P4）：任意の未変形ギャップ高さ場 h(x, y)。形状は自由で、
# 必要なら粗さを上乗せできる。中心揃えの一様格子上でギャップを組み立て、ソルバへ渡す。
nx, ny = 256, 256
dx = dy = 1.2e-3 / nx
x = (np.arange(nx) - (nx - 1) / 2) * dx
y = (np.arange(ny) - (ny - 1) / 2) * dy
sphere = (x[:, None] ** 2 + y[None, :] ** 2) / (2 * 10e-3)          # 滑らかなベース
roughness = (                                                       # 加算的なうねり
    0.2e-6
    * np.cos(2 * np.pi * x[:, None] / 1e-4)
    * np.cos(2 * np.pi * y[None, :] / 1e-4)
)
sol = hertzian.solve_height_field(
    gap=np.ascontiguousarray(sphere + roughness), load=50.0, e_star=70e9, dx=dx, dy=dy
)
print(sol.contact_area, sol.max_pressure)
```

`e_star` は等価弾性係数 $E^*$ で、$\dfrac{1}{E^*} = \dfrac{1-\nu_1^2}{E_1} + \dfrac{1-\nu_2^2}{E_2}$
です。ソルバは GIL を解放して動くので、Python のスレッドから並列に呼び出せます。
v1 で実装しているのは自由空間境界だけです。`boundary="periodic"` は名前だけ予約してあり、
呼ぶと `NotImplementedError` を送出します。

---

## ギャラリー / 可視化

ソルバがいま解いている問題を、収束した圧力場で示します。各図は、左が圧力場、右が
解析解との比較です。定量的な一致（一致率や検証方法）は
[検証ドキュメント](docs/verification.md)にまとめています。

### 円形 Hertz — 平面上の球（P1）

![円形 Hertz 接触：圧力場が解析的な接触円を満たし、セルごとの半径方向圧力が Hertz 楕円体に重なる。](docs/img/circular.png)

軸対称のベンチマークです。圧力場（左）は解析的な接触円（破線）を満たします。すべての格子
セルの圧力を $r/a$ に対してプロットすると（右）、場全体が Hertz 楕円体に重なります：

$$p(r) = p_0\sqrt{1 - (r/a)^2}$$

### 楕円 Hertz — トーラス外赤道上の球（P2）

![楕円 Hertz 接触：解析的な接触楕円に一致する細長い圧力楕円。主軸方向の断面が解析プロファイルに乗る。](docs/img/elliptic.png)

凸–凸の接触は楕円になり、周方向（$x$）が子午線方向（$y$）より長くなります。求まった
接触域は解析的な接触楕円（破線）をなぞり、各主軸に沿った断面は解析的な半楕円体プロファイルに
乗ります：

$$p(x, y) = p_0\sqrt{1 - (x/a_x)^2 - (y/a_y)^2}$$

離心率 $e$ は、完全楕円積分 $K(e),\ E(e)$ で解いた曲率関係
$\dfrac{E/(1-e^2) - K}{K - E} = \dfrac{R_x}{R_y}$ から決まります。

### Sneddon のコーン — 非 Hertz・尖点特異の圧子（P4）

![Sneddon コーン接触：鋭く尖った圧力場。半径方向プロファイルは Sneddon の arccosh 則と対数的な尖点特異性に従う。](docs/img/cone.png)

任意（非放物面）のギャップ $h = m\,r$ を、測定した表面と同じ「高さ場」の経路で処理します。
Hertz と違って、圧力は尖点で対数的に発散します。半径方向のプロファイルは Sneddon の閉形式に
従います：

$$a = \sqrt{\frac{2P}{\pi E^* m}}, \qquad \delta = \frac{\pi}{2}\,m\,a, \qquad p(r) = \frac{E^* m}{2}\operatorname{arccosh}\!\frac{a}{r}.$$

### 粗面接触 — 球＋粗さ、分裂（P4）

![粗面接触：滑らかな単一 Hertz パッチの隣に、高圧の突起の格子へと分裂した球＋粗さの接触。](docs/img/roughness.png)

滑らかな球に余弦状の粗さ $h_r = A\cos(2\pi x/\lambda_x)\cos(2\pi y/\lambda_y)$ を重ねる
（高さ場の足し算）と、単一の Hertz 円板が突起接触の格子へと分裂します。同じ荷重の
もとでも、実接触面積は減り、ピーク圧は上がります。これは粗面接触に特有の現れ方です。粗い
パッチは閉形式を持たないので、独立な密行列ソルバと Tamaas で相互検証します
（[検証ドキュメント](docs/verification.md#相互検証)）。

> ギャラリーは `make gallery`（または
> `uv run --with matplotlib python scripts/render_gallery.py`）で再生成します。matplotlib は
> 描画のためだけに使う依存なので、ロック環境からは意図的に外しています。

---

## 応用例 — ゴシックアーチ軸受溝

ボールベアリングの軌道溝は、単一の円弧ではなく、中心をずらした2つの円弧（2トーラスを
重ねた凹面）として研削されることが多く、先のとがったゴシックアーチ形になります。
玉は溝の底ではなく2つのフランクに乗り、接触は2点に分裂します。これは新しいソルバ
機能ではなく、検証済みの楕円接触（基本要素）を応用したもので、$r/R_s = 1.04$（玉径に
対して溝半径が 52 % という教科書的な保形度）の保形接触です。

![ゴシックアーチ溝：接触圧力場が y = ±y0 の2つの楕円フランクパッチに分裂し、その間に非接触のゴシック点。子午線断面では、ソルバの2つのフランクピークが、単一アーチ全荷重ピークより下にある解析的な半荷重楕円 Hertz の半楕円に乗る。](docs/img/gothic.png)

溝のギャップは二重井戸型になります。中心をずらした2つの楕円放物面のうち、各点で玉に最も
近い面が選ばれます（点ごとの最小）：

$$h(x, y) = \frac{x^2}{2 R_x} + \frac{(|y| - y_0)^2}{2 R_y}.$$

```mermaid
flowchart LR
    A["パラボロイド井戸<br/>中心 y = +y₀"] --> M["点ごとの最小<br/>玉に近い面が勝つ"]
    B["パラボロイド井戸<br/>中心 y = −y₀"] --> M
    M --> G["二重井戸ギャップ h(x, y)<br/>中央にゴシック点・非接触リッジ"]
    G --> S["荷重で2つの楕円フランク接触に分裂<br/>各フランクが P/2 を担う"]
```

子午線半径は保形で $R_y = \dfrac{1}{1/R_s - 1/r}$（凹溝）、周方向半径は凸で
$R_x = \dfrac{1}{1/R_s + 1/R_0}$、フランクオフセットは
$y_0 = \texttt{centre\_offset}\cdot\dfrac{R_s}{r - R_s}$ です。円弧中心のわずかなシムは
保形度によって約 25 倍に拡大され、65 µm のシムでフランクが ±1.6 mm 離れます。
同じ全荷重でも、単一アーチ（$\texttt{centre\_offset} = 0$）なら1つの楕円パッチになりますが、
ゴシックのシムはそれを2つに分裂させます。

この分裂は荷重を保存するので、それ自体が検証にもなります。各フランクは荷重の半分を担う
楕円 Hertz 接触なので、そのピークは P2 ベンチマークと同じ閉形式に乗ります。具体的な
一致は[検証ドキュメント](docs/verification.md#ゴシックアーチ溝の検証)にまとめています。

### シムの調整 — 半分だけ重なる2つのフランク

同じ溝のままシムを詰めると、2つのフランク接触は離れたままではなく、接触楕円が
半分ずつ重なり合うところまで近づきます。設計上のねらいは、子午線方向のフランク
オフセットを $y_0 = b/2$ にすることです（$b$ は半荷重の孤立フランク楕円の子午線半軸）。
半軸 $b$ の2つの楕円は、中心が $b$ だけ離れていれば互いに半分ずつを共有します。重なりが
ゴシック点を埋めるので、接触は一続き（連結）になります。これは、分離したアーチの非接触
リッジとは対照的です。$|y|$ の折り返しはそのままなので、左右対称も保たれます。

![半分重なるゴシックアーチ溝：2つのフランク接触楕円（中心 y = ±y0 = ±b/2）が半分重なって描かれた、ひと続きの連結した圧力パッチ。子午線断面では、ソルバの2つのピークが接触状態のサドルでつながり、網掛けの重なり帯を通って孤立半荷重フランクの半楕円より上を走り、単一アーチ全荷重ピークの下で頭打ちになる。](docs/img/gothic_overlap.png)

重なり領域には閉形式がありません。2つのフランクが弾性場を通じて相互作用し、荷重が
もうきれいには $P/2$ ずつに分かれないからです。解析的な基準がないので、検証は P4 方式で
行います。すなわち、同じ格子上の独立な密行列・射影 Gauss–Seidel 参照解との相互検証です
（[検証ドキュメント](docs/verification.md#半分重なるフランクの検証)）。

---

## 縮約接触則 — マルチボディ動力学のための軽量 `F(δ)`

単一の滑らかな Hertz 接触は、$F = k\,\delta^{3/2}$ と力の向き $\mathbf{e}\parallel(\mathbf{x}-\mathbf{o})$
という2入力2出力の代数式で表せます。しかし、ゴシックアーチ溝のように玉が2つのフランクに
乗る複雑な形状では、もう単一の代数式には収まりません。とはいえ、マルチボディ動力学の
ような繰り返し計算では、毎ステップ FFT ソルバを回す余裕はありません。そこで、検証済みの
場のソルバを軽量な力則 $F(\delta)$ に落とし込み、形状に合わせて回帰でフィッティングします。

### モデル

溝の子午断面では、玉中心の変位 $\boldsymbol{\delta} = (\delta_t, \delta_n)$（横断 $\hat{t}$・
法線 $\hat{n}$）が、接触半角 $\pm\alpha$ だけ傾いた2つのフランク法線
$\hat{n}_\pm = (\pm\sin\alpha,\ \cos\alpha)$ の方向に各フランクを押し込みます。各フランクは
Hertz 荷重を担い（引張は受けないので正の部分 $\lfloor\cdot\rfloor_+$ のみ）、合力はその
ベクトル和になります：

$$
\begin{aligned}
s_\pm &= \boldsymbol{\delta}\cdot\hat{n}_\pm = \delta_n\cos\alpha \pm \delta_t\sin\alpha, \\[2pt]
Q_\pm &= K\,\lfloor s_\pm\rfloor_+^{3/2}, \\[4pt]
F(\boldsymbol{\delta}) &= Q_+\,\hat{n}_+ + Q_-\,\hat{n}_-
  \;\Longrightarrow\;
  F_t = (Q_+ - Q_-)\sin\alpha,\quad
  F_n = (Q_+ + Q_-)\cos\alpha.
\end{aligned}
$$

式は上から順に、各フランクの接近量 $s_\pm$（フランク法線 $\hat{n}_\pm$ への射影）、各フランクの
Hertz 荷重 $Q_\pm$、そして合力 $F$ とその横断・法線成分 $(F_t, F_n)$ です。
$K$ は1フランクの楕円 Hertz 荷重–変位定数、$\alpha$ は幾何学的な接触角（ここでは
$\alpha \approx 24^\circ$）です。これは、単一 Hertz 接触の $F = k\,\delta^{3/2}$ を2フランクに
重ね合わせた、2入力2出力の閉形式そのものです。

### 境界条件 — 2溝→1溝の微分連続（`C¹`）

荷重が傾くと内側フランクが除荷し、$\delta_t = \delta_n\cot\alpha$ で離れます。接触は
2フランクから1フランクへ移り、力則は単一 Hertz 接触 $F = K\,s_+^{3/2}\,\hat{n}_+$
（先に示した1溝の $F = k\,\delta^{3/2}$、$\mathbf{e}\parallel(\mathbf{x}-\mathbf{o})$）に
帰着します。

```mermaid
stateDiagram-v2
    direction LR
    [*] --> Separated
    Separated --> TwoFlank: 押し込み δn 増
    TwoFlank --> OneFlank: δt が δn·cotα を超え内側フランク離反
    OneFlank --> TwoFlank: δt 減で内側フランク再接触
    TwoFlank --> Separated: 引き抜き δn 負
    OneFlank --> Separated: 引き抜き δn 負
    Separated: 分離 — 荷重ゼロ・引張なし
    TwoFlank: 二フランク接触 — 2 つの楕円 Hertz の重ね合わせ
    OneFlank: 単一フランク接触 — 単一 Hertz 接触に帰着
    note right of OneFlank: 二フランクから単一への遷移は C¹（力も接線剛性も連続）
```

この遷移は `C¹` です。力とそのヤコビアン（接線剛性）がどちらも連続になります。理由は、
Hertz 指数が $3/2 > 1$ だからです。除荷するフランクは、荷重も剛性もゼロのまま
なめらかに合流します。$Q_- \propto s_-^{3/2}\to 0$ かつ $\mathrm{d}Q_-/\mathrm{d}s_- \propto s_-^{1/2}\to 0$
となるからです。この 1.5 乗こそが、2→1 のなめらかな切り替えを保証する仕組みです。ただし `C²` では
ありません。接線剛性は $\sqrt{\ \cdot\ }$ のカスプ（$\mathrm{d}^2Q/\mathrm{d}s^2 \propto s^{-1/2}\to\infty$）を
持ちます。これは、Hertz 接触が無限の初期勾配で硬化するという、おなじみの特徴です。

解析的な接線剛性（カップリングを切った $\kappa = 0$ の形）は、各フランクの接線
$g_\pm = \tfrac{3}{2}K\,\lfloor s_\pm\rfloor_+^{1/2}$ を使って

$$
\frac{\mathrm{d}F}{\mathrm{d}\boldsymbol{\delta}} =
\begin{bmatrix}
(g_+ + g_-)\sin^2\alpha & (g_+ - g_-)\sin\alpha\cos\alpha \\
(g_+ - g_-)\sin\alpha\cos\alpha & (g_+ + g_-)\cos^2\alpha
\end{bmatrix}
$$

となります。フランクが除荷すると $g_\pm\to 0$ になるので、行列は離反の境目を越えても連続です
（つまり `C¹`）。一方その微分は $g_\pm\propto\sqrt{s_\pm}$ で発散するので、`C²` ではありません。

### 隣のフランクが相手を持ち上げる — 一次の弾性カップリング

2フランクを独立な Hertz 接触として重ね合わせるのが厳密になるのは、両者が十分に離れている
ときだけです。分離極限では各フランクが半荷重を担い、有効フランク数 $\eta = P/(K\,\delta^{3/2})$ は
$2$ になります。シムを詰めて2つのフランク接触が近づくと弾性場が重なり、一方のフランクの荷重
$Q$ がもう一方の真下の半空間を持ち上げて、隣の接近量、ひいては荷重を削ります。一次の
近似では、各フランクは相手の Boussinesq 遠方場を受けます。すなわち、距離 $d = 2 y_0$
（フランク中心は $y = \pm y_0$）にある点荷重 $Q$ です：

$$u \approx \frac{Q}{\pi E^* d}, \qquad d = 2 y_0.$$

そこで、2つの実効接近量は互いの荷重を通じて連立します：

$$s_\pm^{\text{eff}} = s_\pm - \kappa\,Q_\mp, \qquad Q_\pm = K\,\lfloor s_\pm^{\text{eff}}\rfloor_+^{3/2}, \qquad \kappa = \frac{1}{2\pi E^* y_0}.$$

小さな $2\times 2$ の自己無撞着な解です（`with_flank_coupling` で有効化）。閉形式の安さも、
解析ヤコビアンも、`C¹` もそのままです。除荷するフランクは荷重も剛性も持ち上げも
ゼロのまま合流し、$y_0 \to \infty$ では $\kappa \to 0$ になって分離極限（$\eta = 2$）に戻ります。
これは、検証済みの基本要素に一次の相互作用項を足したもので、縮約則の適用範囲を
「十分に分離」から「半重なり」まで広げます。

このリフトは荷重の大きさの補正で、フランクの接近量（`coupled_loads`）の段階で効きます。
向きについては、もう一段細かい二次の効果が残ります。重なると荷重重心が幾何オフセット $y_0$ の
外側へずれ、フランク法線がわずかに回転するという効果です。ただしこれは $(F_t, F_n)$ の射影だけを
直すもので、本節で扱う $\eta$ や荷重分割には効きません。そのため、完全合体（$\eta \to 1$、
単一アーチへのなめらかな接続）とあわせて次の段階に回します。

横から見た断面で厳密解（場ソルバ）と近似解（縮約則）を3つの状態で並べた図と、$\eta$ の
較正・一致の数値は、[検証ドキュメント](docs/verification.md#縮約接触則の較正と検証)に
まとめています。

```python
import hertzian

# フランク形状から法則を一度だけ較正する（実行時に FFT ソルブなし）。
law = hertzian.GothicArchLaw.from_elliptic_flank(
    radius_x=3.31e-3,  # 1フランクの周方向の相対半径
    radius_y=0.104,  # 子午線方向の（保形）相対半径
    e_star=100e9,
    contact_angle=hertzian.contact_half_angle(offset=1.6e-3, ball_radius=4e-3),
)

# 半重なりまで使うなら、隣のフランクの持ち上げ（一次カップリング）を有効化する：
law = law.with_flank_coupling(e_star=100e9, offset=1.6e-3)  # κ = 1/(2π E* y0)

# あとはマルチボディの内側ループで F(δ) と接線剛性を評価する：
f_t, f_n = law.force(2e-6, 6e-6)  # 接触力ベクトル (N)
stiffness = law.jacobian(2e-6, 6e-6)  # 2x2 接線剛性 dF/dδ (N/m)

# クーロン摩擦には合力 F だけでなく面圧分布が要る：各フランクの（カップリング後の）
# 荷重から、楕円 Hertz 半楕円体のキャップ p(x, y) を立方根スケールで得る（FFT 不要）。
q_plus, q_minus = law.flank_loads(2e-6, 6e-6)  # カップリング込みのフランク荷重 (N)
cap = law.flank_pressure(q_plus)  # 1フランク分の面圧分布
p0 = cap.peak_pressure  # ピーク圧 p0 = 3Q/(2π a_x a_y) (Pa)
tau_max = cap.traction_bound(0.12, 0.0, 0.0)  # 局所クーロンキャップ μ p(x, y) (Pa)

# 両フランクをまとめた溝全体のキャップは、2つの半楕円体の「包絡」＝点ごとの最大
# （素朴な和ではない）。重なっても継ぎ目を二重計上せず、分離時は和と一致する。
groove = law.groove_pressure(q_plus, q_minus, offset=1.6e-3)  # 溝全体の面圧キャップ
tau_groove = groove.traction_bound(0.12, 0.0, 0.0)  # 局所クーロンキャップ μ p(x, y)
```

### 較正パイプライン — 形状から係数を自動取得する（`calibrate`）

上の例では、フランクの相対半径・オフセット・接触角を手で求めて法則に渡していました。
`hertzian.calibrate` は、この還元を自動化します。Rust 側は係数 $(K, \alpha, \kappa)$ を後から
差し込むだけの純粋関数 $F(\delta)$ で、Python 側のパイプラインが物理形状（玉・溝・基準中心の
各半径、アーチのシム、弾性係数）から係数を導いて差し込みます。数行で、すぐ使える法則が
得られます：

```python
import hertzian

# 物理的な溝の仕様（SI 単位）を一度書くだけ。
spec = hertzian.GrooveSpec(
    ball_radius=4.0e-3,  # 玉半径 R_s
    tube_radius=4.16e-3,  # 溝（チューブ）半径 r （r/R_s = 1.04）
    centre_radius=15.0e-3,  # 基準中心円半径 R_0
    centre_offset=65.0e-6,  # アーチのシム（微小オフセット）
    e_star=100e9,  # 等価弾性係数 E*
)

cal = hertzian.calibrate(spec)  # 形状 → 係数（既定で場ソルバ検証つき）
law = cal.law  # すぐ使える GothicArchLaw
f_t, f_n = law.force(1e-6, 6e-6)  # マルチボディ内側ループでそのまま評価
print(cal.describe())  # 係数・精度・速度の確認レポート
```

`calibrate` は既定で、蒸留元である FFT+BCCG 場ソルバに対して結果を自己検証します。単一アーチの
荷重スイープで Hertz の 1.5 乗則と剛性 $K$ を確認し、二フランクの一点で結合後の有効フランク数 $\eta$ を
照合し、力評価の速度を測ります。`describe()` は、それを一目で読める形にまとめます：

```text
hertzian: reduced Gothic-arch law calibration
================================================
groove (input):
  R_s   (ball)      = 4.0000e-03 m
  r     (tube)      = 4.1600e-03 m   (r/R_s = 1.040)
  ...
reduced flank (derived):
  R_x  (circumf.)   = 3.1579e-03 m
  R_y  (meridional) = 1.0400e-01 m
  y0   (flank off.) = 1.6250e-03 m
  alpha (contact)   = 23.97 deg
coefficients (inserted into the Rust law):
  K     (stiffness) = 2.24314e+10 N/m^1.5
  kappa (coupling)  = 9.79415e-10 m/N
verification (FFT+BCCG field solver):
  exponent m        = 1.5000   (Hertz 1.5)
  fit R^2           = 1.000000
  K (solver fit)    = 2.24695e+10 N/m^1.5  (x1.002 vs analytic)
  max load residual = 0.17 %
  eta (flank count) = 1.919 solver / 1.920 law  (0.1 %)
speed:
  field solve       = ~350 ms
  reduced force()   = ~0.9 µs
  speed-up          = ~4e+05 x
```

ここで、精度（指数 1.5、$R^2 \approx 1$、残差 0.2 % 未満、$\eta$ の一致 0.1 %）と速度（場ソルバ比で
約 10⁵ 倍）が両立していることが確認できます。ソルブを走らせず解析係数だけをすぐに得たいときは、
`hertzian.calibrate(spec, verify=False)` とします（`cal.verification` は `None` になります）。実行できる一式は
`scripts/calibrate_gothic_law.py` にあり、`uv run python scripts/calibrate_gothic_law.py` で動きます。

### 面圧分布 — クーロン摩擦のための軽量キャップ

ここまでの $F(\boldsymbol{\delta})$ は合力です。しかしクーロン摩擦は局所的で、接線トラクションは
各点で面圧によって $|\tau(x,y)| \le \mu\,p(x,y)$ と上限を課されます。つまり、合力だけでは足りず、
面圧分布 $p(x,y)$ そのものが必要です。各フランクは（カップリング後の）荷重 $Q_\pm$ を担う楕円
Hertz 接触なので、その面圧はおなじみの半楕円体になります。これを各フランク中心 $\pm y_0$ に置きます：

$$
p(x,y) = p_0\sqrt{\left\lfloor 1 - \Bigl(\tfrac{x}{a_x}\Bigr)^2 - \Bigl(\tfrac{y\mp y_0}{a_y}\Bigr)^2\right\rfloor_+},
\qquad |\tau(x,y)| \le \mu\,p(x,y).
$$

形状は一度だけ、$K$ を較正したのと同じフランクから決まります。Hertz の荷重スケール則で
接触半軸は $a = \hat{a}\,Q^{1/3}$ になるので、ピーク圧も立方根でスケールします：

$$
p_0 = \frac{3Q}{2\pi a_x a_y} = c_p\,Q^{1/3},\qquad c_p = \frac{3}{2\pi\,\hat{a}_x \hat{a}_y},
$$

`flank_pressure(Q)` は `cbrt` 数回で済み、内側ループに離心率の超越方程式は出てきません。$Q = K\,s^{3/2}$
なので $p_0 = c_p K^{1/3}\sqrt{s}$ となり、面圧の上限は離反点で $\sqrt{s}$ としてゼロに接しながら消えます。力を `C¹` に
する 1.5 乗の特徴が、ここにも現れます。半楕円体は $Q$ ちょうどに積分される（$\iint p\,\mathrm{d}A = Q$）ので、
全滑り時の摩擦合力は1フランクあたり $\iint \mu\,p\,\mathrm{d}A = \mu Q$ になります。

**両フランクの合成は、和ではなく包絡。** 1フランクのキャップは、フランクが分離していれば厳密です。
しかし、マルチボディ接触で必要になるのは両フランクをまとめた溝全体のキャップです。これを2つの
半楕円体の単純な和で作ると、重なりで破綻します。足し算が正しいのは、2つの footprint が
重ならない間（各点でせいぜい一方のフランクだけが接触する間）だけです。シムを詰めて半分重なると、
和は重なりを二重計上し、継ぎ目を非物理的なスパイクに積み上げます。

正しい軽量合成は、点ごとの最大、すなわち包絡です。これは、溝ギャップの作り方のちょうど裏返しに
なっています。ギャップは2つのフランク井戸の点ごとの最小 $h = \min(\text{well}_+, \text{well}_-)$
（近い面が勝つ、`GothicArchProfile`）でした。これに対応して、面圧キャップは2つの footprint の
点ごとの最大で、各点でより深く押されたフランクが面圧を決めます：

$$
p(x,y) = \max\!\bigl(p_+(x,\, y - y_0),\ p_-(x,\, y + y_0)\bigr).
$$

`groove_pressure(...)` が、この $p(x,y)$ を返します（`GrooveContactPressure`）。footprint が分離している
ところでは、包絡は和と完全に一致します。重なるところでは二重計上を落として、場ソルバのサドルで
つながった連結パッチを取り戻します。半分重なるところ（$y_0 = b/2$）が、まさに本タスクの比較点です。
厳密解（場ソルバ）と軽量式を並べた検証は、[検証ドキュメント](docs/verification.md#面圧キャップの検証)に
まとめています。そこでは、素朴な和が継ぎ目のスパイクでピークを過大評価する一方、包絡は厳密解に
ほぼ乗りサドルも再現することを示しています。

これは一次のキャップです。包絡は重なりレンズ $\iint \min(p_+, p_-)$ を捨てるので、重なるところでは積分が
$Q_+ + Q_-$ をわずかに下回ります（各フランク荷重そのものは厳密なので、全滑り合力は荷重から
$\mu (Q_+ + Q_-)$ のままです）。このレンズを配り直すこと、すなわち単一アーチへの単一パッチ合体は、
$\eta \to 1$ のブレンドと同じ次の段階に回します（合体すれば1つの接触が $2Q$ を担い、より深いピーク
$c_p (2Q)^{1/3}$ になります）。

**非対称ドライブ — フランクが干渉する 2トーラスで 2:1 の面圧ピーク。** ここまでは、玉を溝へ
まっすぐ押し込む対称な押し込みで、2つのフランクが荷重を等分し、面圧ピークは 1:1 でした。
けれども、クーロン摩擦が効くのは玉が溝を横切って引きずられるときです。横断ドライブが荷重を
片方のフランクへ寄せ、2つの面圧ピークは開きます。本タスクでは形状はそのままに、横断ドライブだけを
与えて面圧ピークが 2:1 になるところまで寄せます。配置はフランクが干渉するもの、すなわち半重なりの
シム $y_0 = b/2$ です（2つの footprint が交差してひと続きの連結パッチになる、フランク干渉の基本配置で、
$y = 0$ のゴシック点も荷重を担います）。場ソルバでは横断変位を、子午線方向の井戸床オフセット $df$
として与えます（2つのフランク井戸の片方を $df$ だけ持ち上げ、近い側のフランクをそれだけ深く押す、玉の
横変位を高さ場で裏返したものです。ギャップは2トーラスの点ごとの最小のままです）。各フランクは依然として
荷重 $Q$ を担う楕円 Hertz パッチで、ピークは立方根則 $p_0 = c_p Q^{1/3}$ に乗るので、2:1 のピーク比は
ちょうど 8:1 の荷重分割になります（深い側が約 8 倍）。重なるので、和ではなく包絡が要ります。素朴な和は
重なりを二重計上して継ぎ目を非物理的なスパイクに積み上げますが、包絡 $\max(p_+, p_-)$ は2つの 2:1
クレストと、それをつなぐ荷重を担う連結サドルを取り戻します。場ソルバ（厳密）と軽量式を並べると、
非対称な 2:1 でも包絡はピークを数 % 以内で追い、連結サドルを再現します（一方、素朴な和はサドルで
大きく過大評価します）。詳しい一致は
[検証ドキュメント](docs/verification.md#非対称な-21-面圧キャップの検証)にまとめています。

---

## 開発

### 前提環境

- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) — 本プロジェクトで
  Python を動かす、唯一サポートされた方法です（後述の「生の Python 禁止」を参照）。
- `rustup` で入れる Rust ツールチェイン。正確なバージョン（`clippy` と `rustfmt` を含む）は
  [`rust-toolchain.toml`](./rust-toolchain.toml) に固定してあり、最初に `cargo` や
  `rustup show` を実行したときに自動でインストールされます。

### クイックスタート

```sh
make setup    # uv sync ＋ git フック ＋ Rust ツールチェインをインストール
make build    # ネイティブ拡張を uv venv にビルド（maturin develop）
make test     # cargo test ＋ pytest
make lint     # CI と全く同じ静的解析をすべて実行（pre-commit）
make fmt      # Python（ruff）と Rust（cargo fmt）を自動整形
make help     # 全ターゲットを一覧表示
```

> `make` は便利のためのラッパにすぎません。正式なチェックは
> [`.pre-commit-config.yaml`](./.pre-commit-config.yaml) にあり、CI も同じフックを実行します。
> つまり、`make lint` がローカルで通れば、CI の静的解析ジョブも通ります。

### 生の Python 禁止

本プロジェクトでは、Python の直接呼び出しを禁止しています（`python …`、`pip …`、
`requirements.txt`、`setup.py`、conda など）。すべて `uv` を経由します：

```sh
uv run python ...     # ✅ `python ...` の代わり
uv run pytest         # ✅
uv add <pkg>          # ✅ `pip install <pkg>` の代わり
uvx <tool>            # ✅ 単発のツール
```

このルールは [`scripts/check-no-raw-python.sh`](./scripts/check-no-raw-python.sh) で
強制され、pre-commit と CI で実行されます。背景と詳しい理由は
[`CONTRIBUTING.md`](./CONTRIBUTING.md) にあります。

---

## ライセンス

[MIT](./LICENSE-MIT) または [Apache-2.0](./LICENSE-APACHE) のいずれかを選択できる
デュアルライセンスです。
