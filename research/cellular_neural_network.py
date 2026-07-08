"""
Cellular Neural Network (CNN) -- a locally-coupled analog processor array (Chua-Yang 1988).

A CNN is a 2-D grid of identical analog cells. Each cell (i,j) carries a real STATE x_ij that
evolves by the coupled ODE

    dx_ij/dt = -x_ij + sum_{kl in N(i,j)} A_kl * y_kl + sum_{kl in N(i,j)} B_kl * u_kl + z

over a fixed 3x3 neighborhood N. Here u is a FIXED input image, y_ij = f(x_ij) is the cell OUTPUT
under the piecewise-linear saturation

    f(v) = 0.5 * (|v+1| - |v-1|)        # == v on [-1,1], clamped to +-1 outside

A is the FEEDBACK template (couples a cell to its neighbors' outputs), B is the CONTROL template
(couples it to the neighbors' inputs), and z is a scalar bias. The machine is "programmed" purely
by the triple (A, B, z); the same silicon computes edge-detection, hole-filling, erosion, etc. by
swapping templates. Roska-Chua's (1993) CNN Universal Machine stacks stored templates + local
logic on top of exactly this base array; here we implement only the base CNN. On-thesis for this
repo as a neuromorphic ANALOG array / early-vision operator (compute at the sensor, no clock).

Determinism / honesty choice (why this port is bulletproof): a general CNN with off-center
feedback is a bistable relaxation network whose settled state can depend on the inter-cell
convergence order -- no clean closed form to assert against. We therefore restrict the FEEDBACK
template to a POSITIVE CENTER ONLY, A = [[0,0,0],[0,a0,0],[0,0,0]] with a0 = 1. With center-only
self-feedback each cell decouples: writing w_ij = (B*u)_ij + z (a CONSTANT, since u is fixed),

    dx/dt = -x + a0*f(x) + w      (a0 = 1)

In the linear band |x|<1 this is dx/dt = w (constant drift, sign fixed by w); once |x| crosses 1
the cell saturates to x* = sign(w)*(1+|w|), so the settled output is the per-cell CLOSED FORM

    y_ij(inf) = sign( (B*u)_ij + z )

independent of every neighbor's output and of integration order. No convergence race, zero RNG.

The classic clean template is EDGE DETECTION on a binary image (pixels in {-1=white, +1=black}):

    A = [[0,0,0],[0,1,0],[0,0,0]],  B = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],  z = -1

giving w_ij = 8*u_ij - (sum of the 8 neighbour inputs) - 1. A black pixel whose whole 3x3
neighbourhood is black nets w = 8 - 8 - 1 = -1 < 0 (interior -> erased to white); a black pixel
touching ANY white neighbour nets w > 0 (boundary -> kept black); every white pixel nets w < 0
(background stays white). So the settled+thresholded grid is exactly the black region's OUTLINE.
Boundary condition: fixed Dirichlet -- virtual cells outside the grid hold u = y = -1 (white
world). The rectangle in the test image never touches the grid border, so the outline is exact.

A second template (logical NOT / inversion), also center-only-A, gives w_ij = -u_ij and hence
the pixelwise complement -- a second independent closed-form assert.

Pure standard library (no imports needed), no randomness, ASCII-only output.

  python research/cellular_neural_network.py        # (also run by run_selfchecks.py)

Papers: Chua, L.O. & Yang, L. (1988), "Cellular Neural Networks: Theory", IEEE Trans. Circuits
and Systems 35(10):1257-1272. Roska, T. & Chua, L.O. (1993), "The CNN Universal Machine: An
Analogic Array Computer", IEEE Trans. Circuits and Systems II 40(3):163-173.
"""


def f(v):
    """Piecewise-linear saturation f(v) = 0.5*(|v+1| - |v-1|): identity on [-1,1], clamped to +-1."""
    return 0.5 * (abs(v + 1.0) - abs(v - 1.0))


def _conv3x3(grid, i, j, tmpl, boundary):
    """Sum tmpl[di+1][dj+1] * grid[i+di][j+dj] over the 3x3 neighbourhood; out-of-grid cells use
    the fixed Dirichlet `boundary` value. tmpl is a 3x3 nested list (row-major, center at [1][1])."""
    rows, cols = len(grid), len(grid[0])
    s = 0.0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            k = tmpl[di + 1][dj + 1]
            if k == 0:
                continue
            ii, jj = i + di, j + dj
            val = grid[ii][jj] if 0 <= ii < rows and 0 <= jj < cols else boundary
            s += k * val
    return s


def cnn_settle(u, A, B, z, dt=0.1, steps=200, boundary=-1.0):
    """Euler-integrate the CNN array to steady state and threshold the output.

    u       : input image, nested list of floats (typically in {-1,+1})
    A, B    : 3x3 feedback / control templates (nested lists)
    z       : scalar bias
    boundary: fixed Dirichlet value for virtual cells outside the grid (shared by u and y)
    returns : settled output BITMAP, nested list of 0/1  (1 where y>0, i.e. saturated black)
    """
    rows, cols = len(u), len(u[0])
    x = [[0.0] * cols for _ in range(rows)]                 # start from rest x=0
    for _ in range(steps):
        y = [[f(x[i][j]) for j in range(cols)] for i in range(rows)]
        nxt = [[0.0] * cols for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                fb = _conv3x3(y, i, j, A, boundary)         # (A * y)_ij  feedback
                ctrl = _conv3x3(u, i, j, B, boundary)       # (B * u)_ij  control
                dx = -x[i][j] + fb + ctrl + z
                nxt[i][j] = x[i][j] + dt * dx
        x = nxt
    return [[1 if f(x[i][j]) > 0.0 else 0 for j in range(cols)] for i in range(rows)]


def render(bits, on="#", off="."):
    """ASCII picture of a 0/1 bitmap."""
    return "\n".join("".join(on if b else off for b in row) for row in bits)


# ---- fixed test image: a solid 4x4 black block inset in a 6x6 white field (never touches border) --
IN_BITS = [
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0],
]

# EDGE DETECTION template (feedback = center-only positive, so settled output is a per-cell closed form)
A_EDGE = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
B_EDGE = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
Z_EDGE = -1.0

# INVERSION / logical-NOT template (w_ij = -u_ij -> pixelwise complement)
A_NOT = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
B_NOT = [[0, 0, 0], [0, -1, 0], [0, 0, 0]]
Z_NOT = 0.0

# Expected outputs HAND-DERIVED from sign((B*u)+z), NOT copied from a code run:
#   edge: only block-perimeter pixels net w>0; the 2x2 interior (rows2-3,cols2-3) nets w=-1 (erased);
#         all background stays white. -> the outline of the 4x4 block.
EXPECTED_EDGE = [
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0],
]
#   inversion: w = -u -> settled output is the pixelwise complement of the input.
EXPECTED_NOT = [[0 if b else 1 for b in row] for row in IN_BITS]


def main():
    u = [[1.0 if b else -1.0 for b in row] for row in IN_BITS]
    out_edge = cnn_settle(u, A_EDGE, B_EDGE, Z_EDGE)
    out_not = cnn_settle(u, A_NOT, B_NOT, Z_NOT)

    print("=" * 64)
    print("CELLULAR NEURAL NETWORK -- locally-coupled analog array (Chua-Yang 1988)")
    print("=" * 64)
    print("template: dx/dt = -x + A*y + B*u + z  |  f(v)=0.5*(|v+1|-|v-1|)  |  dt=0.1")
    print("feedback A center-only (a0=1) -> settled y_ij == sign((B*u)_ij + z), per-cell closed form")
    print("-" * 64)
    print("INPUT (6x6, solid 4x4 black block):")
    print(render(IN_BITS))
    print("-" * 64)
    print("EDGE-DETECT out  [A=center; B=[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]; z=-1]:")
    print(render(out_edge))
    print("-" * 64)
    print("INVERSION out    [A=center; B=[[0,0,0],[0,-1,0],[0,0,0]]; z=0]:")
    print(render(out_not))
    print("=" * 64)

    # ---- self-check (deterministic asserts; expected bitmaps hand-derived, nothing reverse-engineered) ----
    # 1. edge detection: settled+thresholded grid == the block outline computed by hand from sign((B*u)+z)
    assert out_edge == EXPECTED_EDGE, f"edge-detect mismatch:\n{render(out_edge)}"
    # 2. interior black pixel nets w = 8 - 8 - 1 = -1 < 0 -> erased to white (a specific closed-form check)
    interior = _conv3x3([[1.0 if b else -1.0 for b in r] for r in IN_BITS], 2, 2, B_EDGE, -1.0) + Z_EDGE
    assert interior == -1.0 and out_edge[2][2] == 0, f"interior (2,2) w={interior}"
    # 3. an edge black pixel (top-left block corner) nets w = 8 - (-2) - 1 = 9 > 0 -> kept black
    corner = _conv3x3([[1.0 if b else -1.0 for b in r] for r in IN_BITS], 1, 1, B_EDGE, -1.0) + Z_EDGE
    assert corner == 9.0 and out_edge[1][1] == 1, f"corner (1,1) w={corner}"
    # 4. inversion: settled grid == pixelwise complement of the input
    assert out_not == EXPECTED_NOT, f"inversion mismatch:\n{render(out_not)}"
    # 5. deterministic (zero RNG): a repeat run yields the identical bitmap
    assert cnn_settle(u, A_EDGE, B_EDGE, Z_EDGE) == out_edge, "non-deterministic edge run"
    print("self-check OK: center-only-A CNN settles to sign((B*u)+z); edge-detect == hand-computed "
          "block outline (interior w=-1 erased, corner w=9 kept); inversion == pixelwise NOT; deterministic")


if __name__ == "__main__":
    main()
