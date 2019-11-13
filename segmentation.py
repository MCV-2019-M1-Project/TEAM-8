import numpy as np
from scipy.ndimage.morphology import distance_transform_edt as bwdist
from PIL import Image

# Parameters

# mu, nu    : length, area (regularizer terms)
mu, nu = 0.2, 0

phantom_num = 17
image_name = f"00003"
image_ext = "jpg"
image_path = f"/home/malpunek/coding/m1/datasets/qsd1_w5/{image_name}.{image_ext}"

# lambda1, lambda2: data fidelity parameters
lambda1, lambda2 = 1, 1
# tol   : tolerance for the sopping criterium
tol = 0.1 / mu
# dt     : time step
dt = 0.5
# iterMax : MAximum number of iterations
iterMax = 20
# reIni   : Iterations for reinitialization. 0 means no reinitializacion
reIni = 0

save_path = f"try.gif"
max_size = 256 * 256
original_dim = None


def segment(
    img,
    phi,
    mu=0.5,
    nu=0,
    lambda1=1,
    lambda2=1,
    tolerance=0,
    iterMax=20,
    dt=0.5,
    reIni=0,
):

    pdiff = tolerance + 1
    nIter = 0
    animation = [phi.copy()]
    almost_zero = 10 ** -16

    # while pdiff > tol and nIter < iterMax:
    while nIter < iterMax:
        nIter = nIter + 1

        # Fixed phi, Minimization w.r.t c1 and c2 (constant estimation)
        c1 = img[phi >= 0].mean()
        c2 = img[phi < 0].mean()
        print(f"Iteration {nIter}\nColors {c1} {c2}")

        pdiff = 0

        max_j, max_i = phi.shape
        max_j, max_i = max_j - 1, max_i - 1

        for j in range(phi.shape[0]):
            j_up = j - 1 if j != 0 else 0
            j_down = j + 1 if j != max_j else max_j

            for i in range(phi.shape[1]):

                i_left = i - 1 if i != 0 else 0
                i_right = i + 1 if i != max_i else max_i

                px = phi[j, i]
                pl, pr = phi[j, i_left], phi[j, i_right]
                pu, pd = phi[j_up, i], phi[j_down, i]

                delta = dt / (np.pi * (1 + px ** 2))

                iDivR = mu / np.sqrt(
                    almost_zero + (pr - px) ** 2 + ((pd - pu) / 2) ** 2
                )
                iDivL = mu / np.sqrt(
                    almost_zero + (px - pl) ** 2 + ((pd - pu) / 2) ** 2
                )

                iDivD = mu / np.sqrt(
                    almost_zero + (pd - px) ** 2 + ((pr - pl) / 2) ** 2
                )
                iDivU = mu / np.sqrt(
                    almost_zero + (px - pu) ** 2 + ((pr - pl) / 2) ** 2
                )

                dist1 = (img[j, i] - c1) ** 2
                dist2 = (img[j, i] - c2) ** 2

                phi[j, i] = (
                    phi[j, i]
                    + delta
                    * (
                        pr * iDivR
                        + pl * iDivL
                        + pu * iDivU
                        + pd * iDivD
                        - nu
                        - lambda1 * dist1
                        + lambda2 * dist2
                    )
                ) / (1 + delta * (iDivR + iDivL + iDivD + iDivU))
                pdiff += (phi[j, i] - px) ** 2

        # Reinitialization of phi
        if reIni > 0 and nIter % reIni == 0:
            indGT = phi >= 0
            indLT = phi < 0

            phi = bwdist(indLT == 0) - bwdist(indGT == 0)
            # Normalization [-1 1]
            nor = max(abs(phi.min()), phi.max())
            phi = phi / nor

        # Difference. This stopping criterium has the problem that phi can
        # change, but not the zero level set, that it really is what we are
        # looking for.
        pdiff = np.sqrt(pdiff / phi.size)
        print(f"Difference {pdiff}")

        animation.append(phi.copy())

    return animation


def downscale(img):
    global original_dim
    scale = np.sqrt(max_size / (img.size[0] * img.size[1]))
    if scale > 1:
        return img
    original_dim = img.size[1], img.size[0]
    width = int(img.size[1] * scale)
    height = int(img.size[0] * scale)
    dim = (width, height)
    # resize image
    return img.resize(dim)


def get_image(path):
    img = Image.open(path)
    img = downscale(img)
    img.save("source.png")
    img = np.array(img.convert("L"), dtype=np.float64) / 255
    img = img - img.min()
    img = img / img.max()
    return img


img = get_image(image_path)

# Phi
ni, nj = img.shape
X, Y = np.meshgrid(np.arange(ni), np.arange(nj))

phi_0 = np.zeros_like(img)
for i in range(phi_0.shape[0]):
    for j in range(phi_0.shape[1]):
        phi_0[i, j] = np.sin(i * np.pi / 2) * np.sin(j * np.pi / 2)


def make_img(phi):
    img_m = Image.open(image_path)
    img_m = downscale(img_m)
    arr = np.array(img_m)
    if arr.ndim == 3:
        arr[phi <= 0, :] = 0
        arr[phi > 0, :] = 255
    else:
        arr[phi <= 0] = 0
        arr[phi > 0] = 255
    return Image.fromarray(arr)


def make_gif(phis, path="solution.gif"):
    animation = [make_img(phi) for phi in phis[::2]]
    animation[0].save(
        path,
        format="GIF",
        append_images=animation[1:],
        save_all=True,
        duration=200,
        loop=0,
    )
    make_img(phis[-1]).save(f"{save_path[:-4]}_last.png")
    print(f"Saved {len(animation)} frames!")


animation = segment(
    img,
    phi_0,
    mu=mu,
    nu=nu,
    lambda1=lambda1,
    lambda2=lambda2,
    tolerance=tol,
    iterMax=iterMax,
    dt=dt,
    reIni=reIni,
)
make_gif(animation, save_path)
