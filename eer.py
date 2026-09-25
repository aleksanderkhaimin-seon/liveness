import numpy as np

def get_fr_at_threshold(tar, threshold=0.5):
    fr = np.nan
    if len(tar) > 0:
        fr = len(np.where(tar < threshold)[0])
        fr = fr * 100.0 / len(tar)

    return fr


def get_fa_at_threshold(imp, threshold=0.5):
    fa = np.nan
    if len(imp) > 0:
        fa = len(np.where(imp > threshold)[0])
        fa = fa * 100.0 / len(imp)

    return fa


def get_fr_fa_at_threshold(tar, imp, threshold=0.5):
    fr = get_fr_at_threshold(tar, threshold=threshold)
    fa = get_fa_at_threshold(imp, threshold=threshold)

    return fr, fa

def compute_frr_far(tar, imp):

    tar_unique, tar_counts = np.unique(tar, return_counts=True)
    imp_unique, imp_counts = np.unique(imp, return_counts=True)
    thresholds = np.unique(np.hstack((tar_unique, imp_unique)))

    pt = np.hstack(
        (tar_counts, np.zeros(len(thresholds) - len(tar_counts), dtype=int))
    )
    pi = np.hstack(
        (np.zeros(len(thresholds) - len(imp_counts), dtype=int), imp_counts)
    )

    pt = pt[np.argsort(np.hstack((tar_unique, np.setdiff1d(imp_unique, tar_unique))))]
    pi = pi[np.argsort(np.hstack((np.setdiff1d(tar_unique, imp_unique), imp_unique)))]

    fr = np.zeros(pt.shape[0] + 1, dtype=int)
    fa = np.zeros(pi.shape[0] + 1, dtype=int)

    for i in range(1, len(pt) + 1):
        fr[i] = fr[i - 1] + pt[i - 1]

    for i in range(len(pt) - 1, -1, -1):
        fa[i] = fa[i + 1] + pi[i]

    frr = fr / len(tar)
    far = fa / len(imp)

    thresholds = np.hstack((thresholds, thresholds[-1] + 1e-6))

    return thresholds, frr, far

def compute_eer(tar, imp):
    tar_imp, fr, fa = compute_frr_far(tar, imp)

    index_min = np.argmin(np.abs(fr - fa))
    eer = 100.0 * np.mean((fr[index_min], fa[index_min]))
    threshold = tar_imp[index_min]

    return eer, threshold


def get_eer(tar, imp):
    return compute_eer(tar, imp)[0]

def main(th: float = 0.5):

    tars = np.random.rand(10)
    imps = np.random.rand(10)

    tar = tars[~np.isnan(tars)]
    imp = imps[~np.isnan(imps)]

    bpcer, apcer = get_fr_fa_at_threshold(tar=tar, imp=imp, threshold=th)
    acer = (bpcer + apcer) / 2.0
    eer = get_eer(tar=tar, imp=imp)

    print(f'bpcer: {bpcer}, apcer: {apcer}, eer: {eer} at threshold: {th}')

if __name__ == "__main__":
    from fire import Fire

    Fire(main)
