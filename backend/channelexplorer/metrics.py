import numpy as np

IMAGE_BATCH_TYPE = np.ndarray # [batch, channel, width, height]
SUMMARY_BATCH_TYPE = np.ndarray # [batch, vec]
DENSE_BATCH_TYPE = np.ndarray # [batch, vec]

def summary_fn_image_percentile(x: IMAGE_BATCH_TYPE) -> SUMMARY_BATCH_TYPE:
    return np.percentile(np.abs(x), 90, axis=(2,3))


def summary_fn_image_l2(x: IMAGE_BATCH_TYPE) -> SUMMARY_BATCH_TYPE:
    return np.linalg.norm(np.abs(x), axis=(2,3), ord=2)


def summary_fn_image_threshold_mean(x: IMAGE_BATCH_TYPE) -> SUMMARY_BATCH_TYPE:
    threshold = np.median(np.abs(x), axis=(2,3))
    return (x > threshold[...,None,None]).sum(axis=(2,3))


def summary_fn_image_threshold_median(x: IMAGE_BATCH_TYPE) -> SUMMARY_BATCH_TYPE:
    threshold = np.mean(np.abs(x), axis=(2,3))
    return (x > threshold[...,None,None]).sum(axis=(2,3))


def summary_fn_image_threshold_otsu(x: IMAGE_BATCH_TYPE) -> SUMMARY_BATCH_TYPE:
    bins_num = x.shape[2] * x.shape[3]
    batch_thresholds = []
    for batch in x:
        thresholds = []
        for img in batch:
            hist, bin_edges = np.histogram(img, bins=bins_num)
            bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2

            weight1 = np.cumsum(hist)
            weight2 = np.cumsum(hist[::-1])[::-1]
            # Get the class means mu0(t)
            mean1 = np.cumsum(hist * bin_mids) / weight1
            # Get the class means mu1(t)
            mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]

            inter_class_variance = (
                weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
            )

            # Maximize the inter_class_variance function val
            index_of_max_val = np.argmax(inter_class_variance)

            threshold = bin_mids[:-1][index_of_max_val]
            thresholds.append(threshold)
        batch_thresholds.append(thresholds)

    batch_thresholds = np.array(batch_thresholds)
    return (x > batch_thresholds[...,None,None]).sum(axis=(2,3))


def summary_fn_dense_identity(x: DENSE_BATCH_TYPE) -> SUMMARY_BATCH_TYPE:
    return x
