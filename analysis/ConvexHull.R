library(reticulate)
library(cxhull)
np <- import("numpy")

npz1 <- np$load("activationSummary_imagenette.npz")
npz2 <- np$load("labels_imagenette.npz")

activations <- npz1$f[["block5_conv3"]]
labels <- npz2$f[["labels"]]

activations1 <- activations[labels == 0, ]
activations1 <- rbind(activations1)
length(activations1)

activations2 <- activations[labels == 1, ]
activations2 <- rbind(activations2)
length(activations2)

hull1 <- cxhull(activations1)
