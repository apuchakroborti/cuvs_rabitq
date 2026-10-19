# Standalone RaBitQ decompression extension

This additive module reverses the output of the existing cuVS RaBitQ
`(codes, delta, vl)` compression API without modifying upstream cuVS sources.

The optimized GPU path performs scalar decode, optional centroid restoration,
inverse matrix or FHT-Kac rotation, padding removal, and an element-wise MSE
reduction. The external benchmark builds this module alongside its executable
and continues to use the prebuilt `libcuvs.so` for the unchanged compression
pipeline.
