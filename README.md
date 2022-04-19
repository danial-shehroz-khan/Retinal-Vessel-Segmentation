# Retinal-Vessel-Segmentation
Retinal Vessel Segmentation using Python & OpenCV-based machine learning algorithm. From input image to output I performed the following set of operations.
 
# Input: Fundus Image

The main structures that can be visualized on a fundus photo are the central and peripheral retina, optic disc, and macula.

# Green Channel Extraction

Red light dominates the reflected spectrum as it gets less absorbed by the inner eye. That’s why the fundus image is reddish. Red light has a lower coefficient of absorption,  that’s is why pigments are less contrasted than green light.
One other reason why the Green channel is a better option is that the blue channel is very noisy and suffers from a poor dynamic range than the green channel. Blood containing elements (vessels) in the retinal layer is best represented and has the highest contrast in the green channel.

# Morphological Filters

The morphological operators are particularly useful for the analysis of binary images and common usages include edge detection, noise removal, image enhancement, and image segmentation. In this application, two morphological operators opening and closing are applied multiple times to separate the vessels from the background.

Adaptive Histogram Equalization

While performing AHE if the region being processed has a relatively small intensity range then the noise in that region gets more enhanced. To limit the appearance of such noise, a modification of AHE called CLAHE is used. The amount of contrast enhancement for some intensity is directly proportional to the slope of the CDF function at that intensity level.

# Edge Detection

The canny method is used for edge detection. The Canny method performs better than the other edge detection methods, because it uses two thresholds to detect strong and weak edges, and for this reason, the Canny algorithm is chosen for edge detection in the proposed technique.

# Thresholding

Thresholding is useful to remove unnecessary details from an image to concentrate on essentials. In the case of Fundus images, by removing all gray level information, the blood vessels are reduced to binary pixels.
