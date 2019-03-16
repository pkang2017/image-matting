# image-matting

	Poisson Matting
In this paper, we formulate the problem of natural image matting as one of solving Poisson equations with the matte gradient field. Our approach, which we call Poisson matting, has the following advantages. First, the matte is directly reconstructed from a continuous matte gradient field by solving Poisson equations using boundary information from a user-supplied trimap. Second, by interactively manipulating the matte gradient field using a number of filtering tools, the user can further improve Poisson matting results locally until he or she is satisfied. The modified local result is seamlessly integrated into the final result. Experiments on many complex natural images demonstrate that Poisson matting can generate good matting results that are not possible using existing matting techniques.
http://www.cs.jhu.edu/~misha/Fall07/Papers/Sun04.pdf
https://github.com/MarcoForte/poisson-matting

	Bayes Matting
This paper proposes a new Bayesian framework for solving the matting problem, i.e. extracting a foreground element from a background image by estimating an opacity for each pixel of the foreground element. Our approach models both the foreground and background color distributions with spatiallyvarying sets of Gaussians, and assumes a fractional blending of the foreground and background colors to produce the final output. It then uses a maximum-likelihood criterion to estimate the optimal opacity, foreground and background simultaneously. In addition to providing a principled approach to the matting problem, our algorithm effectively handles objects with intricate boundaries, such as hair strands and fur, and provides an improvement over existing techniques for these difficult cases.
https://github.com/MarcoForte/bayesian-matting

	A Closed Form Solution to Natural Image Matting
Interactive digital matting, the process of extracting a foreground object from an image based on limited user input, is an important task in image and video editing. From a computer vision perspective, this task is extremely challenging because it is massively ill-posed — at each pixel we must estimate the foreground and the background colors, as well as the foreground opacity (“alpha matte”) from a single color measurement. Current approaches either restrict the estimation to a small part of the image, estimating foreground and background colors based on nearby pixels where they are known, or perform iterative nonlinear estimation by alternating foreground and background color estimation with alpha estimation.
https://github.com/sjtrny/MatteKit
http://alphamatting.com/code.php

	Spectral Matting
We present 'spectral matting': a new approach to natural image matting that automatically computes a set of fundamental fuzzy matting components from the smallest eigenvectors of a suitably defined Laplacian matrix. Thus, our approach extends spectral segmentation techniques, whose goal is to extract hard segments, to the extraction of soft matting components. These components may then be used as building blocks to easily construct semantically meaningful foreground mattes, either in an unsupervised fashion, or based on a small amount of user input. 
http://alphamatting.com/code.php
http://www.vision.huji.ac.il/SpectralMatting/


	An Iterative Optimization Approach for Unified Image Segmentation and Matting
Separating a foreground object from the background in a static image involves determining both full and partial pixel coverages, also known as extracting a matte. Previous approaches require the input image to be pre-segmented into three regions: foreground, background and unknown, which is called a trimap. Partial opacity values are then computed only for pixels inside the unknown region. This presegmentation based approach fails for images with large portions of semi-transparent foreground where the trimap is difficult to create even manually. In this paper we combine the segmentation and matting problem together and propose a unified optimization approach based on Belief Propagation. We iteratively estimate the opacity value for every pixel in the image, based on a small sample of foreground and background pixels marked by the user. Experimental results show that compared with previous approaches, our method is more efficient to extract high quality mattes for foregrounds with significant semi-transparent regions.
http://juew.org/data/data.htm

	Fast Matting Using Large Kernel Matting Laplacian Matrices
Image matting is of great importance in both computer vision and graphics applications. Most existing state-of-the-art techniques rely on large sparse matrices such as the matting Laplacian [12]. However, solving these linear systems is often time-consuming, which is unfavored for the user interaction. In this paper, we propose a fast method for high quality matting. We first derive an efficient algorithm to solve a large kernel matting Laplacian. A large kernel propagates information more quickly and may improve the matte quality. To further reduce running time, we also use adaptive kernel sizes by a KD-tree trimap segmentation technique. A variety of experiments show that our algorithm provides high quality results and is 5 to 20 times faster than previous methods.
http://kaiminghe.com/publications/cvpr10matting.pdf
https://github.com/nathanbain314/alphaMatting

	Nonlocal Matting
This work attempts to considerably reduce the amount of user effort in the natural image matting problem. The key observation is that the nonlocal principle, introduced to denoise images, can be successfully applied to the alpha matte to obtain sparsity in matte representation, and therefore dramatically reduce the number of pixels a user needs to manually label. We show how to avoid making the user provide redundant and unnecessary input, develop a method for clustering the image pixels for the user to label, and a method to perform high-quality matte extraction. We show that this algorithm is therefore faster, easier, and higher quality than state of the art methods.
https://github.com/rocketman768/NonlocalMatting


	Shared Sampling for Real-Time Alpha Matting
Image matting aims at extracting foreground elements from an image by means of color and opacity (alpha) estimation. While a lot of progress has been made in recent years on improving the accuracy of matting techniques, one common problem persisted: the low speed of matte computation. We present the first real-time matting technique for natural images and videos. Our technique is based on the observation that, for small neighborhoods, pixels tend to share similar attributes. Therefore, independently treating each pixel in the unknown regions of a trimap results in a lot of redundant work. We show how this computation can be significantly and safely reduced by means of a careful selection of pairs of background and foreground samples. Our technique achieves speedups of up to two orders of magnitude compared to previous ones, while producing high-quality alpha mattes. The quality of our results has been verified through an independent benchmark. The speed of our technique enables, for the first time, real-time alpha matting of videos, and has the potential to enable a new class of exciting applications.
http://www.inf.ufrgs.br/~eslgastal/SharedMatting/

	Learning Based Digital Matting
We cast some new insights into solving the digital matting problem by treating it as a semi-supervised learning task in machine learning. A local learning based approach and a global learning based approach are then produced, to fit better the scribble based matting and the trimap based matting, respectively. Our approaches are easy to implement because only some simple matrix operations are needed. They are also extremely accurate because they can efficiently handle the nonlinear local color distributions by incorporating the kernel trick, that are beyond the ability of many previous works. Our approaches can outperform many recent matting methods, as shown by the theoretical analysis and comprehensive experiments. The new insights may also inspire several more works.
https://www.mathworks.com/matlabcentral/fileexchange/31412-learning-based-digital-matting
https://github.com/MarcoForte/learning-based-matting


	Learning based alpha matting using support vector regression
Alpha matting refers to the problem of estimating the opacity mask of the foreground in an image. Many recent algorithms solve it with color samples or some local assumptions, causing artifacts when they fail to collect appropriate samples or the assumptions do not hold. In this paper, we treat alpha matting as a supervised learning problem and propose a new matting approach. Given the input image and a trimap (labeling some foreground/background pixels), we segment the unlabeled region into pieces and learn the relations between pixel features and alpha values for these pieces. We use support vector regression (SVR) in the learning process. To obtain better learning results, we design a training samples selection method and use adaptive parameters for SVR. Qualitative and quantitative evaluations on a matting benchmark show that our approach outperforms many recent algorithms in terms of accuracy.
https://zhzhanp.github.io/papers/ICIP2012.pdf
https://github.com/Sunting78/effective-SVR-Matting


	An Alternative Matting Laplacian
Cutting out and object and estimate its transparency mask is a key task in many applications. We take on the work on closed-form matting by Levin et al.[1], that is used at the core of many matting techniques, and propose an alternative formulation that offers more flexible controls over the matting priors. We also show that this new approach is efficient at upscaling transparency maps from coarse estimates.
https://github.com/frcs/alternative-matting-laplacian

	A Global Sampling Method for Alpha Matting
Alpha matting refers to the problem of softly extracting the foreground from an image. Given a trimap (specifying known foreground/background and unknown pixels), a straightforward way to compute the alpha value is to sample some known foreground and background colors for each unknown pixel. Existing sampling-based matting methods often collect samples near the unknown pixels only. They fail if good samples cannot be found nearby. In this paper, we propose a global sampling method that uses all samples available in the image. Our global sample set avoids missing good samples. A simple but effective cost function is defined to tackle the ambiguity in the sample selection process. To handle the computational complexity introduced by the large number of samples, we pose the sampling task as a correspondence problem. The correspondence search is efficiently achieved by generalizing a randomized algorithm previously designed for patch matching[3]. A variety of experiments show that our global sampling method produces both visually and quantitatively high-quality matting results.
https://github.com/atilimcetin/global-matting
https://github.com/atilimcetin/guided-filter

	KNN matting
We are interested in a general alpha matting approach for the simultaneous extraction of multiple image layers; each layer may have disjoint segments for material matting not limited to foreground mattes typical of natural image matting. The estimated alphas also satisfy the summation constraint. Our approach does not assume the local color-line model, does not need sophisticated sampling strategies, and generalizes well to any color or feature space in any dimensions. Our matting technique, aptly called KNN matting, capitalizes on the nonlocal principle by using K nearest neighbors (KNN) in matching nonlocal neighborhoods, and contributes a simple and fast algorithm giving competitive results with sparse user markups. KNN matting has a closed-form solution that can leverage on the preconditioned conjugate gradient method to produce an efficient implementation. Experimental evaluation on benchmark datasets indicates that our matting results are comparable to or of higher quality than state of the art methods.
http://dingzeyu.li/projects/knn/

	Image Matting with Local and Nonlocal Smooth Priors
In this paper we propose a novel alpha matting method with local and nonlocal smooth priors. We observe that the manifold preserving editing propagation [4] essentially introduced a nonlocal smooth prior on the alpha matte. This nonlocal smooth prior and the well known local smooth prior from matting Laplacian complement each other. So we combine them with a simple data term from color sampling in a graph model for nature image matting. Our method has a closed-form solution and can be solved efficiently. Compared with the state-of-the-art methods, our method produces more accurate results according to the evaluation on standard benchmark datasets.
https://github.com/criminalking/image-matting

	Improving image matting using comprehensive sample sets
In this paper, we present a new image matting algorithm that achieves state-of-the-art performance on a benchmark dataset of images. This is achieved by solving two major problems encountered by current sampling based algorithms. The first is that the range in which the foreground and background are sampled is often limited to such an extent that the true foreground and background colors are not present. Here, we describe a method by which a more comprehensive and representative set of samples is collected so as not to miss out on the true samples. This is accomplished by expanding the sampling range for pixels farther from the foreground or background boundary and ensuring that samples from each color distribution are included. The second problem is the overlap in color distributions of foreground and background regions. This causes sampling based methods to fail to pick the correct samples for foreground and background. Our design of an objective function forces those foreground and background samples to be picked that are generated from well-separated distributions. Comparison on the dataset at and evaluation by www.alphamatting.com shows that the proposed method ranks first in terms of error measures used in the website.
https://github.com/supitalp/Comprehensive-Sample-Set-Image-Matting


	Weighted Color and Texture Sample Selection for Image Matting
Color information is leveraged by color sampling-based matting methods to find the best known samples for foreground and background color of unknown pixels. Such methods do not perform well if there is an overlap in the color distribution of foreground and background regions because color cannot distinguish between these regions and hence, the selected samples cannot reliably estimate the matte. Similarly, alpha propagation based matting methods may fail when the affinity among neighboring pixels is reduced by strong edges. In this paper, we overcome these two problems by considering texture as a feature that can complement color to improve matting. The contribution of texture and color is automatically estimated by analyzing the content of the image. An objective function containing color and texture components is optimized to choose the best foreground and background pair among a set of candidate pairs. Experiments are carried out on a benchmark data set and an independent evaluation of the results show that the proposed method is ranked first among all other image matting methods.
https://ieeexplore.ieee.org/document/6247741
https://github.com/walkoncross/alpha-image-matting

	Image Matting with KL-Divergence Based Sparse Sampling
Previous sampling-based image matting methods typically rely on certain heuristics in collecting representative samples from known regions, and thus their performance deteriorates if the underlying assumptions are not satisfied. To alleviate this, in this paper we take an entirely new approach and formulate sampling as a sparse subset selection problem where we propose to pick a small set of candidate samples that best explains the unknown pixels. Moreover, we describe a new distance measure for comparing two samples which is based on KL-divergence between the distributions of features extracted in the vicinity of the samples. Using a standard benchmark dataset for image matting, we demonstrate that our approach provides more accurate results compared with the state-of-the-art methods.
https://web.cs.hacettepe.edu.tr/~karacan/projects/klsparsematting/

	Designing Effective Inter-Pixel Information Flow for Natural Image Matting
We present a novel, purely affinity-based natural image matting algorithm. Our method relies on carefully defined pixel-to-pixel connections that enable effective use of information available in the image. We control the information flow from the known-opacity regions into the unknown region, as well as within the unknown region itself, by utilizing multiple definitions of pixel affinities. Among other forms of information flow, we introduce color-mixture flow, which builds upon local linear embedding and effectively encapsulates the relation between different pixel opacities. Our resulting novel linear system formulation can be solved in closed-form and is robust against several fundamental challenges of natural matting such as holes and remote intricate structures. Our evaluation using the alpha matting benchmark suggests a significant performance improvement over the current methods. While our method is primarily designed as a standalone matting tool, we show that it can also be used for regularizing mattes obtained by sampling-based methods. We extend our formulation to layer color estimation and show that the use of multiple channels of flow increases the layer color quality. We also demonstrate our performance in green-screen keying and further analyze the characteristics of the affinities used in our method.
http://people.inf.ethz.ch/aksoyy/ifm/


	Three-layer Graph Framework with the sumD Feature for Alpha Matting
Alpha matting, the process of extracting opacity mask of the foreground in an image, is an important task in image and video editing. All of the matting methods need exploit the relationships between pixels. The traditional propagation-based methods construct constrains based on nonlocal principle and color line model to reflect the relationships. However, these methods would produce artifacts if the constrains are not reliable. So we improve this problem in three points. Firstly, we design a novel feature called sumD feature to increase the pixel discrimination. This feature is simple and could encourage pixels with similar texture to have similar feature values. Secondly, we design a three-layer graph framework to construct nonlocal constrains. This framework finds constrains in multi-scale range and selects reliable constrains, then unifies nonlocal constrains according to their reliabilities. Thirdly, we develop a new label extension method to add hard constrains. Experimental results confirm that the effectiveness of the three changes, and the proposed method achieves high rank on the benchmark dataset.
http://www.alphamatting.com/code.php
https://www.sciencedirect.com/science/article/pii/S1077314217301236

	The W-Penalty and its Application to Alpha Matting with Sparse Labels
Alpha matting is an ill-posed problem, as such the user must supply dense partial labels for an acceptable solution to be reached. Unfortunately this labelling can be time consuming. In this paper we introduce the w-penalty function, which when incorporated into existing matting techniques allows users to supply extremely sparse input. The formulated objective function encourages driving matte values to 0 and 1. The experiments demonstrate the proposed model outperforms the state-of-the-art KNN matting algorithm. MATLAB code for our proposed method is freely available in the MatteKit package.
https://github.com/sjtrny/MatteKit
https://github.com/sjtrny/MatteKit/tree/master/wmatting

	AlphaGAN: Generative adversarial networks for natural image matting
We present the first generative adversarial network (GAN) for natural image matting. Our novel generator network is trained to predict visually appealing alphas with the addition of the adversarial loss from the discriminator that is trained to classify well composited images. Further, we improve existing encoder-decoder architectures to better deal with the spatial localization issues inherited in convolutional neural networks (CNN) by using dilated convolutions to capture global context information without downscaling feature maps and losing spatial information. We present state-of-the-art results on the alphamatting online benchmark for the gradient error and give comparable results in others. Our method is particularly well suited for fine structures like hair, which is of great importance in practical matting applications, e.g. in film/TV production.

https://github.com/CDOTAD/AlphaGAN-Matting

	TOM-Net: Learning Transparent Object Matting from a Single Image
This paper addresses the problem of transparent object matting. Existing image matting approaches for transparent objects often require tedious capturing procedures and long processing time, which limit their practical use. In this paper, we first formulate transparent object matting as a refractive flow estimation problem. We then propose a deep learning framework, called TOM-Net, for learning the refractive flow. Our framework comprises two parts, namely a multi-scale encoder-decoder network for producing a coarse prediction, and a residual network for refinement. At test time, TOM-Net takes a single image as input, and outputs a matte (consisting of an object mask, an attenuation mask and a refractive flow field) in a fast feed-forward pass. As no off-the-shelf dataset is available for transparent object matting, we create a large-scale synthetic dataset consisting of 178K images of transparent objects rendered in front of images sampled from the Microsoft COCO dataset. We also collect a real dataset consisting of 876 samples using 14 transparent objects and 60 background images. Promising experimental results have been achieved on both synthetic and real data, which clearly demonstrate the effectiveness of our approach.

http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_TOM-Net_Learning_Transparent_CVPR_2018_paper.pdf

https://github.com/guanyingc/TOM-Net

	Deep Image Matting
Image matting is a fundamental computer vision problem and has many applications. Previous algorithms have poor performance when an image has similar foreground and background colors or complicated textures. The main reasons are prior methods 1) only use low-level features and 2) lack high-level context. In this paper, we propose a novel deep learning based algorithm that can tackle both these problems. Our deep model has two parts. The first part is a deep convolutional encoder-decoder network that takes an image and the corresponding trimap as inputs and predict the alpha matte of the image. The second part is a small convolutional network that refines the alpha matte predictions of the first network to have more accurate alpha values and sharper edges. In addition, we also create a large-scale image matting dataset including 49300 training images and 1000 testing images. We evaluate our algorithm on the image matting benchmark, our testing set, and a wide variety of real images. Experimental results clearly demonstrate the superiority of our algorithm over previous methods.

https://github.com/Joker316701882/Deep-Image-Matting



Portrait Matting
	https://github.com/takiyu/portrait_matting
	https://github.com/PetroWu/AutoPortraitMatting
	http://xiaoyongshen.me/webpage_portrait/index.html
	http://www.cse.cuhk.edu.hk/leojia/projects/automatting/index.html
	https://github.com/lizhengwei1992/Fast_Portrait_Segmentation

