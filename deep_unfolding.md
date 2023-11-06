# ./html/deep_unfolding

#### 1 uretinex-net: retinex-based deep unfolding network for low-light image enhancement
**Publication Date**: 06/01/2022, 00:00:00
**Citation Count**: 82
**Abstract**: Retinex model-based methods have shown to be effective in layer-wise manipulation with well-designed priors for low-light image enhancement. However, the commonly used handcrafted priors and optimization-driven solutions lead to the absence of adaptivity and efficiency. To address these issues, in this paper, we propose a Retinex-based deep unfolding network (URetinex-Net), which unfolds an optimization problem into a learnable network to decompose a low-light image into reflectance and illumination layers. By formulating the decomposition problem as an implicit priors regularized model, three learning-based modules are carefully designed, responsible for data-dependent initialization, high-efficient unfolding optimization, and user-specified illumination enhancement, respectively. Particularly, the proposed unfolding optimization module, introducing two networks to adaptively fit implicit priors in data-driven manner, can realize noise suppression and details preservation for the final decomposition results. Extensive experiments on real-world low-light images qualitatively and quantitatively demonstrate the effectiveness and superiority of the proposed method over state-of-the-art methods. The code is available at https://github.com/AndersonYong/URetinex-Net.
```bibtex
@Article{Wu2022URetinexNetRD,
 author = {Wen-Bin Wu and Jian Weng and Pingping Zhang and Xu Wang and Wenhan Yang and Jianmin Jiang},
 booktitle = {Computer Vision and Pattern Recognition},
 journal = {2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
 pages = {5891-5900},
 title = {URetinex-Net: Retinex-based Deep Unfolding Network for Low-light Image Enhancement},
 year = {2022}
}

```


#### 2 deep unfolding network for image super-resolution
**Publication Date**: 03/23/2020, 00:00:00
**Citation Count**: 371
**Abstract**: Learning-based single image super-resolution (SISR) methods are continuously showing superior effectiveness and efficiency over traditional model-based methods, largely due to the end-to-end training. However, different from model-based methods that can handle the SISR problem with different scale factors, blur kernels and noise levels under a unified MAP (maximum a posteriori) framework, learning-based methods generally lack such flexibility. To address this issue, this paper proposes an end-to-end trainable unfolding network which leverages both learningbased methods and model-based methods. Specifically, by unfolding the MAP inference via a half-quadratic splitting algorithm, a fixed number of iterations consisting of alternately solving a data subproblem and a prior subproblem can be obtained. The two subproblems then can be solved with neural modules, resulting in an end-to-end trainable, iterative network. As a result, the proposed network inherits the flexibility of model-based methods to super-resolve blurry, noisy images for different scale factors via a single model, while maintaining the advantages of learning-based methods. Extensive experiments demonstrate the superiority of the proposed deep unfolding network in terms of flexibility, effectiveness and also generalizability.
```bibtex
@Article{Zhang2020DeepUN,
 author = {K. Zhang and L. Gool and R. Timofte},
 booktitle = {Computer Vision and Pattern Recognition},
 journal = {2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
 pages = {3214-3223},
 title = {Deep Unfolding Network for Image Super-Resolution},
 year = {2020}
}

```


#### 3 deep unfolding network for spatiospectral image super-resolution
**Citation Count**: 19
**Abstract**: In this paper, we explore the spatiospectral image super-resolution (SSSR) task, i.e., joint spatial and spectral super-resolution, which aims to generate a high spatial resolution hyperspectral image (HR-HSI) from a low spatial resolution multispectral image (LR-MSI). To tackle such a severely ill-posed problem, one straightforward but inefficient way is to sequentially perform a single image super-resolution (SISR) network followed by a spectral super-resolution (SSR) network in a two-stage manner or reverse order. In this paper, we propose a model-based deep learning network for SSSR task, named unfolding spatiospectral super-resolution network (US3RN), which not only uses closed-form solutions to solve SISR subproblem and SSR subproblem, but also has extremely small parameters (only 295 K). In specific, we reformulate the image degradation and incorporate the spatiospectral super-resolution (SSSR) model, which takes the observation models of SISR and SSR into consideration. Then we solve the model-based energy function via the alternative direction multiplier method (ADMM) technique. Finally, we unfold the iterative process of the ADMM algorithm into a multistage network. Therefore, US3RN combines the merits of interpretability and generality of model-based methods with the advantages of learning-based methods. The experimental results show that, compared with the two-step method, US3RN achieves better results both quantitatively and qualitatively, while sharply reducing the number of parameters and FLOPs. Source code will be available at https://github.com/junjun-jiang/US3RN.
```bibtex
@Article{Ma2022DeepUN,
 author = {Qing Ma and Junjun Jiang and Xianming Liu and Jiayi Ma},
 booktitle = {IEEE Transactions on Computational Imaging},
 journal = {IEEE Transactions on Computational Imaging},
 pages = {28-40},
 title = {Deep Unfolding Network for Spatiospectral Image Super-Resolution},
 volume = {8},
 year = {2022}
}

```


#### 4 pancsc-net: a model-driven deep unfolding method for pansharpening
**Citation Count**: 36
**Abstract**: Recently, deep learning (DL) approaches have been widely applied to the pansharpening problem, which is defined as fusing a low-resolution multispectral (LRMS) image with a high-resolution panchromatic (PAN) image to obtain a high-resolution multispectral (HRMS) image. However, most DL-based methods handle this task by designing black-box network architectures to model the mapping relationship from LRMS and PAN to HRMS. These network architectures always lack sufficient interpretability, which limits their further performance improvements. To address this issue, we adopt the model-driven method to design an interpretable deep network structure for pansharpening. First, we present a new pansharpening model using the convolutional sparse coding (CSC), which is quite different from the current pansharpening frameworks. Second, an alternative algorithm is developed to optimize this model. This algorithm is further unfolded to a network, where each network module corresponds to a specific operation of the iterative algorithm. Therefore, the proposed network has clear physical interpretations, and all the learnable modules can be automatically learned in an end-to-end way from the given dataset. Experimental results on some benchmark datasets show that our network performs better than other advanced methods both quantitatively and qualitatively.
```bibtex
@Article{Cao2022PanCSCNetAM,
 author = {Xiangyong Cao and Xueyang Fu and D. Hong and Zongben Xu and Deyu Meng},
 booktitle = {IEEE Transactions on Geoscience and Remote Sensing},
 journal = {IEEE Transactions on Geoscience and Remote Sensing},
 pages = {1-13},
 title = {PanCSC-Net: A Model-Driven Deep Unfolding Method for Pansharpening},
 volume = {60},
 year = {2022}
}

```


#### 5 memory-augmented deep unfolding network for guided image super-resolution
**Publication Date**: 02/12/2022, 00:00:00
**Citation Count**: 21
```bibtex
@Article{Zhou2022MemoryAugmentedDU,
 author = {Man Zhou and Keyu Yan and Jin-shan Pan and Wenqi Ren and Qiaokang Xie and Xiangyong Cao},
 booktitle = {International Journal of Computer Vision},
 journal = {International Journal of Computer Vision},
 pages = {215-242},
 title = {Memory-Augmented Deep Unfolding Network for Guided Image Super-resolution},
 volume = {131},
 year = {2022}
}

```


#### 6 tight integration of neural- and clustering-based diarization through deep unfolding of infinite gaussian mixture model
**Publication Date**: 02/14/2022, 00:00:00
**Citation Count**: 12
**Abstract**: Speaker diarization has been investigated extensively as an important central task for meeting analysis. Recent trend shows that integration of end-to-end neural (EEND)- and clustering-based diarization is a promising approach to handle realistic conversational data containing overlapped speech with an arbitrarily large number of speakers, and achieved state-of-the-art results on various tasks. However, the approaches proposed so far have not realized tight integration yet, because the clustering employed therein was not optimal in any sense for clustering the speaker embeddings estimated by the EEND module. To address this problem, this paper introduces a trainable clustering algorithm into the integration framework, by deep-unfolding a non-parametric Bayesian model called the infinite Gaussian mixture model (iGMM). Specifically, the speaker embeddings are optimized during training such that it better fits iGMM clustering, based on a novel clustering loss based on Adjusted Rand Index (ARI). Experimental results based on CALLHOME data show that the proposed approach outperforms the conventional approach in terms of diarization error rate (DER), especially by substantially reducing speaker confusion errors, that indeed reflects the effectiveness of the proposed iGMM integration.
```bibtex
@Article{Kinoshita2022TightIO,
 author = {K. Kinoshita and Marc Delcroix and Tomoharu Iwata},
 booktitle = {IEEE International Conference on Acoustics, Speech, and Signal Processing},
 journal = {ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
 pages = {8382-8386},
 title = {Tight Integration Of Neural- And Clustering-Based Diarization Through Deep Unfolding Of Infinite Gaussian Mixture Model},
 year = {2022}
}

```


#### 7 ista-net++: flexible deep unfolding network for compressive sensing
**Publication Date**: 03/22/2021, 00:00:00
**Citation Count**: 45
**Abstract**: While deep neural networks have achieved impressive success in image compressive sensing (CS), most of them lack flexibility when dealing with multi-ratio tasks and multi-scene images in practical applications. To tackle these challenges, we propose a novel end-to-end flexible ISTA-unfolding deep network, dubbed ISTA-Net++, with superior performance and strong flexibility. Specifically, by developing a dynamic unfolding strategy, our model enjoys the adaptability of handling CS problems with different ratios, i.e., multi-ratio tasks, through a single model. A cross-block strategy is further utilized to reduce blocking artifacts and enhance the CS recovery quality. Furthermore, we adopt a balanced dataset for training, which brings more robustness when reconstructing images of multiple scenes. Extensive experiments on four datasets show that ISTA-Net++ achieves state-of-the-art results in terms of both quantitative metrics and visual quality. Considering its flexibility, effectiveness and practicability, our model is expected to serve as a suitable baseline in future CS research. The source code is available on https://github.com/jianzhangcs/ISTA-Netpp.
```bibtex
@Article{You2021ISTANETFD,
 author = {Di You and Jingfen Xie and Jian Zhang},
 booktitle = {IEEE International Conference on Multimedia and Expo},
 journal = {2021 IEEE International Conference on Multimedia and Expo (ICME)},
 pages = {1-6},
 title = {ISTA-NET++: Flexible Deep Unfolding Network for Compressive Sensing},
 year = {2021}
}

```


#### 8 memory-augmented deep unfolding network for compressive sensing
**Publication Date**: 10/17/2021, 00:00:00
**Citation Count**: 35
**Abstract**: Mapping a truncated optimization method into a deep neural network, deep unfolding network (DUN) has attracted growing attention in compressive sensing (CS) due to its good interpretability and high performance. Each stage in DUNs corresponds to one iteration in optimization. By understanding DUNs from the perspective of the human brain's memory processing, we find there exists two issues in existing DUNs. One is the information between every two adjacent stages, which can be regarded as short-term memory, is usually lost seriously. The other is no explicit mechanism to ensure that the previous stages affect the current stage, which means memory is easily forgotten. To solve these issues, in this paper, a novel DUN with persistent memory for CS is proposed, dubbed Memory-Augmented Deep Unfolding Network (MADUN). We design a memory-augmented proximal mapping module (MAPMM) by combining two types of memory augmentation mechanisms, namely High-throughput Short-term Memory (HSM) and Cross-stage Long-term Memory (CLM). HSM is exploited to allow DUNs to transmit multi-channel short-term memory, which greatly reduces information loss between adjacent stages. CLM is utilized to develop the dependency of deep information across cascading stages, which greatly enhances network representation capability. Extensive CS experiments on natural and MR images show that with the strong ability to maintain and balance information our MADUN outperforms existing state-of-the-art methods by a large margin. The source code is available at https://github.com/jianzhangcs/MADUN/.
```bibtex
@Article{Song2021MemoryAugmentedDU,
 author = {Jie Song and Bin Chen and Jian Zhang},
 booktitle = {ACM Multimedia},
 journal = {Proceedings of the 29th ACM International Conference on Multimedia},
 title = {Memory-Augmented Deep Unfolding Network for Compressive Sensing},
 year = {2021}
}

```


#### 9 dense deep unfolding network with 3d-cnn prior for snapshot compressive imaging
**Publication Date**: 09/14/2021, 00:00:00
**Citation Count**: 34
**Abstract**: Snapshot compressive imaging (SCI) aims to record three-dimensional signals via a two-dimensional camera. For the sake of building a fast and accurate SCI recovery algorithm, we incorporate the interpretability of model-based methods and the speed of learning-based ones and present a novel dense deep unfolding network (DUN) with 3D-CNN prior for SCI, where each phase is unrolled from an iteration of Half-Quadratic Splitting (HQS). To better exploit the spatial-temporal correlation among frames and address the problem of information loss between adjacent phases in existing DUNs, we propose to adopt the 3D-CNN prior in our proximal mapping module and develop a novel dense feature map (DFM) strategy, respectively. Besides, in order to promote network robustness, we further propose a dense feature map adaption (DFMA) module to allow inter-phase information to fuse adaptively. All the parameters are learned in an end-to-end fashion. Extensive experiments on simulation data and real data verify the superiority of our method. The source code is available at https://github.com/jianzhangcs/SCI3D.
```bibtex
@Article{Wu2021DenseDU,
 author = {Zhuoyuan Wu and Jian Zhang and Chong Mou},
 booktitle = {IEEE International Conference on Computer Vision},
 journal = {2021 IEEE/CVF International Conference on Computer Vision (ICCV)},
 pages = {4872-4881},
 title = {Dense Deep Unfolding Network with 3D-CNN Prior for Snapshot Compressive Imaging},
 year = {2021}
}

```


#### 10 amp-net: denoising-based deep unfolding for compressive image sensing
**Publication Date**: 04/21/2020, 00:00:00
**Citation Count**: 98
**Abstract**: Most compressive sensing (CS) reconstruction methods can be divided into two categories, i.e. model-based methods and classical deep network methods. By unfolding the iterative optimization algorithm for model-based methods onto networks, deep unfolding methods have the good interpretation of model-based methods and the high speed of classical deep network methods. In this article, to solve the visual image CS problem, we propose a deep unfolding model dubbed AMP-Net. Rather than learning regularization terms, it is established by unfolding the iterative denoising process of the well-known approximate message passing algorithm. Furthermore, AMP-Net integrates deblocking modules in order to eliminate the blocking artifacts that usually appear in CS of visual images. In addition, the sampling matrix is jointly trained with other network parameters to enhance the reconstruction performance. Experimental results show that the proposed AMP-Net has better reconstruction accuracy than other state-of-the-art methods with high reconstruction speed and a small number of network parameters.
```bibtex
@Article{Zhang2020AMPNetDD,
 author = {Zhonghao Zhang and Y. Liu and Jiani Liu and Fei Wen and Ce Zhu},
 booktitle = {IEEE Transactions on Image Processing},
 journal = {IEEE Transactions on Image Processing},
 pages = {1487-1500},
 title = {AMP-Net: Denoising-Based Deep Unfolding for Compressive Image Sensing},
 volume = {30},
 year = {2020}
}

```


#### 11 lr-csnet: low-rank deep unfolding network for image compressive sensing
**Publication Date**: 12/09/2022, 00:00:00
**Citation Count**: 6
**Abstract**: Deep unfolding networks (DUNs) have proven to be a viable approach to compressive sensing (CS). In this work, we propose a DUN called low-rank CS network (LR-CSNet) for natural image CS. Real-world image patches are often well-represented by low-rank approximations. LR-CSNet exploits this property by adding a low-rank prior to the CS optimization task. We derive a corresponding iterative optimization procedure using variable splitting, which is then translated to a new DUN architecture. The architecture uses low-rank generation modules (LRGMs), which learn low-rank matrix factorizations, as well as gradient descent and proximal mappings (GDPMs), which are proposed to extract high-frequency features to refine image details. In addition, the deep features generated at each reconstruction stage in the DUN are transferred between stages to boost the performance. Our extensive experiments on three widely considered datasets demonstrate the promising performance of LR-CSNet compared to state-of-the-art methods in natural image CS.
```bibtex
@Article{Zhang2022LRCSNetLD,
 author = {Tianfang Zhang and Lei Li and C. Igel and Stefan Oehmcke and F. Gieseke and Zhenming Peng},
 booktitle = {International Conference on Innovative Computing and Cloud Computing},
 journal = {2022 IEEE 8th International Conference on Computer and Communications (ICCC)},
 pages = {1951-1957},
 title = {LR-CSNet: Low-Rank Deep Unfolding Network for Image Compressive Sensing},
 year = {2022}
}

```


#### 12 high-throughput deep unfolding network for compressive sensing mri
**Publication Date**: 06/01/2022, 00:00:00
**Citation Count**: 7
**Abstract**: Deep unfolding network (DUN) has become the mainstream for compressive sensing MRI (CS-MRI) due to its good interpretability and high performance. Different optimization algorithms are usually unfolded into deep networks with different architectures, in which one iteration corresponds to one stage of DUN. However, there are few works discussing the following two questions: Which optimization algorithm is better after being unfolded into a DUN? What are the bottlenecks in existing DUNs? This paper attempts to answer these questions and give a feasible solution. For the first question, our mathematical and empirical analysis verifies the similarity of DUNs unfolded by alternating minimization (AM), alternating iterative shrinkage-thresholding algorithm (ISTA) and alternating direction method of multipliers (ADMM). For the second question, we point out that one major bottleneck of existing DUNs is that the input and output of each stage are just images of one channel, which greatly limits the transmission of network information. To break the information bottleneck, this paper proposes a novel, simple yet powerful high-throughput deep unfolding network (HiTDUN), which is not constrained by any optimization algorithm and can transmit multi-channel information between adjacent network stages. The developed multi-channel fusion strategy can also be easily incorporated into existing DUNs to further boost their performance. Extensive CS-MRI experiments on three benchmark datasets demonstrate that the proposed HiTDUN outperforms existing state-of-the-art DUNs by large margins while maintaining fast computational speed.11For reproducible research, the source codes and training models of our HiTDUN. [Online]. Available: https://github.com/jianzhangcs/HiTDUN.
```bibtex
@Article{Zhang2022HighThroughputDU,
 author = {Jian Zhang and Zhenyu Zhang and Jingfen Xie and Yongbing Zhang},
 booktitle = {IEEE Journal on Selected Topics in Signal Processing},
 journal = {IEEE Journal of Selected Topics in Signal Processing},
 pages = {750-761},
 title = {High-Throughput Deep Unfolding Network for Compressive Sensing MRI},
 volume = {16},
 year = {2022}
}

```


#### 13 mixed-timescale deep-unfolding for joint channel estimation and hybrid beamforming
**Publication Date**: 06/08/2022, 00:00:00
**Citation Count**: 8
**Abstract**: In massive multiple-input multiple-output (MIMO) systems, hybrid analog-digital beamforming is an essential technique for exploiting the potential array gain without using a dedicated radio frequency chain for each antenna. However, due to the large number of antennas, the conventional channel estimation and hybrid beamforming algorithms generally require high computational complexity and signaling overhead. In this work, we propose an end-to-end deep-unfolding neural network (NN) joint channel estimation and hybrid beamforming (JCEHB) algorithm to maximize the system sum rate in time-division duplex (TDD) massive MIMO. Specifically, the recursive least-squares (RLS) algorithm and stochastic successive convex approximation (SSCA) algorithm are unfolded for channel estimation and hybrid beamforming, respectively. In order to reduce the signaling overhead, we consider a mixed-timescale hybrid beamforming scheme, where the analog beamforming matrices are optimized based on the channel state information (CSI) statistics offline, while the digital beamforming matrices are designed at each time slot based on the estimated low-dimensional equivalent CSI matrices. We jointly train the analog beamformers together with the trainable parameters of the RLS and SSCA induced deep-unfolding NNs based on the CSI statistics offline. During data transmission, we estimate the low-dimensional equivalent CSI by the RLS induced deep-unfolding NN and update the digital beamformers. In addition, we propose a mixed-timescale deep-unfolding NN where the analog beamformers are optimized online, and extend the framework to frequency-division duplex (FDD) systems where channel feedback is considered. Simulation results show that the proposed algorithm can significantly outperform conventional algorithms with reduced computational complexity and signaling overhead.
```bibtex
@Article{Kang2022MixedTimescaleDF,
 author = {Kai Kang and Qiyu Hu and Yunlong Cai and Guanding Yu and J. Hoydis and Y. Eldar},
 booktitle = {IEEE Journal on Selected Areas in Communications},
 journal = {IEEE Journal on Selected Areas in Communications},
 pages = {2510-2528},
 title = {Mixed-Timescale Deep-Unfolding for Joint Channel Estimation and Hybrid Beamforming},
 volume = {40},
 year = {2022}
}

```


#### 14 two-stage is enough: a concise deep unfolding reconstruction network for flexible video compressive sensing
**Publication Date**: 01/15/2022, 00:00:00
**Citation Count**: 7
**Abstract**: We consider the reconstruction problem of video compressive sensing (VCS) under the deep unfolding/rolling structure. Yet, we aim to build a flexible and concise model using minimum stages. Different from existing deep unfolding networks used for inverse problems, where more stages are used for higher performance but without flexibility to different masks and scales, hereby we show that a 2-stage deep unfolding network can lead to the state-of-the-art (SOTA) results (with a 1.7dB gain in PSNR over the single stage model, RevSCI) in VCS. The proposed method possesses the properties of adaptation to new masks and ready to scale to large data without any additional training thanks to the advantages of deep unfolding. Furthermore, we extend the proposed model for color VCS to perform joint reconstruction and demosaicing. Experimental results demonstrate that our 2-stage model has also achieved SOTA on color VCS reconstruction, leading to a>2.3dB gain in PSNR over the previous SOTA algorithm based on plug-and-play framework, meanwhile speeds up the reconstruction by>17 times. In addition, we have found that our network is also flexible to the mask modulation and scale size for color VCS reconstruction so that a single trained network can be applied to different hardware systems. The code and models will be released to the public.
```bibtex
@Article{Zheng2022TwoStageIE,
 author = {Siming Zheng and Xiaoyu Yang and Xin Yuan},
 booktitle = {arXiv.org},
 journal = {ArXiv},
 title = {Two-Stage is Enough: A Concise Deep Unfolding Reconstruction Network for Flexible Video Compressive Sensing},
 volume = {abs/2201.05810},
 year = {2022}
}

```


#### 15 fast hierarchical deep unfolding network for image compressed sensing
**Publication Date**: 08/03/2022, 00:00:00
**Citation Count**: 4
**Abstract**: By integrating certain optimization solvers with deep neural network, deep unfolding network (DUN) has attracted much attention in recent years for image compressed sensing (CS). However, there still exist several issues in existing DUNs: 1) For each iteration, a simple stacked convolutional network is usually adopted, which apparently limits the expressiveness of these models. 2) Once the training is completed, most hyperparameters of existing DUNs are fixed for any input content, which significantly weakens their adaptability. In this paper, by unfolding the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA), a novel fast hierarchical DUN, dubbed FHDUN, is proposed for image compressed sensing, in which a well-designed hierarchical unfolding architecture is developed to cooperatively explore richer contextual prior information in multi-scale spaces. To further enhance the adaptability, series of hyperparametric generation networks are developed in our framework to dynamically produce the corresponding optimal hyperparameters according to the input content. Furthermore, due to the accelerated policy in FISTA, the newly embedded acceleration module makes the proposed FHDUN save more than 50% of the iterative loops against recent DUNs. Extensive CS experiments manifest that the proposed FHDUN outperforms existing state-of-the-art CS methods, while maintaining fewer iterations.
```bibtex
@Article{Cui2022FastHD,
 author = {Wenxue Cui and Shaohui Liu and Debin Zhao},
 booktitle = {ACM Multimedia},
 journal = {Proceedings of the 30th ACM International Conference on Multimedia},
 title = {Fast Hierarchical Deep Unfolding Network for Image Compressed Sensing},
 year = {2022}
}

```


#### 16 deep generalized unfolding networks for image restoration
**Publication Date**: 04/28/2022, 00:00:00
**Citation Count**: 65
**Abstract**: Deep neural networks (DNN) have achieved great suc-cess in image restoration. However, most DNN methods are designed as a black box, lacking transparency and inter-pretability. Although some methods are proposed to combine traditional optimization algorithms with DNN, they usually demand pre-defined degradation processes or hand-crafted assumptions, making it difficult to deal with complex and real-world applications. In this paper, we propose a Deep Generalized Unfolding Network (DGUNet) for image restoration. Concretely, without loss of interpretability, we integrate a gradient estimation strategy into the gradi-ent descent step of the Proximal Gradient Descent (PGD) algorithm, driving it to deal with complex and real-world image degradation. In addition, we design inter-stage in-formation pathways across proximal mapping in different PGD iterations to rectify the intrinsic information loss in most deep unfolding networks (DUN) through a multi-scale and spatial-adaptive way. By integrating the flexible gradi-ent descent and informative proximal mapping, we unfold the iterative PGD algorithm into a trainable DNN. Exten-sive experiments on various image restoration tasks demon-strate the superiority of our method in terms of state-of-the-art performance, interpretability, and generalizability. The source code is available at github.com/MC-E/DGUNet.
```bibtex
@Article{Mou2022DeepGU,
 author = {Chong Mou and Qian Wang and Jian Zhang},
 booktitle = {Computer Vision and Pattern Recognition},
 journal = {2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
 pages = {17378-17389},
 title = {Deep Generalized Unfolding Networks for Image Restoration},
 year = {2022}
}

```


#### 17 deep unfolding-based weighted averaging for federated learning under heterogeneous environments
**Citation Count**: 3
**Abstract**: Federated learning is a collaborative model training method by iterating model updates at multiple clients and aggregation of the updates at a central server. Device and statistical heterogeneity of the participating clients cause performance degradation so that an appropriate weight should be assigned per client in the server’s aggregation phase. This paper employs deep unfolding to learn the weights that adapt to the heterogeneity, which gives the model with high accuracy on uniform test data. The results of numerical experiments indicate the high performance of the proposed method and the interpretable behavior of the learned weights.
```bibtex
@Article{Nakai-Kasai2022DeepUW,
 author = {Ayano Nakai-Kasai and T. Wadayama},
 booktitle = {arXiv.org},
 journal = {ArXiv},
 title = {Deep Unfolding-based Weighted Averaging for Federated Learning under Heterogeneous Environments},
 volume = {abs/2212.12191},
 year = {2022}
}

```


#### 18 hybrid beamforming in mmwave mimo-ofdm systems via deep unfolding
**Publication Date**: 06/01/2022, 00:00:00
**Citation Count**: 3
**Abstract**: Designing hybrid beamforming transceivers in millimeter wave (mmWave) MIMO-OFDM systems with satisfactory performance and acceptable complexity is a challenging problem. The well-known weighted minimum mean square error manifold optimization (WMMSE-MO) algorithm offers desired performance but has high computational complexity. In this paper, we propose to apply the deep unfolding technique to the WMMSE-MO algorithm. The proposed deep unfolding model yields faster convergence to better solutions as compared to the original algorithm. Simulation results demonstrate remarkable spectral efficiency performance with reduced computational time complexity for the proposed scheme, under different hardware (RF chains) and algorithmic (inner/outer iterations) settings for a massive MIMO-OFDM system.
```bibtex
@Article{Chen2022HybridBI,
 author = {Kuan-Yuan Chen and Hsin-Yuan Chang and Ronald Y. Chang and W. Chung},
 booktitle = {IEEE Vehicular Technology Conference},
 journal = {2022 IEEE 95th Vehicular Technology Conference: (VTC2022-Spring)},
 pages = {1-7},
 title = {Hybrid Beamforming in mmWave MIMO-OFDM Systems via Deep Unfolding},
 year = {2022}
}

```


#### 19 deep unfolding for singular value decomposition compressed ghost imaging
**Publication Date**: 09/19/2022, 00:00:00
**Citation Count**: 3
```bibtex
@Article{Zhang2022DeepUF,
 author = {Chen Zhang and Jiaxuan Zhou and Jun Tang and Feng Wu and Hong Cheng and Sui Wei},
 booktitle = {Applied physics B},
 journal = {Applied Physics B},
 title = {Deep unfolding for singular value decomposition compressed ghost imaging},
 volume = {128},
 year = {2022}
}

```


#### 20 deep unfolding as iterative regularization for imaging inverse problems
**Publication Date**: 11/24/2022, 00:00:00
**Citation Count**: 3
**Abstract**: Recently, deep unfolding methods that guide the design of deep neural networks (DNNs) through iterative algorithms have received increasing attention in the field of inverse problems. Unlike general end-to-end DNNs, unfolding methods have better interpretability and performance. However, to our knowledge, their accuracy and stability in solving inverse problems cannot be fully guaranteed. To bridge this gap, we modified the training procedure and proved that the unfolding method is an iterative regularization method. More precisely, we jointly learn a convex penalty function adversarially by an input-convex neural network (ICNN) to characterize the distance to a real data manifold and train a DNN unfolded from the proximal gradient descent algorithm with this learned penalty. Suppose the real data manifold intersects the inverse problem solutions with only the unique real solution. We prove that the unfolded DNN will converge to it stably. Furthermore, we demonstrate with an example of MRI reconstruction that the proposed method outperforms conventional unfolding methods and traditional regularization methods in terms of reconstruction quality, stability and convergence speed.
```bibtex
@Article{Cui2022DeepUA,
 author = {Zhuoxu Cui and Qingyong Zhu and Jing Cheng and Dong Liang},
 booktitle = {arXiv.org},
 journal = {ArXiv},
 title = {Deep unfolding as iterative regularization for imaging inverse problems},
 volume = {abs/2211.13452},
 year = {2022}
}

```


#### 21 ddpg-driven deep-unfolding with adaptive depth for channel estimation with sparse bayesian learning
**Publication Date**: 01/20/2022, 00:00:00
**Citation Count**: 3
**Abstract**: Deep-unfolding neural networks (NNs) have received great attention since they achieve satisfactory performance with relatively low complexity. Typically, these deep-unfolding NNs are restricted to a fixed-depth for all inputs. However, the optimal number of layers required for convergence changes with different inputs. In this paper, we first develop a framework of deep deterministic policy gradient (DDPG)-driven deep-unfolding with adaptive depth for different inputs, where the trainable parameters of deep-unfolding NN are learned by DDPG, rather than updated by the stochastic gradient descent algorithm directly. Specifically, the optimization variables, trainable parameters, and architecture of deep-unfolding NN are designed as the state, action, and state transition of DDPG, respectively. Then, this framework is employed to deal with the channel estimation problem in massive multiple-input multiple-output systems. Specifically, first of all we formulate the channel estimation problem with an off-grid basis and develop a sparse Bayesian learning (SBL)-based algorithm to solve it. Secondly, the SBL-based algorithm is unfolded into a layer-wise structure with a set of introduced trainable parameters. Thirdly, the proposed DDPG-driven deep-unfolding framework is employed to solve this channel estimation problem based on the unfolded structure of the SBL-based algorithm. To realize adaptive depth, we design the halting score to indicate when to stop, which is a function of the channel reconstruction error. Furthermore, the proposed framework is extended to realize the adaptive depth of the general deep neural networks (DNNs). Simulation results show that the proposed algorithm outperforms the conventional optimization algorithms and DNNs with fixed depth with much reduced number of layers.
```bibtex
@Article{Hu2022DDPGDrivenDW,
 author = {Qiyu Hu and Shuhan Shi and Yunlong Cai and Guanding Yu},
 booktitle = {IEEE Transactions on Signal Processing},
 journal = {IEEE Transactions on Signal Processing},
 pages = {4665-4680},
 title = {DDPG-Driven Deep-Unfolding With Adaptive Depth for Channel Estimation With Sparse Bayesian Learning},
 volume = {70},
 year = {2022}
}

```


#### 22 deep unfolding dictionary learning for seismic denoising
**Publication Date**: 09/12/2022, 00:00:00
**Citation Count**: 3
**Abstract**: Seismic denoising is an essential step for seismic data processing. Conventionally, dictionary learning methods for seismic denoising always assume the representation coefficients to be sparse and the dictionary to be normalized or a tight frame. Current dictionary learning methods need to update the dictionary and the coefficients in an alternating iterative process. However, the dictionary obtained from dictionary learning method often needs to be recalculated for each input data. Moreover, the performance of dictionary learning for seismic noise removal is related to the parameter selection and the prior constraints of dictionary and representation coefficients. Recently, deep learning demonstrates promising performance in data prediction and classification. Following the architecture of dictionary learning algorithms strictly, we propose a novel and interpretable deep unfolding dictionary learning method for seismic denoising by unfolding the iterative algorithm of dictionary learning into a deep neural network. The proposed architecture of deep unfolding dictionary learning contains two main parts, the first is to update the dictionary and representation coefficients using least-square inversion and the second is to apply a deep neural network to learn the prior representation of dictionary and representation coefficients, respectively. Numerical synthetic and field examples show the effectiveness of our proposed method. More importantly,the proposed method for seismic denoising obtains the dictionary for each seismic data adaptively and is suitable for seismic data with different noise levels.
```bibtex
@Article{Sui2022DeepUD,
 author = {Y. Sui and Xiaojing Wang and Jianwei Ma},
 booktitle = {Geophysics},
 journal = {GEOPHYSICS},
 title = {Deep unfolding dictionary learning for seismic denoising},
 year = {2022}
}

```


#### 23 model-guided multi-contrast deep unfolding network for mri super-resolution reconstruction
**Publication Date**: 09/15/2022, 00:00:00
**Citation Count**: 3
**Abstract**: Magnetic resonance imaging (MRI) with high resolution (HR) provides more detailed information for accurate diagnosis and quantitative image analysis. Despite the significant advances, most existing super-resolution (SR) reconstruction network for medical images has two flaws: 1) All of them are designed in a black-box principle, thus lacking sufficient interpretability and further limiting their practical applications. Interpretable neural network models are of significant interest since they enhance the trustworthiness required in clinical practice when dealing with medical images. 2) most existing SR reconstruction approaches only use a single contrast or use a simple multi-contrast fusion mechanism, neglecting the complex relationships between different contrasts that are critical for SR improvement. To deal with these issues, in this paper, a novel Model-Guided interpretable Deep Unfolding Network (MGDUN) for medical image SR reconstruction is proposed. The Model-Guided image SR reconstruction approach solves manually designed objective functions to reconstruct HR MRI. We show how to unfold an iterative MGDUN algorithm into a novel model-guided deep unfolding network by taking the MRI observation matrix and explicit multi-contrast relationship matrix into account during the end-to-end optimization. Extensive experiments on the multi-contrast IXI dataset and BraTs 2019 dataset demonstrate the superiority of our proposed model.
```bibtex
@Article{Yang2022ModelGuidedMD,
 author = {Gang Yang and Li Zhang and Man Zhou and Aiping Liu and Xun Chen and Zhiwei Xiong and Feng Wu},
 booktitle = {ACM Multimedia},
 journal = {Proceedings of the 30th ACM International Conference on Multimedia},
 title = {Model-Guided Multi-Contrast Deep Unfolding Network for MRI Super-resolution Reconstruction},
 year = {2022}
}

```


#### 24 matrix-inverse-free deep unfolding of the weighted mmse beamforming algorithm
**Citation Count**: 3
**Abstract**: Downlink beamforming is a key technology for cellular networks. However, computing beamformers that maximize the weighted sum rate (WSR) subject to a power constraint is an NP-hard problem. The popular weighted minimum mean square error (WMMSE) algorithm converges to a local optimum but still exhibits considerable complexity. In order to address this trade-off between complexity and performance, we propose to apply deep unfolding to the WMMSE algorithm for a MU-MISO downlink channel. The main idea consists of mapping a fixed number of iterations of the WMMSE into trainable neural network layers. However, the formulation of the WMMSE algorithm, as provided in Shi et al., involves matrix inversions, eigendecompositions, and bisection searches. These operations are hard to implement as standard network layers. Therefore, we present a variant of the WMMSE algorithm i) that circumvents these operations by applying a projected gradient descent and ii) that, as a result, involves only operations that can be efficiently computed in parallel on hardware platforms designed for deep learning. We demonstrate that our variant of the WMMSE algorithm convergences to a stationary point of the WSR maximization problem and we accelerate its convergence by incorporating Nesterov acceleration and a generalization thereof as learnable structures. By means of simulations, we show that the proposed network architecture i) performs on par with the WMMSE algorithm truncated to the same number of iterations, yet at a lower complexity, and ii) generalizes well to changes in the channel distribution.
```bibtex
@Article{Pellaco2022MatrixInverseFreeDU,
 author = {Lissy Pellaco and Mats Bengtsson and J. Jaldén},
 booktitle = {IEEE Open Journal of the Communications Society},
 journal = {IEEE Open Journal of the Communications Society},
 pages = {65-81},
 title = {Matrix-Inverse-Free Deep Unfolding of the Weighted MMSE Beamforming Algorithm},
 volume = {3},
 year = {2022}
}

```


#### 25 deep unfolding for compressed sensing with denoiser
**Publication Date**: 07/18/2022, 00:00:00
**Citation Count**: 3
**Abstract**: Recent years have witnessed increasingly more exercises and uses of deep unfolding network (DUN) in image compressed sensing (CS) due to its high performance and interpretability. However, the existing DUN does not make full use of more flexible regularization methods. Besides, the intermediate information generated during the iterations of the DUN, which is crucial for the quality improvement of image reconstruction, has been largely overlooked in the existing methods. To alleviate this problem, we propose a novel DUN for image CS with regularization by denoising which casts half quadratic splitting (HQS) algorithm into the neural network. Further, we design an information collection strategy to leverage the useful information generated during the iterations. The information is provided to the image denoiser of the proposed network, which could enhance the image processing ability of the denoiser. The extensive experiments demonstrate that the proposed method is more efficient and achieves state-of-the-art reconstruction quality.
```bibtex
@Article{Ma2022DeepUF,
 author = {Chi Ma and Joey Tianyi Zhou and Xiao Zhang and Yu Zhou},
 booktitle = {IEEE International Conference on Multimedia and Expo},
 journal = {2022 IEEE International Conference on Multimedia and Expo (ICME)},
 pages = {01-06},
 title = {Deep Unfolding for Compressed Sensing with Denoiser},
 year = {2022}
}

```


#### 26 ensemble learning priors driven deep unfolding for scalable video snapshot compressive imaging
**Citation Count**: 9
```bibtex
@Article{Yang2022EnsembleLP,
 author = {Chengshuai Yang and Shenmin Zhang and Xin Yuan},
 booktitle = {European Conference on Computer Vision},
 pages = {600-618},
 title = {Ensemble Learning Priors Driven Deep Unfolding for Scalable Video Snapshot Compressive Imaging},
 year = {2022}
}

```


#### 27 deep unfolding for iterative stripe noise removal
**Publication Date**: 07/18/2022, 00:00:00
**Citation Count**: 1
**Abstract**: The non-uniform photoelectric response of infrared imaging systems results in fixed-pattern stripe noise being superimposed on infrared images, which severely reduces image quality. As the applications of degraded infrared images are limited, it is crucial to effectively preserve original details. Existing image destriping methods struggle to concurrently remove all stripe noise artifacts, preserve image details and structures, and balance real-time performance. In this paper we propose a novel algorithm for destriping degraded images, which takes advantage of neighbouring column signal correlation to remove independent column stripe noise. This is achieved through an iterative deep unfolding algorithm where the estimated noise of one network iteration is used as input to the next iteration. This progression substantially reduces the search space of possible function approximations, allowing for efficient training on larger datasets. The proposed method allows for a more precise estimation of stripe noise to preserve scene details more accurately. Extensive experimental results demonstrate that the proposed model outperforms existing destriping methods on artificially corrupted images on both quantitative and qualitative assessments.
```bibtex
@Article{Fayyaz2022DeepUF,
 author = {Zeshan Fayyaz and Daniel M. Platnick and Hannan Fayyaz and N. Farsad},
 booktitle = {IEEE International Joint Conference on Neural Network},
 journal = {2022 International Joint Conference on Neural Networks (IJCNN)},
 pages = {1-7},
 title = {Deep Unfolding for Iterative Stripe Noise Removal},
 year = {2022}
}

```


#### 28 a model-driven deep unfolding method for jpeg artifacts removal
**Publication Date**: 06/03/2021, 00:00:00
**Citation Count**: 24
**Abstract**: Deep learning-based methods have achieved notable progress in removing blocking artifacts caused by lossy JPEG compression on images. However, most deep learning-based methods handle this task by designing black-box network architectures to directly learn the relationships between the compressed images and their clean versions. These network architectures are always lack of sufficient interpretability, which limits their further improvements in deblocking performance. To address this issue, in this article, we propose a model-driven deep unfolding method for JPEG artifacts removal, with interpretable network structures. First, we build a maximum posterior (MAP) model for deblocking using convolutional dictionary learning and design an iterative optimization algorithm using proximal operators. Second, we unfold this iterative algorithm into a learnable deep network structure, where each module corresponds to a specific operation of the iterative algorithm. In this way, our network inherits the benefits of both the powerful model ability of data-driven deep learning method and the interpretability of traditional model-driven method. By training the proposed network in an end-to-end manner, all learnable modules can be automatically explored to well characterize the representations of both JPEG artifacts and image content. Experiments on synthetic and real-world datasets show that our method is able to generate competitive or even better deblocking results, compared with state-of-the-art methods both quantitatively and qualitatively.
```bibtex
@Article{Fu2021AMD,
 author = {Xueyang Fu and Menglu Wang and Xiangyong Cao and Xinghao Ding and Zhengjun Zha},
 booktitle = {IEEE Transactions on Neural Networks and Learning Systems},
 journal = {IEEE Transactions on Neural Networks and Learning Systems},
 pages = {6802-6816},
 title = {A Model-Driven Deep Unfolding Method for JPEG Artifacts Removal},
 volume = {33},
 year = {2021}
}

```


#### 29 interpretable neural networks for video separation: deep unfolding rpca with foreground masking
**Citation Count**: 2
**Abstract**: —This paper presents two deep unfolding neural networks for the simultaneous tasks of background subtraction and foreground detection in video. Unlike conventional neural networks based on deep feature extraction, we incorporate domain-knowledge models by considering a masked variation of the robust principal component analysis problem (RPCA). With this approach, we separate video clips into low-rank and sparse components, respectively corresponding to the backgrounds and foreground masks indicating the presence of moving objects. Our models, coined ROMAN-S and ROMAN-R, map the iterations of two alternating direction of multipliers methods (ADMM) to trainable convolutional layers, and the proximal operators are mapped to non-linear activation functions with trainable thresholds. This approach leads to lightweight networks with enhanced interpretability that can be trained on few data. In ROMAN-S, the correlation in time of successive binary masks is controlled with a side-information scheme based on ℓ 1 - ℓ 1 minimization. ROMAN-R enhances the foreground detection by learning a dictionary of atoms to represent the moving foreground in a high-dimensional feature space and by using reweighted-ℓ 1 - ℓ 1 minimization. Experiments are conducted on both synthetic and real video datasets, for which we also include an analysis of the generalization to unseen clips. Comparisons are made with existing deep unfolding RPCA neural networks, which do not use a mask formulation for the foreground. The models are also compared to a 3D U-Net baseline. Results show that our proposed models outperform other deep unfolding models, as well as the untrained optimization algorithms. ROMAN-R, in particular, is competitive with the U-Net baseline for foreground detection, with the additional advantage of providing video backgrounds and requiring substantially fewer training parameters and smaller training sets.
```bibtex
@Inproceedings{Joukovsky2022InterpretableNN,
 author = {Student Member Ieee Boris Joukovsky and F. I. Yonina C. Eldar and Member Ieee Nikos Deligiannis},
 title = {Interpretable Neural Networks for Video Separation: Deep Unfolding RPCA with Foreground Masking},
 year = {2022}
}

```


#### 30 deep unfolding for cooperative rate splitting multiple access in hybrid satellite terrestrial networks
**Publication Date**: 07/01/2022, 00:00:00
**Citation Count**: 1
**Abstract**: Rate splitting multiple access (RSMA) has shown great potentials for the next generation communication systems. In this work, we consider a two-user system in hybrid satellite terrestrial network (HSTN) where one of them is heavily shadowed and the other uses cooperative RSMA to improve the transmission quality. The non-convex weighted sum rate (WSR) problem formulated based on this model is usually optimized by computational burdened weighted minimum mean square error (WMMSE) algorithm. We propose to apply deep unfolding to solve the optimization problem, which maps WMMSE iterations into a layer-wise network and could achieve better performance within limited iterations. We also incorporate momentum accelerated projection gradient descent (PGD) algorithm to circumvent the complicated operations in WMMSE that are not amenable for unfolding and mapping. The momentum and step size in deep unfolding network are selected as trainable parameters for training. As shown in the simulation results, deep unfolding scheme has WSR and convergence speed advantages over original WMMSE algorithm.
```bibtex
@Article{Zhang2022DeepUF,
 author = {Qingmiao Zhang and Lidong Zhu and Shan Jiang and Xiaogang Tang},
 booktitle = {China Communications},
 journal = {China Communications},
 pages = {100-109},
 title = {Deep unfolding for cooperative rate splitting multiple access in hybrid satellite terrestrial networks},
 volume = {19},
 year = {2022}
}

```


#### 31 deep unfolding network for papr reduction in multicarrier ofdm systems
**Publication Date**: 11/01/2022, 00:00:00
**Citation Count**: 1
**Abstract**: In this letter, we propose a Deep Unfolding Network called PR-DUN to reduce the peak-to-average power ratio (PAPR), which is a thorny problem in Orthogonal Frequency-Division Multiplexing (OFDM) systems. The proposed multi-layer model is constructed by unrolling an iterative algorithm resulting in layers with trainable parameters, which are optimized to minimize a loss function related to the PAPR value. The deep unfolding model uses the backpropagation algorithm to transfer gradients backwards to adjust parameters. Furthermore, the proposed scheme can accommodate any transmit power constraint, and therefore can control the power increase caused by the auxiliary signal. Simulation results show that the proposed PR-DUN model achieves a larger PAPR reduction and a smaller bit error rate while being less computationally intensive than related solutions.
```bibtex
@Article{Nguyen2022DeepUN,
 author = {Minh-Thang Nguyen and Georges Kaddoum and Bassant Selim and K. V. Srinivas and Paulo Freitas de Araujo-Filho},
 booktitle = {IEEE Communications Letters},
 journal = {IEEE Communications Letters},
 pages = {2616-2620},
 title = {Deep Unfolding Network for PAPR Reduction in Multicarrier OFDM Systems},
 volume = {26},
 year = {2022}
}

```


#### 32 pu-detnet: deep unfolding aided smart sensing framework for cognitive radio
**Citation Count**: 2
**Abstract**: Spectrum sensing in cognitive radio (CR) paradigm can be broadly categorized as analytical-based and data-driven approaches. The former is sensitive to model inaccuracies in evolving network environment, while the latter (machine learning (ML)/deep learning (DL) based approach) suffers from high computational cost. For devices with low computational abilities, such approaches could be rendered less useful. In this context, we propose a deep unfolding architecture namely the Primary User-Detection Network (PU-DetNet) that harvests the strength of both: analytical and data-driven approaches. In particular, a technique is described that reduces computation in terms of inference time and the number of floating point operations (FLOPs). It involves binding the loss function such that each layer of the proposed architecture possesses its own loss function whose aggregate is optimized during training. Compared to the state-of-the-art, experimental results demonstrate that at SNR $= -10$ dB, the probability of detection is significantly improved as compared to the long short term memory (LSTM) scheme (between 39% and 56%), convolutional neural network (CNN) scheme (between 45% and 84%), and artificial neural network (ANN) scheme (between 53% and 128%) over empirical, 5G new radio, DeepSig, satellite communications, and radar datasets. The accuracy of proposed scheme also outperforms other existing schemes in terms of the F1-score. Additionally, inference time reduces by 91.69%, 90.90%, and 93.15%, while FLOPs reduces by 62.50%, 56.25%, 64.70% w.r.t. LSTM, CNN and ANN schemes, respectively. Moreover, the proposed scheme also shows improvement in throughput by 56.39%, 51.23%, and 69.52% as compared to LSTM, CNN and ANN schemes respectively, at SNR $= -6$ dB.
```bibtex
@Article{Soni2022PUDetNetDU,
 author = {Brijesh Soni and Dhaval K. Patel and Sanket B. Shah and M. López-Benítez and S. Govindasamy},
 booktitle = {IEEE Access},
 journal = {IEEE Access},
 pages = {98737-98751},
 title = {PU-DetNet: Deep Unfolding Aided Smart Sensing Framework for Cognitive Radio},
 volume = {10},
 year = {2022}
}

```


#### 33 deep unfolding of the dbfb algorithm with application to roi ct imaging with limited angular density
**Publication Date**: 09/27/2022, 00:00:00
**Citation Count**: 2
**Abstract**: This article presents a new method for reconstructing regions of interest (ROI) from a limited number of computed tomography (CT) measurements. Classical model-based iterative reconstruction methods lead to images with predictable features. Still, they often suffer from tedious parameterization and slow convergence. On the contrary, deep learning methods are fast, and they can reach high reconstruction quality by leveraging information from large datasets, but they lack interpretability. At the crossroads of both methods, deep unfolding networks have been recently proposed. Their design includes the physics of the imaging system and the steps of an iterative optimization algorithm. Motivated by the success of these networks for various applications, we introduce an unfolding neural network called U-RDBFB designed for ROI CT reconstruction from limited data. Few-view truncated data are effectively handled thanks to a robust non-convex data fidelity term combined with a sparsity-inducing regularization function. We unfold the Dual Block coordinate Forward-Backward (DBFB) algorithm, embedded in an iterative reweighted scheme, allowing the learning of key parameters in a supervised manner. Our experiments show an improvement over several state-of-the-art methods, including a model-based iterative scheme, a multi-scale deep learning architecture, and other deep unfolding methods.
```bibtex
@Article{Savanier2022DeepUO,
 author = {Marion Savanier and É. Chouzenoux and J. Pesquet and C. Riddell},
 booktitle = {IEEE Transactions on Computational Imaging},
 journal = {IEEE Transactions on Computational Imaging},
 pages = {502-516},
 title = {Deep Unfolding of the DBFB Algorithm With Application to ROI CT Imaging With Limited Angular Density},
 volume = {9},
 year = {2022}
}

```


#### 34 deep unfolding contrast source inversion for strong scatterers via generative adversarial mechanism
**Publication Date**: 11/01/2022, 00:00:00
**Citation Count**: 1
**Abstract**: To alleviate the extremely intrinsical ill-posedness and nonlinearity of electromagnetic inverse scattering under high contrast and low signal to noise ratio (SNR), we propose a deep unfolding network based on generative adversarial network (GAN) under contrast source inversion (CSI) framework, termed UCSI-GAN. The method solves inverse scattering problems (ISPs) using end-to-end generating confrontation way by incorporating a physical model together with its iterative updating formulation into the internal architecture of GAN. First, the nonlinear iterative scheme is extended to a deep unfolding generator network, and the contrast source and contrast updates are mapped to each module of the generator network. Second, to stabilize the imaging process, we add a refinement network to each variable update. Finally, the discriminator network is employed to ensure the authenticity of reconstructed images. The generator network and discriminator network are alternately trained with a generative adversarial learning strategy to reconstruct the properties of the medium object. Numerical experiments demonstrated that the performance of UCSI-GAN is better than traditional CSI and state-of-the-art learning approaches under high contrast and low SNR condition.
```bibtex
@Article{Zhou2022DeepUC,
 author = {Huilin Zhou and Yang Cheng and Huimin Zheng and Qiegen Liu and Yuhao Wang},
 booktitle = {IEEE transactions on microwave theory and techniques},
 journal = {IEEE Transactions on Microwave Theory and Techniques},
 pages = {4966-4979},
 title = {Deep Unfolding Contrast Source Inversion for Strong Scatterers via Generative Adversarial Mechanism},
 volume = {70},
 year = {2022}
}

```


#### 35 correct: a deep unfolding framework for motion-corrected quantitative r2* mapping
**Publication Date**: 10/12/2022, 00:00:00
**Citation Count**: 2
**Abstract**: Quantitative MRI (qMRI) refers to a class of MRI methods for quantifying the spatial distribution of biological tissue parameters. Traditional qMRI methods usually deal separately with artifacts arising from accelerated data acquisition, involuntary physical motion, and magnetic-field inhomogeneities, leading to suboptimal end-to-end performance. This paper presents CoRRECT, a unified deep unfolding (DU) framework for qMRI consisting of a model-based end-to-end neural network, a method for motion-artifact reduction, and a self-supervised learning scheme. The network is trained to produce R2* maps whose k-space data matches the real data by also accounting for motion and field inhomogeneities. When deployed, CoRRECT only uses the k-space data without any pre-computed parameters for motion or inhomogeneity correction. Our results on experimentally collected multi-Gradient-Recalled Echo (mGRE) MRI data show that CoRRECT recovers motion and inhomogeneity artifact-free R2* maps in highly accelerated acquisition settings. This work opens the door to DU methods that can integrate physical measurement models, biophysical signal models, and learned prior models for high-quality qMRI.
```bibtex
@Article{Xu2022CoRRECTAD,
 author = {Xiaojian Xu and Weijie Gan and Satya V. V. N. Kothapalli and D. Yablonskiy and U. Kamilov},
 booktitle = {arXiv.org},
 journal = {ArXiv},
 title = {CoRRECT: A Deep Unfolding Framework for Motion-Corrected Quantitative R2* Mapping},
 volume = {abs/2210.06330},
 year = {2022}
}

```


#### 36 a deep unfolding network for massive multi-user mimo-ofdm detection
**Publication Date**: 04/10/2022, 00:00:00
**Citation Count**: 1
**Abstract**: This paper proposes a novel deep unfolding (DU)- based iterative detection algorithm for massive multi-user (MU) multiple-input multiple-output (MIMO) orthogonal frequency-division multiplexing (OFDM) systems. Our model-driven algorithm fuses over-relaxed alternating direction method of multipliers (OR-ADMM) with DU tools, which combines the power of domain knowledge and data. By unfolding the iterations into neural network layers and designing a differentiable multi-level projection function, learnable parameters in the algorithm can be optimized via deep-learning techniques. For i.i.d. Gaussian channels, only one set of parameters is sufficient for all channel realizations. Different sets of parameters are learned to handle severe frequency selectivity and channel correlation under realistic 3GPP-3D channels. Our simulations demonstrate that this scheme outperforms traditional and DU-based detection algorithms in MU-MIMO-OFDM systems with similar or lower complexity. According to comparison results, our network achieves faster convergence and performs robustly on different channels, especially when the number of Tx and Rx antennas are equal.
```bibtex
@Article{Liu2022ADU,
 author = {Changjiang Liu and J. Thompson and T. Arslan},
 booktitle = {IEEE Wireless Communications and Networking Conference},
 journal = {2022 IEEE Wireless Communications and Networking Conference (WCNC)},
 pages = {2405-2410},
 title = {A Deep Unfolding Network for Massive Multi-user MIMO-OFDM Detection},
 year = {2022}
}

```


#### 37 structured deep unfolding network for optical remote sensing image super-resolution
**Citation Count**: 1
**Abstract**: Single-image super-resolution technology is critical in remote sensing, effectively improving the resolution of target images, with super-resolution algorithms based on deep learning demonstrating superior performance. However, most neural networks present shortcomings, such as a lack of interpretability and requiring a long training time, limiting them in some application scenarios. Moreover, due to the multidegradation factors, tasks put forward higher requirements for the adaptability of algorithms. Therefore, this work develops a structured deep unfolding network (SDUNet), which is adaptable and requires a lower training time by cascading multiple small network modules. Additionally, the unfolding strategy proposed deals with multiple degradations, fully exploiting prior knowledge. The suggested method is challenged against state-of-the-art neural network methods on one optical remote sensing image (ORSI) dataset and one natural image dataset. The experimental results demonstrate our method’s effectiveness in requiring less training time, involving fewer parameters, and achieving a higher reconstruction performance for ORSI super-resolution.
```bibtex
@Article{Shi2022StructuredDU,
 author = {M. Shi and Yesheng Gao and Lin Chen and Xingzhao Liu},
 booktitle = {IEEE Geoscience and Remote Sensing Letters},
 journal = {IEEE Geoscience and Remote Sensing Letters},
 pages = {1-5},
 title = {Structured Deep Unfolding Network for Optical Remote Sensing Image Super-Resolution},
 volume = {19},
 year = {2022}
}

```


#### 38 a deep unfolding method for satellite super resolution
**Citation Count**: 1
**Abstract**: Despite that existing deep-learning-based super resolution methods for satellite images have achieved great performance, these methods are generally designed to stack unaccountable and dense modules (i.e., residual blocks and dense blocks) to reach an optimal mapping function between low-resolution and high-resolution patches/images at the expense of computing resources. To address this challenge, we propose a deep unfolding method (LDUM) that includes two major components: 1) the pretreatment network and 2) the unfolding blocks. The main responsibility of the pretreatment network is to generate initial high-resolution images. Further, we model high-resolution images with the prior information, which can be seen as a combination of low-resolution and high-frequency residual images, and solve the optimization problem via the iterative proximal strategy. Specifically, we unfold the iterative process into a deep neural network to refine the reconstructed results, as each layer serves as an iterative step of the proposed optimization model. Thus, the proposed method is able to iteratively generate residual maps and high-resolution images by combining the powerful feature extraction capability of data-driven deep-learning-based methods and the interpretability of traditional model-driven algorithms. Experiments show that the proposed method, featured by its interpretable and lightweight merits, outperforms other state-of-the-art methods from quantitative and qualitative perspectives.
```bibtex
@Article{Wang2022ADU,
 author = {Jiaming Wang and Z. Shao and Xiao Huang and Tao Lu and Ruiqian Zhang},
 booktitle = {IEEE Transactions on Computational Imaging},
 journal = {IEEE Transactions on Computational Imaging},
 pages = {933-944},
 title = {A Deep Unfolding Method for Satellite Super Resolution},
 volume = {8},
 year = {2022}
}

```


#### 39 model-driven based deep unfolding equalizer for underwater acoustic ofdm communications
**Publication Date**: 07/10/2022, 00:00:00
**Citation Count**: 1
**Abstract**: It is challenging to design an equalizer for the complex time-frequency doubly-selective channel. In this paper, we employ the deep unfolding approach to establish an equalizer for the underwater acoustic (UWA) orthogonal frequency division multiplexing (OFDM) system, namely UDNet. Each layer of UDNet is designed according to the classical minimum mean square error (MMSE) equalizer. Moreover, we consider the QPSK equalization as a four-classification task and adopt minimum Kullback-Leibler (KL) to achieve a smaller symbol error rate (SER) with the one-hot coding instead of the MMSE criterion. In addition, we introduce a sliding structure based on the banded approximation of the channel matrix to reduce the network size and aid UDNet to perform well for different-length signals without changing the network structure. Furthermore, we apply the measured at-sea doubly-selective UWA channel and offshore background noise to evaluate the proposed equalizer. Experimental results show that the proposed UDNet performs better with low computational complexity. Concretely, the SER of UDNet is nearly an order of magnitude lower than that of MMSE.
```bibtex
@Article{Zhao2022ModelDrivenBD,
 author = {Hao Zhao and Cui Yang and Yalu Xu and Fei Ji and Miaowen Wen and Yankun Chen},
 booktitle = {IEEE Transactions on Vehicular Technology},
 journal = {IEEE Transactions on Vehicular Technology},
 pages = {6056-6067},
 title = {Model-Driven Based Deep Unfolding Equalizer for Underwater Acoustic OFDM Communications},
 volume = {72},
 year = {2022}
}

```


#### 40 an iterative discrete least square estimator with dynamic parameterization via deep-unfolding
**Publication Date**: 10/31/2022, 00:00:00
**Citation Count**: 1
**Abstract**: We propose a new dynamic parameterization approach via deep unfolding as an extension of the recently-introduced iterative discrete least square (IDLS) scheme, shown to elegantly generalize the conventional linear minimum mean squared error (LMMSE) method to enable the solution of inversion problems in complex multidimensional linear systems subject to discrete inputs. Configuring a layer-wise structure analogous to a deep neural network, the new approach enables an efficient optimization of the iterative IDLS algorithm, by finding optimal hyper-parameters for the related optimization problem through backpropagation and stochastic gradient descent techniques. The effectiveness of the proposed approach is confirmed via computer simulations.
```bibtex
@Article{Ando2022AnID,
 author = {Kengo Ando and Hiroki Iimori and Hyeon Seok Rou and G. Abreu and David González González and Osvaldo Gonsa},
 booktitle = {Asilomar Conference on Signals, Systems and Computers},
 journal = {2022 56th Asilomar Conference on Signals, Systems, and Computers},
 pages = {32-36},
 title = {An Iterative Discrete Least Square Estimator with Dynamic Parameterization via Deep-Unfolding},
 year = {2022}
}

```


#### 41 static output feedback synthesis of time-delay linear systems via deep unfolding
**Publication Date**: 05/19/2022, 00:00:00
**Citation Count**: 1
**Abstract**: : We propose a deep unfolding-based approach for stabilization of time-delay linear systems. Deep unfolding is an emerging framework for design and improvement of iterative algorithms and attracting signiﬁcant attentions in signal processing. In this paper, we propose an algorithm to design a static output feedback gain for stabilizing time-delay linear systems via deep unfolding. Within the algorithm, the learning part is driven by NeuralODE developed in the community of machine learning, while the gain veriﬁcation is performed with linear matrix inequalities developed in the systems and control theory. The eﬀectiveness of the proposed algorithm is illustrated with numerical simulations.
```bibtex
@Inproceedings{Ogura2022StaticOF,
 author = {Masaki Ogura and Koki Kobayashi and Kenji Sugimoto},
 title = {Static Output Feedback Synthesis of Time-Delay Linear Systems via Deep Unfolding},
 year = {2022}
}

```


#### 42 deep unfolding of image denoising by quantum interactive patches
**Publication Date**: 10/16/2022, 00:00:00
**Citation Count**: 4
**Abstract**: In this paper, we propose a blueprint of a new deep network unfolding a baseline quantum mechanics-based adaptive denoising scheme (De-QuIP). Relying on the theory of quantum many-body physics, the De-QuIP architecture incorporates local patch similarity measures through a term akin to interaction in quantum physics. Our proposed deep network embeds both quantum interactions and other quantum concepts, mainly the Hamiltonian operator. The integration of these quantum tools brings a nonlocal structure to the proposed deep network that harnesses the power of the convolutional layers to enhance the adaptability of the model. Thus, recasting De-QuIP in the framework of a deep learning network while preserving the essence of the baseline structure is the main contribution of this work. Experiments conducted on the Gaussian denoising problem, chosen here for illustration purpose, over a large sample set demonstrate start-of-the-art performance of the proposed deep network, dubbed as Deep-De-QuIP hereafter. Based on the properties of De-QuIP, its intrinsic adaptive structure, Deep-De-QuIP network could be easily extended to other noise models.
```bibtex
@Article{Dutta2022DeepUO,
 author = {S. Dutta and A. Basarab and B. Georgeot and D. Kouamé},
 booktitle = {International Conference on Information Photonics},
 journal = {2022 IEEE International Conference on Image Processing (ICIP)},
 pages = {491-495},
 title = {Deep Unfolding of Image Denoising by Quantum Interactive Patches},
 year = {2022}
}

```


#### 43 image denoising with deep unfolding and normalizing flows
**Publication Date**: 05/23/2022, 00:00:00
**Citation Count**: 1
**Abstract**: Many application domains, spanning from low-level computer vision to medical imaging, require high-fidelity images from noisy measurements. State-of-the-art methods for solving denoising problems combine deep learning with iterative model-based solvers, a concept known as deep algorithm unfolding or unrolling. By combining a-priori knowledge of the forward measurement model with learned proximal image-to-image mappings based on deep networks, these methods yield solutions that are both physically feasible (data-consistent) and perceptually plausible (consistent with prior belief). However, current proximal mappings based on (predominantly convolutional) neural networks only implicitly learn such image priors. In this paper, we propose to make these image priors fully explicit by embedding deep generative models in the form of normalizing flows within the unfolded proximal gradient algorithm, and training the entire algorithm in an end-to-end fashion. We demonstrate that the proposed method outperforms competitive baselines on image denoising.
```bibtex
@Article{Wei2022ImageDW,
 author = {Xinyi Wei and Hans Van Gorp and L. Gonzalez-Carabarin and D. Freedman and Y. Eldar and R. V. Sloun},
 booktitle = {IEEE International Conference on Acoustics, Speech, and Signal Processing},
 journal = {ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
 pages = {1551-1555},
 title = {Image Denoising with Deep Unfolding And Normalizing Flows},
 year = {2022}
}

```


#### 44 efficient deep unfolding for siso-ofdm channel estimation
**Publication Date**: 10/11/2022, 00:00:00
**Citation Count**: 6
**Abstract**: In modern communication systems, channel state information is of paramount importance to achieve capacity. It is then crucial to accurately estimate the channel. It is possible to perform SISO-OFDM channel estimation using sparse recovery techniques. However, this approach relies on the use of a physical wave propagation model to build a dictionary, which requires perfect knowledge of the system's parameters. In this paper, an unfolded neural network is used to lighten this constraint. Its architecture, based on a sparse recovery algorithm, allows SISO-OFDM channel estimation even if the system's parameters are not perfectly known. Indeed, its unsupervised online learning allows to learn the system's imperfections in order to enhance the estimation performance. The practicality of the proposed method is improved with respect to the state of the art in two aspects: constrained dictionaries are introduced in order to reduce sample complexity and hierarchical search within dictionaries is proposed in order to reduce time complexity. Finally, the performance of the proposed unfolded network is evaluated and compared to several baselines using realistic channel data, showing the great potential of the approach.
```bibtex
@Article{Chatelier2022EfficientDU,
 author = {Baptiste Chatelier and Luc Le Magoarou and Getachew Redieteab},
 booktitle = {ICC 2023 - IEEE International Conference on Communications},
 journal = {ICC 2023 - IEEE International Conference on Communications},
 pages = {3450-3455},
 title = {Efficient Deep Unfolding for SISO-OFDM Channel Estimation},
 year = {2022}
}

```


#### 45 deep unfolding architecture for mri reconstruction enhanced by adaptive noise maps
**Publication Date**: 09/01/2022, 00:00:00
**Citation Count**: 6
```bibtex
@Article{Aghabiglou2022DeepUA,
 author = {Amir Aghabiglou and E. Eksioglu},
 booktitle = {Biomedical Signal Processing and Control},
 journal = {Biomed. Signal Process. Control.},
 pages = {104016},
 title = {Deep unfolding architecture for MRI reconstruction enhanced by adaptive noise maps},
 volume = {78},
 year = {2022}
}

```


#### 46 learned robust pca: a scalable deep unfolding approach for high-dimensional outlier detection
**Publication Date**: 10/11/2021, 00:00:00
**Citation Count**: 20
**Abstract**: Robust principal component analysis (RPCA) is a critical tool in modern machine learning, which detects outliers in the task of low-rank matrix reconstruction. In this paper, we propose a scalable and learnable non-convex approach for high-dimensional RPCA problems, which we call Learned Robust PCA (LRPCA). LRPCA is highly efficient, and its free parameters can be effectively learned to optimize via deep unfolding. Moreover, we extend deep unfolding from finite iterations to infinite iterations via a novel feedforward-recurrent-mixed neural network model. We establish the recovery guarantee of LRPCA under mild assumptions for RPCA. Numerical experiments show that LRPCA outperforms the state-of-the-art RPCA algorithms, such as ScaledGD and AltProj, on both synthetic datasets and real-world applications.
```bibtex
@Article{Cai2021LearnedRP,
 author = {HanQin Cai and Jialin Liu and W. Yin},
 booktitle = {Neural Information Processing Systems},
 journal = {ArXiv},
 title = {Learned Robust PCA: A Scalable Deep Unfolding Approach for High-Dimensional Outlier Detection},
 volume = {abs/2110.05649},
 year = {2021}
}

```


#### 47 deep unfolding in multicell mu-mimo
**Publication Date**: 08/29/2022, 00:00:00
**Citation Count**: 1
**Abstract**: The weighted sum-rate maximization in coordinated multicell MIMO networks with intra- and intercell interference and local channel state at the base stations is considered. Based on the concept of unrolling applied to the classical weighted minimum mean squared error (WMMSE) algorithm and ideas from graph signal processing, we present the GCN-WMMSE deep network architecture for transceiver design in multicell MU-MIMO interference channels with local channel state information. Similar to the original WMMSE algorithm it facilitates a distributed implementation in multicell networks. However, GCN-WMMSE significantly accelerates the convergence and con-sequently alleviates the communication overhead in a distributed deployment. Additionally, the architecture is agnostic to different wireless network topologies while exhibiting a low number of trainable parameters and high efficiency w.r.t. training data.
```bibtex
@Article{Schynol2022DeepUI,
 author = {Lukas Schynol and M. Pesavento},
 booktitle = {European Signal Processing Conference},
 journal = {2022 30th European Signal Processing Conference (EUSIPCO)},
 pages = {1631-1635},
 title = {Deep Unfolding in Multicell MU-MIMO},
 year = {2022}
}

```


#### 48 deep unfolding with normalizing flow priors for inverse problems
**Publication Date**: 07/06/2021, 00:00:00
**Citation Count**: 18
**Abstract**: Many application domains, spanning from computational photography to medical imaging, require recovery of high-fidelity images from noisy, incomplete or partial/compressed measurements. State-of-the-art methods for solving these inverse problems combine deep learning with iterative model-based solvers, a concept known as deep algorithm unfolding or unrolling. By combining a-priori knowledge of the forward measurement model with learned proximal image-to-image mappings based on deep networks, these methods yield solutions that are both physically feasible (data-consistent) and perceptually plausible (consistent with prior belief). However, current proximal mappings based on (predominantly convolutional) neural networks only implicitly learn such image priors. In this paper, we propose to make these image priors fully explicit by embedding deep generative models in the form of normalizing flows within the unfolded proximal gradient algorithm, and training the entire algorithm end-to-end for a given task. We demonstrate that the proposed method outperforms competitive baselines on various image recovery tasks, spanning from image denoising to inpainting and deblurring, effectively adapting the prior to the restoration task at hand.
```bibtex
@Article{Wei2021DeepUW,
 author = {Xinyi Wei and Hans Van Gorp and Lizeth González Carabarin and D. Freedman and Yonina C. Eldar and R. V. Sloun},
 booktitle = {IEEE Transactions on Signal Processing},
 journal = {IEEE Transactions on Signal Processing},
 pages = {2962-2971},
 title = {Deep Unfolding With Normalizing Flow Priors for Inverse Problems},
 volume = {70},
 year = {2021}
}

```


#### 49 deep plug-and-play and deep unfolding methods for image restoration
**Citation Count**: 2
```bibtex
@Article{Zhang2022DeepPA,
 author = {K. Zhang and R. Timofte},
 booktitle = {Advanced Methods and Deep Learning in Computer Vision},
 journal = {Advanced Methods and Deep Learning in Computer Vision},
 title = {Deep plug-and-play and deep unfolding methods for image restoration},
 year = {2022}
}

```


#### 50 deep unfolding with weighted ℓ₂ minimization for compressive sensing
**Publication Date**: 02/15/2021, 00:00:00
**Citation Count**: 12
**Abstract**: Compressive sensing (CS) aims to accurately reconstruct high-dimensional signals from a small number of measurements by exploiting signal sparsity and structural priors. However, signal priors utilized in existing CS reconstruction algorithms rely mainly on hand-crafted design, which often cannot offer the best sparsity-undersampling tradeoff because high-order structural priors of signals are hard to be captured in this manner. In this article, a new recovery guarantee of the unified CS reconstruction model-weighted $\ell _{1}$ minimization (WL1M) is derived, which indicates universal priors could hardly lead to the optimal selection of the weights. Motivated by the analysis, we propose a deep unfolding network for the general WL1M model. The proposed deep unfolding-based WL1M (D-WL1M) integrates universal priors with learning capability so that all of the parameters, including the crucial weights, can be learned from training data. We demonstrate the proposed D-WL1M outperforms several state-of-the-art CS-based methods and deep learning-based methods by a large margin via the experiments on the Caltech-256 image data set.
```bibtex
@Article{Zhang2021DeepUW,
 author = {Jun Zhang and Yuanqing Li and Z. Yu and Z. Gu and Yu Cheng and Huoqing Gong},
 booktitle = {IEEE Internet of Things Journal},
 journal = {IEEE Internet of Things Journal},
 pages = {3027-3041},
 title = {Deep Unfolding With Weighted ℓ₂ Minimization for Compressive Sensing},
 volume = {8},
 year = {2021}
}

```


#### 51 designing interpretable recurrent neural networks for video reconstruction via deep unfolding
**Publication Date**: 04/02/2021, 00:00:00
**Citation Count**: 12
**Abstract**: Deep unfolding methods design deep neural networks as learned variations of optimization algorithms through the unrolling of their iterations. These networks have been shown to achieve faster convergence and higher accuracy than the original optimization methods. In this line of research, this paper presents novel interpretable deep recurrent neural networks (RNNs), designed by the unfolding of iterative algorithms that solve the task of sequential signal reconstruction (in particular, video reconstruction). The proposed networks are designed by accounting that video frames’ patches have a sparse representation and the temporal difference between consecutive representations is also sparse. Specifically, we design an interpretable deep RNN (coined reweighted-RNN) by unrolling the iterations of a proximal method that solves a reweighted version of the <inline-formula> <tex-math notation="LaTeX">$\ell _{1}$ </tex-math></inline-formula>-<inline-formula> <tex-math notation="LaTeX">$\ell _{1}$ </tex-math></inline-formula> minimization problem. Due to the underlying minimization model, our reweighted-RNN has a different thresholding function (alias, different activation function) for each hidden unit in each layer. In this way, it has higher network expressivity than existing deep unfolding RNN models. We also present the derivative <inline-formula> <tex-math notation="LaTeX">$\ell _{1}$ </tex-math></inline-formula>-<inline-formula> <tex-math notation="LaTeX">$\ell _{1}$ </tex-math></inline-formula>-RNN model, which is obtained by unfolding a proximal method for the <inline-formula> <tex-math notation="LaTeX">$\ell _{1}$ </tex-math></inline-formula>-<inline-formula> <tex-math notation="LaTeX">$\ell _{1}$ </tex-math></inline-formula> minimization problem. We apply the proposed interpretable RNNs to the task of video frame reconstruction from low-dimensional measurements, that is, sequential video frame reconstruction. The experimental results on various datasets demonstrate that the proposed deep RNNs outperform various RNN models.
```bibtex
@Article{Luong2021DesigningIR,
 author = {Huynh Van Luong and B. Joukovsky and Nikos Deligiannis},
 booktitle = {IEEE Transactions on Image Processing},
 journal = {IEEE Transactions on Image Processing},
 pages = {4099-4113},
 title = {Designing Interpretable Recurrent Neural Networks for Video Reconstruction via Deep Unfolding},
 volume = {30},
 year = {2021}
}

```


#### 52 iterative algorithm induced deep-unfolding neural networks: precoding design for multiuser mimo systems
**Publication Date**: 06/01/2020, 00:00:00
**Citation Count**: 89
**Abstract**: Optimization theory assisted algorithms have received great attention for precoding design in multiuser multiple-input multiple-output (MU-MIMO) systems. Although the resultant optimization algorithms are able to provide excellent performance, they generally require considerable computational complexity, which gets in the way of their practical application in real-time systems. In this work, in order to address this issue, we first propose a framework for deep-unfolding, where a general form of iterative algorithm induced deep-unfolding neural network (IAIDNN) is developed in matrix form to better solve the problems in communication systems. Then, we implement the proposed deep-unfolding framework to solve the sum-rate maximization problem for precoding design in MU-MIMO systems. An efficient IAIDNN based on the structure of the classic weighted minimum mean-square error (WMMSE) iterative algorithm is developed. Specifically, the iterative WMMSE algorithm is unfolded into a layer-wise structure, where a number of trainable parameters are introduced to replace the high-complexity operations in the forward propagation. To train the network, a generalized chain rule of the IAIDNN is proposed to depict the recurrence relation of gradients between two adjacent layers in the back propagation. Moreover, we discuss the computational complexity and generalization ability of the proposed scheme. Simulation results show that the proposed IAIDNN efficiently achieves the performance of the iterative WMMSE algorithm with reduced computational complexity.
```bibtex
@Article{Hu2020IterativeAI,
 author = {Qiyu Hu and Yunlong Cai and Qingjiang Shi and Kaidi Xu and Guanding Yu and Z. Ding},
 booktitle = {IEEE Transactions on Wireless Communications},
 journal = {IEEE Transactions on Wireless Communications},
 pages = {1394-1410},
 title = {Iterative Algorithm Induced Deep-Unfolding Neural Networks: Precoding Design for Multiuser MIMO Systems},
 volume = {20},
 year = {2020}
}

```


#### 53 generalization error bounds for deep unfolding rnns
**Citation Count**: 8
**Abstract**: Recurrent Neural Networks (RNNs) are powerful models with the ability to model sequential data. However, they are often viewed as black-boxes and lack in interpretability. Deep unfolding methods take a step towards interpretability by designing deep neural networks as learned variations of iterative optimization algorithms to solve various signal processing tasks. In this paper, we explore theoretical aspects of deep unfolding RNNs in terms of their generalization ability. Specifically, we derive generalization error bounds for a class of deep unfolding RNNs via Rademacher complexity analysis. To our knowledge, these are the first generalization bounds proposed for deep unfolding RNNs. We show theoretically that our bounds are tighter than similar ones for other recent RNNs, in terms of the number of timesteps. By training models in a classification setting, we demonstrate that deep unfolding RNNs can outperform traditional RNNs in standard sequence classification tasks. These experiments allow us to relate the empirical generalization error to the theoretical bounds. In particular, we show that over-parametrized deep unfolding models like reweighted-RNN achieve tight theoretical error bounds with minimal decrease in accuracy, when trained with explicit regularization.
```bibtex
@Article{Joukovsky2021GeneralizationEB,
 author = {B. Joukovsky and Tanmoy Mukherjee and Huynh Van Luong and Nikos Deligiannis},
 booktitle = {Conference on Uncertainty in Artificial Intelligence},
 pages = {1515-1524},
 title = {Generalization error bounds for deep unfolding RNNs},
 year = {2021}
}

```


#### 54 admm-dad net: a deep unfolding network for analysis compressed sensing
**Publication Date**: 10/13/2021, 00:00:00
**Citation Count**: 8
**Abstract**: In this paper, we propose a new deep unfolding neural network based on the ADMM algorithm for analysis Compressed Sensing. The proposed network jointly learns a redundant analysis operator for sparsification and reconstructs the signal of interest. We compare our proposed network with a state-of-the-art unfolded ISTA decoder, that also learns an orthogonal sparsifier. Moreover, we consider not only image, but also speech datasets as test examples. Computational experiments demonstrate that our proposed network outperforms the state-of-the-art deep unfolding network, consistently for both real-world image and speech datasets.
```bibtex
@Article{Kouni2021ADMMDADNA,
 author = {Vicky Kouni and Georgios Paraskevopoulos and H. Rauhut and G. Alexandropoulos},
 booktitle = {IEEE International Conference on Acoustics, Speech, and Signal Processing},
 journal = {ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
 pages = {1506-1510},
 title = {ADMM-DAD Net: A Deep Unfolding Network for Analysis Compressed Sensing},
 year = {2021}
}

```


#### 55 deep-unfolding neural-network aided hybrid beamforming based on symbol-error probability minimization
**Publication Date**: 09/14/2021, 00:00:00
**Citation Count**: 7
**Abstract**: In massive multiple-input multiple-output (MIMO) systems, hybrid analog-digital (AD) beamforming can be used to attain a high directional gain without requiring a dedicated radio frequency (RF) chain for each antenna element, which substantially reduces both the hardware costs and power consumption. While massive MIMO transceiver design typically relies on the conventional mean-square error (MSE) criterion, directly minimizing the symbol error rate (SER) can lead to a superior performance. In this article, we first mathematically formulate the problem of hybrid transceiver design under the minimum SER (MSER) optimization criterion and then develop an MSER-based iterative gradient descent (GD) algorithm to find the related stationary points. We then propose a deep-unfolding neural network (NN). The iterative GD algorithm is unfolded into a multi-layer structure wherein trainable parameters are introduced to accelerate the convergence and enhance the overall system performance. To implement the training stage, we derive the relationship between adjacent layers' gradients based on the generalized chain rule (GCR). The deep-unfolding NN is developed for both quadrature phase shift keying (QPSK) and $M$-ary quadrature amplitude modulated (QAM) signals, and its convergence is investigated theoretically. Furthermore, we analyze the transfer capability, computational complexity, and generalization capability of the proposed deep-unfolding NN. Our simulation results show that the latter significantly outperforms its conventional counterpart at a reduced complexity.
```bibtex
@Article{Shi2021DeepUnfoldingNA,
 author = {Shuhan Shi and Yunlong Cai and Qiyu Hu and B. Champagne and L. Hanzo},
 booktitle = {IEEE Transactions on Vehicular Technology},
 journal = {IEEE Transactions on Vehicular Technology},
 pages = {529-545},
 title = {Deep-Unfolding Neural-Network Aided Hybrid Beamforming Based on Symbol-Error Probability Minimization},
 volume = {72},
 year = {2021}
}

```


#### 56 photothermal-sr-net: a customized deep unfolding neural network for photothermal super resolution imaging
**Publication Date**: 04/21/2021, 00:00:00
**Citation Count**: 7
**Abstract**: This article presents deep unfolding neural networks to handle inverse problems in photothermal radiometry enabling super-resolution (SR) imaging. The photothermal SR approach is a well-known technique to overcome the spatial resolution limitation in photothermal imaging by extracting high-frequency spatial components based on the deconvolution with the thermal point spread function (PSF). However, stable deconvolution can only be achieved by using the sparse structure of defect patterns, which often requires tedious, hand-crafted tuning of hyperparameters and results in computationally intensive algorithms. On this account, this article proposes Photothermal-SR-Net, which performs deconvolution by deep unfolding considering the underlying physics. Since defects appear sparsely in materials, our approach includes trained block-sparsity thresholding in each convolutional layer. This enables to super-resolve 2-D thermal images for nondestructive testing (NDT) with a substantially improved convergence rate compared to classic approaches. The performance of the proposed approach is evaluated on various deep unfolding and thresholding approaches. Furthermore, we explored how to increase the reconstruction quality and the computational performance. Thereby, it was found that the computing time for creating high-resolution images could be significantly reduced without decreasing the reconstruction quality by using pixel binning as a preprocessing step.
```bibtex
@Article{Ahmadi2021PhotothermalSRNetAC,
 author = {S. Ahmadi and Linh Kästner and J. C. Hauffen and P. Jung and M. Ziegler},
 booktitle = {IEEE Transactions on Instrumentation and Measurement},
 journal = {IEEE Transactions on Instrumentation and Measurement},
 pages = {1-9},
 title = {Photothermal-SR-Net: A Customized Deep Unfolding Neural Network for Photothermal Super Resolution Imaging},
 volume = {71},
 year = {2021}
}

```


#### 57 deep unfolding network for block-sparse signal recovery
**Publication Date**: 06/06/2021, 00:00:00
**Citation Count**: 6
**Abstract**: Block-sparse signal recovery has drawn increasing attention in many areas of signal processing, where the goal is to recover a high-dimensional signal whose non-zero coefficients only arise in a few blocks from compressive measurements. However, most off-the-shelf data-driven reconstruction networks do not exploit the block-sparse structure. Thus, they suffer from deteriorating performance in block-sparse signal recovery. In this paper, we put forward a block-sparse reconstruction network named Ada-BlockLISTA based on the concept of deep unfolding. Our proposed network consists of a gradient descent step on every single block followed by a block-wise shrinkage step. We evaluate the performance of the proposed Ada-BlockLISTA network through simulations based on the signal model of two-dimensional (2D) harmonic retrieval problems.
```bibtex
@Article{Fu2021DeepUN,
 author = {Rong Fu and Vincent Monardo and Tianyao Huang and Yimin Liu},
 booktitle = {IEEE International Conference on Acoustics, Speech, and Signal Processing},
 journal = {ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
 pages = {2880-2884},
 title = {Deep Unfolding Network for Block-Sparse Signal Recovery},
 year = {2021}
}

```


#### 58 soft-output joint channel estimation and data detection using deep unfolding
**Publication Date**: 10/17/2021, 00:00:00
**Citation Count**: 6
**Abstract**: We propose a novel soft-output joint channel estimation and data detection (JED) algorithm for multiuser (MU) multiple-input multiple-output (MIMO) wireless communication systems. Our algorithm approximately solves a maximum a-posteriori JED optimization problem using deep unfolding and generates soft-output information for the transmitted bits in every iteration. The parameters of the unfolded algorithm are computed by a hyper-network that is trained with a binary cross entropy (BCE) loss. We evaluate the performance of our algorithm in a coded MU-MIMO system with 8 basestation antennas and 4 user equipments and compare it to state-of-the-art algorithms separate channel estimation from soft-output data detection. Our results demonstrate that our JED algorithm outperforms such data detectors with as few as 10 iterations.
```bibtex
@Article{Song2021SoftOutputJC,
 author = {Haochuan Song and X. You and Chuan Zhang and Christoph Studer},
 booktitle = {Information Theory Workshop},
 journal = {2021 IEEE Information Theory Workshop (ITW)},
 pages = {1-5},
 title = {Soft-Output Joint Channel Estimation and Data Detection using Deep Unfolding},
 year = {2021}
}

```


#### 59 hybrid precoding design based on dual-layer deep-unfolding neural network
**Publication Date**: 09/13/2021, 00:00:00
**Citation Count**: 5
**Abstract**: Dual-layer iterative algorithms are generally required when solving resource allocation problems in wireless communication systems. Specifically, the spectrum efficiency maximization problem for hybrid precoding architecture is hard to solve by the single-layer iterative algorithm. The dual-layer penalty dual decomposition (PDD) algorithm has been proposed to address the problem. Although the PDD algorithm achieves significant performance, it requires high computational complexity, which hinders its practical applications in real-time systems. To address this issue, we first propose a novel framework for deep-unfolding, where a dual-layer deep-unfolding neural network (DLDUNN) is formulated. We then apply the proposed frame-work to solve the spectrum efficiency maximization problem for hybrid precoding architecture. An efficient DLDUNN is designed based on unfolding the iterative PDD algorithm into a layer-wise structure. We also introduce some trainable parameters in place of the high-complexity operations. Simulation results show that the DLDUNN presents the performance of the PDD algorithm with remarkably reduced complexity.
```bibtex
@Article{Zhang2021HybridPD,
 author = {Guangyi Zhang and Xiao Fu and Qiyu Hu and Yunlong Cai and Guanding Yu},
 booktitle = {IEEE International Symposium on Personal, Indoor and Mobile Radio Communications},
 journal = {2021 IEEE 32nd Annual International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC)},
 pages = {678-683},
 title = {Hybrid Precoding Design Based on Dual-Layer Deep-Unfolding Neural Network},
 year = {2021}
}

```


#### 60 stochastic deep unfolding for imaging inverse problems
**Publication Date**: 06/06/2021, 00:00:00
**Citation Count**: 3
**Abstract**: Deep unfolding networks are rapidly gaining attention for solving imaging inverse problems. However, the computational and memory complexity of existing deep unfolding networks scales with the size of the full measurement set, limiting their applicability to certain large-scale imaging inverse problems. We propose SCRED-Net as a novel methodology that introduces a stochastic approximation to the unfolded regularization by denoising (RED) algorithm. Our method uses only a subset of measurements within each cascade block, making it scalable to a large number of measurements for efficient end-to-end training. We present numerical results showing the effectiveness of SCRED-Net on intensity diffraction tomography (IDT) and sparse-view computed tomography (CT). Our results show that SCRED-Net matches the performance of a batch deep unfolding network at a fraction of training and operational complexity.
```bibtex
@Article{Liu2021StochasticDU,
 author = {Jiaming Liu and Yu Sun and Weijie Gan and Xiaojian Xu and B. Wohlberg and U. Kamilov},
 booktitle = {IEEE International Conference on Acoustics, Speech, and Signal Processing},
 journal = {ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
 pages = {1395-1399},
 title = {Stochastic Deep Unfolding for Imaging Inverse Problems},
 year = {2021}
}

```


#### 61 panchromatic side sparsity model based deep unfolding network for pansharpening
**Citation Count**: 2
**Abstract**: Deep learning recently receives state of the art results in pansharpening. However, most of existing pansharpening neural networks are purely data-driven, without taking account of the characteristics of pansharpening task. To address this issue, we propose a novel deep unfolding network for pansharpening that combines the insight of variational optimization model and the capability of deep neural network. We firstly develop a Panchromatic Side Sparsity (PASS) prior based variational optimization model for pansharpend image reconstruction, which is formulated as the ℓ1-ℓ1 minimization. In particular, the PASS prior is defined using the transform sparsity, which can alleviate the influences of the irregular outliers between the multispectral (MS) and panchromatic (PAN) images. The iterations of half-quadratic splitting algorithm for solving the ℓ1-ℓ1 minimization are then deeply unfolded into a deep neural network, referred as PASS-Net. To capture the nonlinear relationship between MS and PAN images, the linear transforms used in PASS prior are extended into the subnetworks in PASS-Net. Moreover, a pair of learnable downsampling and upsampling modules are designed to realize the downsampling and upsampling operations, which can improve the flexibility. The experimental results on different satellites datasets confirm that PASS-Net is superior to some representational traditional methods and state-of-the-art deep learning based methods.
```bibtex
@Article{Yin2021PanchromaticSS,
 author = {Haitao Yin},
 booktitle = {IEEE Transactions on Geoscience and Remote Sensing},
 journal = {IEEE Transactions on Geoscience and Remote Sensing},
 pages = {1-1},
 title = {Panchromatic Side Sparsity Model Based Deep Unfolding Network for Pansharpening},
 volume = {PP},
 year = {2021}
}

```


#### 62 hcgm-net: a deep unfolding network for financial index tracking
**Publication Date**: 06/06/2021, 00:00:00
**Citation Count**: 1
**Abstract**: Tracking the performance of a financial index by selecting a subset of assets composing the index is a problem that raises several difficulties due to the large size of the stock market. Typically, optimisation algorithms with high complexity are employed to address such problems. In this paper, we focus on sparse index tracking and employ a Frank-Wolfe-based algorithm which we translate into a deep neural network, a strategy known as deep unfolding. Numerical experiments show that the learned model outperforms the iterative algorithm, leading to high accuracy at a low computational cost. To the best of our knowledge, this is the first deep unfolding design proposed for financial data processing.
```bibtex
@Article{Pauwels2021HCGMNetAD,
 author = {Ruben Pauwels and Evaggelia Tsiligianni and Nikos Deligiannis},
 booktitle = {IEEE International Conference on Acoustics, Speech, and Signal Processing},
 journal = {ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
 pages = {3910-3914},
 title = {HCGM-Net: A Deep Unfolding Network for Financial Index Tracking},
 year = {2021}
}

```


#### 63 deep unfolding based hyper‐parameter optimisation for self‐interference cancellation in lte‐a/5g‐transceivers
**Publication Date**: 05/29/2021, 00:00:00
**Citation Count**: 1
**Abstract**: Deep unfolding is a very promising concept that allows to combine the advantages of traditional estimation techniques, such as adaptive filters, and machine learning approaches, like artificial neural networks. Focusing on a challenging self-interference problem occurring in frequency-division duplex radio frequency transceivers, namely modulated spurs, it is shown that deep unfolding enables remarkable performance gains. Based on the hyper-parameter optimisation of several least-mean squares (LMS) variants and the recursive-least squares algorithm, the importance of a well-chosen loss function are highlighted. Especially the variable step-size LMS and the transform-domain LMS vastly benefit without increased runtime complexity.
```bibtex
@Article{Motz2021DeepUB,
 author = {C. Motz and T. Paireder and M. Huemer},
 booktitle = {Electronics Letters},
 journal = {Electronics Letters},
 title = {Deep unfolding based hyper‐parameter optimisation for self‐interference cancellation in LTE‐A/5G‐transceivers},
 year = {2021}
}

```


#### 64 super-resolution power of deep unfolding networks to tomosar 3d imaging
**Publication Date**: 08/29/2021, 00:00:00
**Citation Count**: 1
**Abstract**: Deep unfolding networks have been proved to be able to achieve super-resolution (SR). Different from sparsity regularization methods, deep unfolding networks can speed up the imaging process and own good robustness against noise. However, the optimal network design and maximum SR ability of deep unfolding networks have not been quantified. In this paper, the optimal number of layers and the SR extremity of the network are confirmed by simulation. And some sparsity regularization methods are compared with the proposed method. Simulation results validate the superiority of the proposed method in terms of SR imaging and accuracy.
```bibtex
@Conference{Fan2021SuperResolutionPO,
 author = {L. Fan and Qi Yang and Yang Zeng and B. Deng and Hongqiang Wang},
 booktitle = {International Conference on Infrared, Millimeter, and Terahertz Waves},
 journal = {2021 46th International Conference on Infrared, Millimeter and Terahertz Waves (IRMMW-THz)},
 pages = {1-2},
 title = {Super-Resolution Power of Deep Unfolding Networks to TomoSAR 3D Imaging},
 year = {2021}
}

```


#### 65 theoretical linear convergence of deep unfolding network for block-sparse signal recovery
**Publication Date**: 11/18/2021, 00:00:00
**Citation Count**: 1
**Abstract**: In this paper, we consider the recovery of the high-dimensional block-sparse signal from a compressed set of measurements, where the nonzero coefficients of the recovered signal occur in several blocks. Adopting the idea of deep unfolding, we explore the block-sparse structure and put forward a block-sparse reconstruction network named Ada-BlockLISTA, which performs gradient descent on every single block followed by a block-wise shrinkage. Furthermore, we prove the linear convergence rate of our proposed network, which also theoretically guarantees exact recovery for a potentially higher sparsity level based on the underlying block structure. Numerical results indicate that Ada-BlockLISTA yields better signal recovery performance than existing algorithms, which ignore the additional block structure in the signal model.
```bibtex
@Conference{Fu2021TheoreticalLC,
 author = {Rong Fu and Yimin Liu and Xiuhong Li},
 booktitle = {Conference on Computer Science and Communication Technology},
 pages = {125066J - 125066J-6},
 title = {Theoretical linear convergence of deep unfolding network for block-sparse signal recovery},
 volume = {12506},
 year = {2021}
}

```


#### 66 deep unfolding for communications systems: a survey and some new directions
**Publication Date**: 06/13/2019, 00:00:00
**Citation Count**: 144
**Abstract**: Deep unfolding is a method of growing popularity that fuses iterative optimization algorithms with tools from neural networks to efficiently solve a range of tasks in machine learning, signal and image processing, and communication systems. This survey summarizes the principle of deep unfolding and discusses its recent use for communication systems with focus on detection and precoding in multi-antenna (MIMO) wireless systems and belief propagation decoding of error-correcting codes. To showcase the efficacy and generality of deep unfolding, we describe a range of other tasks relevant to communication systems that can be solved using this emerging paradigm. We conclude the survey by outlining a list of open research problems and future research directions.
```bibtex
@Article{Balatsoukas-Stimming2019DeepUF,
 author = {Alexios Balatsoukas-Stimming and Christoph Studer},
 booktitle = {IEEE Workshop on Signal Processing Systems},
 journal = {2019 IEEE International Workshop on Signal Processing Systems (SiPS)},
 pages = {266-271},
 title = {Deep Unfolding for Communications Systems: A Survey and Some New Directions},
 year = {2019}
}

```


#### 67 redefining wireless communication for 6g: signal processing meets deep learning with deep unfolding
**Publication Date**: 04/22/2020, 00:00:00
**Citation Count**: 43
**Abstract**: The year 2019 witnessed the rollout of the 5G standard, which promises to offer significant data rate improvement over 4G. While 5G is still in its infancy, there has been an increased shift in the research community for communication technologies beyond 5G. The recent emergence of machine learning approaches for enhancing wireless communications and empowering them with much-desired intelligence holds immense potential for redefining wireless communication for 6G. The evolving communication systems will be bottlenecked in terms of latency, throughput, and reliability by the underlying signal processing at the physical layer. In this position letter, we motivate the need to redesign iterative signal processing algorithms by leveraging deep unfolding techniques to fulfill the physical layer requirements for 6G networks. To this end, we begin by presenting the service requirements and the key challenges posed by the envisioned 6G communication architecture. We outline the deficiencies of the traditional algorithmic principles and data-hungry deep learning (DL) approaches in the context of 6G networks. Specifically, deep unfolded signal processing is presented by sketching the interplay between domain knowledge and DL. The deep unfolded approaches reviewed in this letter are positioned explicitly in the context of the requirements imposed by the next generation of cellular networks. Finally, this letter motivates open research challenges to truly realize hardware-efficient edge intelligence for future 6G networks. Impact Statement—In this letter, we discuss why the infusion of domain knowledge into machine learning frameworks holds the key to future embedded intelligent communication systems. Applying traditional signal processing and deep learning approaches independently entails significant computational and memory constraints. This becomes challenging in the context of future communication networks, such as 6G with significant communication demands where dense deployments of embedded Internet of Things (IoT) devices are envisioned. Hence, we put forth deep unfolded approaches as the potential enabling technology for 6G artificial intelligence (AI) radio to mitigate the computational and memory demands as well as to fulfill the future 6G latency, reliability, and throughput requirements. To this end, we present a general deep unfolding methodology that can be applied to iterative signal processing algorithms. Thereafter, we survey some initial steps taken in this direction and more importantly discuss the potential it has in overcoming challenges in the context of 6G requirements. This letter concludes by providing future research directions in this promising realm.
```bibtex
@Article{Jagannath2020RedefiningWC,
 author = {Anu Jagannath and Jithin Jagannath and T. Melodia},
 booktitle = {IEEE Transactions on Artificial Intelligence},
 journal = {IEEE Transactions on Artificial Intelligence},
 pages = {528-536},
 title = {Redefining Wireless Communication for 6G: Signal Processing Meets Deep Learning With Deep Unfolding},
 volume = {2},
 year = {2020}
}

```


#### 68 deep-unfolding beamforming for intelligent reflecting surface assisted full-duplex systems
**Publication Date**: 12/04/2021, 00:00:00
**Citation Count**: 15
**Abstract**: In this paper, we investigate an intelligent reflecting surface (IRS) assisted multi-user multiple-input multiple-output (MIMO) full-duplex (FD) system. We jointly optimize the active beamforming matrices at the access point (AP) and uplink users, and the passive beamforming matrix at the IRS to maximize the weighted sum-rate of the system. Since it is practically difficult to acquire the channel state information (CSI) for IRS-related links due to its passive operation and large number of elements, we conceive a mixed-timescale beamforming scheme. Specifically, the high-dimensional passive beamforming matrix at the IRS is updated based on the channel statistics while the active beamforming matrices are optimized relied on the low-dimensional real-time effective CSI at each time slot. We propose an efficient stochastic successive convex approximation (SSCA)-based algorithm for jointly designing the active and passive beamforming matrices. Moreover, due to the high computational complexity caused by the matrix inversion computation in the SSCA-based optimization algorithm, we further develop a deep-unfolding neural network (NN) to address this issue. The proposed deep-unfolding NN maintains the structure of the SSCA-based algorithm but introduces a novel non-linear activation function and some learnable parameters induced by the first-order Taylor expansion to approximate the matrix inversion. In addition, we develop a black-box NN as a benchmark. Simulation results show that the proposed mixed-timescale algorithm outperforms the existing single-timescale algorithm and the proposed deep-unfolding NN approaches the performance of the SSCA-based algorithm with much reduced computational complexity when deployed online.
```bibtex
@Article{Liu2021DeepUnfoldingBF,
 author = {Yanzhen Liu and Qiyu Hu and Yunlong Cai and Guanding Yu and Geoffrey Y. Li},
 booktitle = {IEEE Transactions on Wireless Communications},
 journal = {IEEE Transactions on Wireless Communications},
 pages = {1-1},
 title = {Deep-Unfolding Beamforming for Intelligent Reflecting Surface assisted Full-Duplex Systems},
 volume = {PP},
 year = {2021}
}

```


#### 69 accurate and lightweight image super-resolution with model-guided deep unfolding network
**Publication Date**: 09/14/2020, 00:00:00
**Citation Count**: 34
**Abstract**: Deep neural networks (DNNs) based methods have achieved great success in single image super-resolution (SISR). However, existing state-of-the-art SISR techniques are designed like black boxes lacking transparency and interpretability. Moreover, the improvement in visual quality is often at the price of increased model complexity due to black-box design. In this paper, we present and advocate an explainable approach toward SISR named model-guided deep unfolding network (MoG-DUN). Targeting at breaking the coherence barrier, we opt to work with a well-established image prior named nonlocal auto-regressive model and use it to guide our DNN design. By integrating deep denoising and nonlocal regularization as trainable modules within a deep learning framework, we can unfold the iterative process of model-based SISR into a multistage concatenation of building blocks with three interconnected modules (denoising, nonlocal-AR, and reconstruction). The design of all three modules leverages the latest advances including dense/skip connections as well as fast nonlocal implementation. In addition to explainability, MoG-DUN is accurate (producing fewer aliasing artifacts), computationally efficient (with reduced model parameters), and versatile (capable of handling multiple degradations). The superiority of the proposed MoG-DUN method to existing state-of-the-art image SR methods including RCAN, SRMDNF, and SRFBN is substantiated by extensive experiments on several popular datasets and various degradation scenarios.
```bibtex
@Article{Ning2020AccurateAL,
 author = {Qiang Ning and W. Dong and Guangming Shi and Leida Li and Xin Li},
 booktitle = {IEEE Journal on Selected Topics in Signal Processing},
 journal = {IEEE Journal of Selected Topics in Signal Processing},
 pages = {240-252},
 title = {Accurate and Lightweight Image Super-Resolution With Model-Guided Deep Unfolding Network},
 volume = {15},
 year = {2020}
}

```


#### 70 multimodal deep unfolding for guided image super-resolution
**Publication Date**: 01/21/2020, 00:00:00
**Citation Count**: 26
**Abstract**: The reconstruction of a high resolution image given a low resolution observation is an ill-posed inverse problem in imaging. Deep learning methods rely on training data to learn an end-to-end mapping from a low-resolution input to a high-resolution output. Unlike existing deep multimodal models that do not incorporate domain knowledge about the problem, we propose a multimodal deep learning design that incorporates sparse priors and allows the effective integration of information from another image modality into the network architecture. Our solution relies on a novel deep unfolding operator, performing steps similar to an iterative algorithm for convolutional sparse coding with side information; therefore, the proposed neural network is interpretable by design. The deep unfolding architecture is used as a core component of a multimodal framework for guided image super-resolution. An alternative multimodal design is investigated by employing residual learning to improve the training efficiency. The presented multimodal approach is applied to super-resolution of near-infrared and multi-spectral images as well as depth upsampling using RGB images as side information. Experimental results show that our model outperforms state-of-the-art methods.
```bibtex
@Article{Marivani2020MultimodalDU,
 author = {Iman Marivani and Evaggelia Tsiligianni and Bruno Cornelis and N. Deligiannis},
 booktitle = {IEEE Transactions on Image Processing},
 journal = {IEEE Transactions on Image Processing},
 pages = {8443-8456},
 title = {Multimodal Deep Unfolding for Guided Image Super-Resolution},
 volume = {29},
 year = {2020}
}

```


#### 71 deep unfolding of the weighted mmse beamforming algorithm
**Publication Date**: 06/15/2020, 00:00:00
**Citation Count**: 22
**Abstract**: Downlink beamforming is a key technology for cellular networks. However, computing the transmit beamformer that maximizes the weighted sum rate subject to a power constraint is an NP-hard problem. As a result, iterative algorithms that converge to a local optimum are used in practice. Among them, the weighted minimum mean square error (WMMSE) algorithm has gained popularity, but its computational complexity and consequent latency has motivated the need for lower-complexity approximations at the expense of performance. Motivated by the recent success of deep unfolding in the trade-off between complexity and performance, we propose the novel application of deep unfolding to the WMMSE algorithm for a MISO downlink channel. The main idea consists of mapping a fixed number of iterations of the WMMSE algorithm into trainable neural network layers, whose architecture reflects the structure of the original algorithm. With respect to traditional end-to-end learning, deep unfolding naturally incorporates expert knowledge, with the benefits of immediate and well-grounded architecture selection, fewer trainable parameters, and better explainability. However, the formulation of the WMMSE algorithm, as described in Shi et al., is not amenable to be unfolded due to a matrix inversion, an eigendecomposition, and a bisection search performed at each iteration. Therefore, we present an alternative formulation that circumvents these operations by resorting to projected gradient descent. By means of simulations, we show that, in most of the settings, the unfolded WMMSE outperforms or performs equally to the WMMSE for a fixed number of iterations, with the advantage of a lower computational load.
```bibtex
@Article{Pellaco2020DeepUO,
 author = {Lissy Pellaco and M. Bengtsson and Joakim Jald'en},
 booktitle = {arXiv.org},
 journal = {ArXiv},
 title = {Deep unfolding of the weighted MMSE beamforming algorithm},
 volume = {abs/2006.08448},
 year = {2020}
}

```


#### 72 unsupervised resnet-inspired beamforming design using deep unfolding technique
**Publication Date**: 12/01/2020, 00:00:00
**Citation Count**: 13
**Abstract**: Beamforming is a key technology in communication systems of the fifth generation and beyond. However, traditional optimization-based algorithms are often computationally prohibited from performing in a real-time manner. On the other hand, the performance of existing deep learning (DL)-based algorithms can be further improved. As an alternative, we propose an unsupervised ResNet-inspired beamforming (RI-BF) algorithm in this paper that inherits the advantages of both pure optimization-based and DL-based beamforming for efficiency. In particular, a deep unfolding technique is introduced to reference the optimization process of the gradient ascent beamforming algorithm for the design of our neural network (NN) architecture. Moreover, the proposed RI-BF has three features. First, unlike the existing DL-based beamforming method, which employs a regularization term for the loss function or an output scaling mechanism to satisfy system power constraints, a novel NN architecture is introduced in RI-BF to generate initial beamforming with a promising performance. Second, inspired by the success of residual neural network (ResNet)-based DL models, a deep unfolding module is constructed to mimic the residual block of the ResNet-based model, further improving the performance of RI-BF based on the initial beamforming. Third, the entire RI-BF is trained in an unsupervised manner; as a result, labelling efforts are unnecessary. The simulation results demonstrate that the performance and computational complexity of our RI-BF improves significantly compared to the existing DL-based and optimization-based algorithms.
```bibtex
@Article{Lin2020UnsupervisedRB,
 author = {Chia-Hung Lin and Yen-Ting Lee and W. Chung and Shih-Chun Lin and Ta-Sung Lee},
 booktitle = {Global Communications Conference},
 journal = {GLOBECOM 2020 - 2020 IEEE Global Communications Conference},
 pages = {1-7},
 title = {Unsupervised ResNet-Inspired Beamforming Design Using Deep Unfolding Technique},
 year = {2020}
}

```


#### 73 an unsupervised deep unfolding framework for robust symbol-level precoding
**Publication Date**: 07/20/2021, 00:00:00
**Citation Count**: 5
**Abstract**: Symbol Level Precoding (SLP) has attracted significant research interest due to its ability to exploit interference for energy-efficient transmission. This paper proposes an unsupervised deep-neural network (DNN) based SLP framework. Instead of naively training a DNN architecture for SLP without considering the specifics of the optimization objective of the SLP domain, our proposal unfolds a power minimization SLP formulation based on the interior point method (IPM) proximal ‘log’ barrier function. Furthermore, we extend our proposal to a robust precoding design under channel state information (CSI) uncertainty. The results show that our proposed learning framework provides near-optimal performance while reducing the computational cost from <inline-formula> <tex-math notation="LaTeX">$\mathcal{O}(n^{7.5})$ </tex-math></inline-formula> to <inline-formula> <tex-math notation="LaTeX">$\mathcal{O}(n^{3})$ </tex-math></inline-formula> for the symmetrical system case where <inline-formula> <tex-math notation="LaTeX">$n=\text {number of transmit antennas}=\text {number of users}$ </tex-math></inline-formula>. This significant complexity reduction is also reflected in a proportional decrease in the proposed approach’s execution time compared to the SLP optimization-based solution.
```bibtex
@Article{Mohammad2021AnUD,
 author = {A. Mohammad and C. Masouros and Y. Andreopoulos},
 booktitle = {IEEE Open Journal of the Communications Society},
 journal = {IEEE Open Journal of the Communications Society},
 pages = {1075-1090},
 title = {An Unsupervised Deep Unfolding Framework for Robust Symbol-Level Precoding},
 volume = {4},
 year = {2021}
}

```


#### 74 deep unfolding of iteratively reweighted admm for wireless rf sensing
**Publication Date**: 06/07/2021, 00:00:00
**Citation Count**: 2
**Abstract**: We address the detection of material defects, which are inside a layered material structure using compressive sensing-based multiple-input and multiple-output (MIMO) wireless radar. Here, strong clutter due to the reflection of the layered structure’s surface often makes the detection of the defects challenging. Thus, sophisticated signal separation methods are required for improved defect detection. In many scenarios, the number of defects that we are interested in is limited, and the signaling response of the layered structure can be modeled as a low-rank structure. Therefore, we propose joint rank and sparsity minimization for defect detection. In particular, we propose a non-convex approach based on the iteratively reweighted nuclear and ℓ1-norm (a double-reweighted approach) to obtain a higher accuracy compared to the conventional nuclear norm and ℓ1-norm minimization. To this end, an iterative algorithm is designed to estimate the low-rank and sparse contributions. Further, we propose deep learning-based parameter tuning of the algorithm (i.e., algorithm unfolding) to improve the accuracy and the speed of convergence of the algorithm. Our numerical results show that the proposed approach outperforms the conventional approaches in terms of mean squared errors of the recovered low-rank and sparse components and the speed of convergence.
```bibtex
@Article{Thanthrige2021DeepUO,
 author = {U. M. Thanthrige and P. Jung and A. Sezgin},
 booktitle = {Italian National Conference on Sensors},
 journal = {Sensors (Basel, Switzerland)},
 title = {Deep Unfolding of Iteratively Reweighted ADMM for Wireless RF Sensing},
 volume = {22},
 year = {2021}
}

```


#### 75 temporal deep unfolding for constrained nonlinear stochastic optimal control
**Publication Date**: 11/03/2021, 00:00:00
**Citation Count**: 5
```bibtex
@Article{Kishida2021TemporalDU,
 author = {M. Kishida and Masaki Ogura},
 booktitle = {IET Control Theory & Applications},
 journal = {IET Control Theory & Applications},
 title = {Temporal deep unfolding for constrained nonlinear stochastic optimal control},
 year = {2021}
}

```


#### 76 deep unfolding of a proximal interior point method for image restoration
**Publication Date**: 12/11/2018, 00:00:00
**Citation Count**: 84
**Abstract**: Variational methods are widely applied to ill-posed inverse problems for they have the ability to embed prior knowledge about the solution. However, the level of performance of these methods significantly depends on a set of parameters, which can be estimated through computationally expensive and time-consuming methods. In contrast, deep learning offers very generic and efficient architectures, at the expense of explainability, since it is often used as a black-box, without any fine control over its output. Deep unfolding provides a convenient approach to combine variational-based and deep learning approaches. Starting from a variational formulation for image restoration, we develop iRestNet, a neural network architecture obtained by unfolding a proximal interior point algorithm. Hard constraints, encoding desirable properties for the restored image, are incorporated into the network thanks to a logarithmic barrier, while the barrier parameter, the stepsize, and the penalization weight are learned by the network. We derive explicit expressions for the gradient of the proximity operator for various choices of constraints, which allows training iRestNet with gradient descent and backpropagation. In addition, we provide theoretical results regarding the stability of the network for a common inverse problem example. Numerical experiments on image deblurring problems show that the proposed approach compares favorably with both state-of-the-art variational and machine learning methods in terms of image quality.
```bibtex
@Article{Bertocchi2018DeepUO,
 author = {Carla Bertocchi and É. Chouzenoux and M. Corbineau and J. Pesquet and M. Prato},
 booktitle = {Inverse Problems},
 journal = {Inverse Problems},
 title = {Deep unfolding of a proximal interior point method for image restoration},
 volume = {36},
 year = {2018}
}

```


#### 77 joint sparse recovery using deep unfolding with application to massive random access
**Publication Date**: 05/01/2020, 00:00:00
**Citation Count**: 10
**Abstract**: We propose a learning-based joint sparse recovery method for the multiple measurement vector (MMV) problem using deep unfolding. We unfold an iterative alternating direction method of multipliers (ADM) algorithm for MMV joint sparse recovery algorithm into a trainable deep network. This ADM algorithm is first obtained by modifying the squared error penalty function of an existing ADM algorithm to a back-projected squared error penalty function. Numerical results for a massive random access system show that our proposed modification to the MMV-ADM method and deep unfolding provide significant improvement in convergence and estimation performance.
```bibtex
@Article{Sabulal2020JointSR,
 author = {Anand P. Sabulal and S. Bhashyam},
 booktitle = {IEEE International Conference on Acoustics, Speech, and Signal Processing},
 journal = {ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
 pages = {5050-5054},
 title = {Joint Sparse Recovery Using Deep Unfolding With Application to Massive Random Access},
 year = {2020}
}

```


#### 78 adaptive equalization of transmitter and receiver iq skew by multi-layer linear and widely linear filters with deep unfolding.
**Publication Date**: 08/03/2020, 00:00:00
**Citation Count**: 7
**Abstract**: We propose a multi-layer cascaded filter architecture consisting of differently sized strictly linear (SL) and widely linear (WL) filters to compensate for the relevant linear impairments in optical fiber communications including in-phase/quadrature (IQ) skew in both transmitter and receiver by using deep unfolding. To control the filter coefficients adaptively, we adopt a gradient calculation with back propagation from machine learning with neural networks to minimize the magnitude of deviation of the filter outputs of the last layer from the desired state in a stochastic gradient descent (SGD) manner. We derive a filter coefficient update algorithm for multi-layer SL and WL multi-input multi-output finite-impulse response filters. The results of a transmission experiment on 32-Gbaud polarization-division multiplexed 64-quadrature amplitude modulation over a 100-km single-mode fiber span showed that the proposed multi-layer SL and WL filters with SGD control could compensate for IQ skew in both transmitter and receiver under the accumulation of chromatic dispersion, polarization rotation, and frequency offset of a local oscillator laser source.
```bibtex
@Article{Arikawa2020AdaptiveEO,
 author = {M. Arikawa and K. Hayashi},
 booktitle = {Optics Express},
 journal = {Optics express},
 pages = {
          23478-23494
        },
 title = {Adaptive equalization of transmitter and receiver IQ skew by multi-layer linear and widely linear filters with deep unfolding.},
 volume = {28 16},
 year = {2020}
}

```


#### 79 deep unfolding approach for signal processing algorithms: convergence acceleration and its theoretical interpretation
**Publication Date**: 07/01/2020, 00:00:00
**Citation Count**: 3
**Abstract**: Deep learning techniques can be used for not only learning deep neural networks but also internal parameter optimization of diﬀerentiable iterative algorithms. By embedding learnable parameters into an excellent iterative algorithm, we can construct a ﬂexible derived algorithm with data-driven learnability. This approach is called deep unfolding. We present an overview of deep unfolding and its features, focusing on sparse signal recovery algorithms. In the ﬁrst half of this paper, examples of deep unfolding including a sparse signal recovery algorithm, TISTA, will be presented. We observed the convergence acceleration phenomenon for deep unfolding-based algorithms. In the second half of the paper, our theoretical results (spectral radius control based on the Chebyshev step) for convergence acceleration are outlined.
```bibtex
@Article{Wadayama2020DeepUA,
 author = {T. Wadayama and Satoshi Takabe},
 booktitle = {IEICE ESS FUNDAMENTALS REVIEW},
 journal = {IEICE ESS Fundamentals Review},
 title = {Deep Unfolding Approach for Signal Processing Algorithms: Convergence Acceleration and Its Theoretical Interpretation},
 year = {2020}
}

```


#### 80 deep unfolding-aided gaussian belief propagation for correlated large mimo detection
**Publication Date**: 12/01/2020, 00:00:00
**Citation Count**: 4
**Abstract**: This paper proposes a deep unfolding-aided belief propagation (BP) for large multi-user multi-input multi-output (MU-MIMO) detection under correlated fading channels. A BP-based detector is a well-known strategy for realizing large-scale MU detection (MUD) with low-complexity and high-accuracy. However, its convergence property is severely degraded under insufficient large-system conditions and spatial fading correlation among RX antenna elements. To compensate for this drawback, we design a trainable Gaussian BP (T-GaBP) having well-organized trainable internal parameters based on the BP structure. These parameters are optimized by the deep learning techniques in the signal-flow graph of unfolded GaBP; this approach is referred to as data-driven tuning. By training the parameters according to the system model, T-GaBP can maintain the high detection capability even in practical system configurations that differ from the ideal uncorrelated massive MIMO assumption. Numerical results show that the proposed detector improves the convergence property and achieves a comparable detection performance to the cutting-edge expectation propagation (EP) detector in correlated MUD, with a lower computational cost.
```bibtex
@Article{Shirase2020DeepUG,
 author = {Daichi Shirase and Takumi Takahashi and S. Ibi and K. Muraoka and N. Ishii and S. Sampei},
 booktitle = {Global Communications Conference},
 journal = {GLOBECOM 2020 - 2020 IEEE Global Communications Conference},
 pages = {1-6},
 title = {Deep Unfolding-Aided Gaussian Belief Propagation for Correlated Large MIMO Detection},
 year = {2020}
}

```


#### 81 federated deep unfolding for sparse recovery
**Publication Date**: 10/23/2020, 00:00:00
**Citation Count**: 3
**Abstract**: This paper proposes a federated learning technique for deep algorithm unfolding with applications to sparse signal recovery and compressed sensing. We refer to this architecture as Fed-CS. Specifically, we unfold and learn the iterative shrinkage thresholding algorithm for sparse signal recovery without transporting the training data distributed across many clients to a central location. We propose a layer-wise federated learning technique, in which each client uses local data to train a common model. Then we transmit only the model parameters of that layer from all the clients to the server, which aggregates these local models to arrive at a consensus model. The proposed layer-wise federated learning for sparse recovery is communication efficient and preserves data privacy. Through numerical experiments on synthetic and real datasets, we demonstrate Fed-CS's efficacy and present various trade-offs in terms of the number of participating clients and communications involved compared to a centralized approach of deep unfolding.
```bibtex
@Article{Mogilipalepu2020FederatedDU,
 author = {Komal Krishna Mogilipalepu and Sumanth Kumar Modukuri and Amarlingam Madapu and S. P. Chepuri},
 booktitle = {European Signal Processing Conference},
 journal = {2021 29th European Signal Processing Conference (EUSIPCO)},
 pages = {1950-1954},
 title = {Federated Deep Unfolding for Sparse Recovery},
 year = {2020}
}

```


#### 82 memory-augmented deep conditional unfolding network for pansharpening
**Publication Date**: 06/01/2022, 00:00:00
**Citation Count**: 12
**Abstract**: Pansharpening aims to obtain high-resolution multispectral (MS) images for remote sensing systems and deep learning-based methods have achieved remarkable success. However, most existing methods are designed in a black-box principle, lacking sufficient interpretability. Additionally, they ignore the different characteristics of each band of MS images and directly concatenate them with panchromatic (PAN) images, leading to severe copy artifacts [9]. To address the above issues, we propose an interpretable deep neural network, namely Memory-augmented Deep Conditional Unfolding Network with two specified core designs. Firstly, considering the degradation process, it formulates the Pansharpening problem as the minimization of a variational model with denoising-based prior and non-local auto-regression prior which is capable of searching the similarities between long-range patches, benefiting the texture enhancement. A novel iteration algorithm with built-in CNNs is exploited for transparent model design. Secondly, to fully explore the potentials of different bands of MS images, the PAN image is combined with each band of MS images, selectively providing the high-frequency details and alleviating the copy artifacts. Extensive experimental results validate the superiority of the proposed algorithm against other state-of-the-art methods.
```bibtex
@Article{Yang2022MemoryaugmentedDC,
 author = {Gang Yang and Man Zhou and Keyu Yan and Aiping Liu and Xueyang Fu and Fan Wang},
 booktitle = {Computer Vision and Pattern Recognition},
 journal = {2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
 pages = {1778-1787},
 title = {Memory-augmented Deep Conditional Unfolding Network for Pansharpening},
 year = {2022}
}

```


#### 83 online unsupervised deep unfolding for mimo channel estimation
**Publication Date**: 04/30/2020, 00:00:00
**Citation Count**: 3
**Abstract**: —Channel estimation is a difﬁcult problem in MIMO systems. Using a physical model allows to ease the problem, injecting a priori information based on the physics of propa- gation. However, such models rest on simplifying assumptions and require to know precisely the system conﬁguration, which is unrealistic. In this paper, we propose to perform online learning for channel estimation in a massive MIMO context, adding ﬂexibility to physical models by unfolding a channel estimation algorithm (matching pursuit) as a neural network. This leads to a computationally efﬁcient neural network that can be trained online when initialized with an imperfect model. The method allows a base station to automatically correct its channel estimation algorithm based on incoming data, without the need for a separate ofﬂine training phase. It is applied to realistic channels and shows great performance, achieving channel estimation error almost as low as one would get with a perfectly calibrated system.
```bibtex
@Inproceedings{Magoarou2020OnlineUD,
 author = {Luc Le Magoarou and S. Paquelet},
 title = {Online unsupervised deep unfolding for MIMO channel estimation},
 year = {2020}
}

```


#### 84 online unsupervised deep unfolding for massive mimo channel estimation
**Publication Date**: 04/29/2020, 00:00:00
**Citation Count**: 3
**Abstract**: Massive MIMO communication systems have a huge potential both in terms of data rate and energy efficiency, although channel estimation becomes challenging for a large number antennas. Using a physical model allows to ease the problem by injecting a priori information based on the physics of propagation. However, such a model rests on simplifying assumptions and requires to know precisely the configuration of the system, which is unrealistic in practice. In this letter, we propose to perform online learning for channel estimation in a massive MIMO context, adding flexibility to physical channel models by unfolding a channel estimation algorithm (matching pursuit) as a neural network. This leads to a computationally efficient neural network structure that can be trained online when initialized with an imperfect model. The method allows a base station to automatically correct its channel estimation algorithm based on incoming data, without the need for a separate offline training phase. It is applied to realistic millimeter wave channels and shows great performance, achieving a channel estimation error almost as low as one would get with a perfectly calibrated system.
```bibtex
@Article{Magoarou2020OnlineUD,
 author = {Luc Le Magoarou and S. Paquelet},
 journal = {arXiv: Signal Processing},
 title = {Online unsupervised deep unfolding for massive MIMO channel estimation},
 year = {2020}
}

```


#### 85 deep unfolding for topic models
**Publication Date**: 02/01/2018, 00:00:00
**Citation Count**: 37
**Abstract**: Deep unfolding provides an approach to integrate the probabilistic generative models and the deterministic neural networks. Such an approach is benefited by deep representation, easy interpretation, flexible learning and stochastic modeling. This study develops the unsupervised and supervised learning of deep unfolded topic models for document representation and classification. Conventionally, the unsupervised and supervised topic models are inferred via the variational inference algorithm where the model parameters are estimated by maximizing the lower bound of logarithm of marginal likelihood using input documents without and with class labels, respectively. The representation capability or classification accuracy is constrained by the variational lower bound and the tied model parameters across inference procedure. This paper aims to relax these constraints by directly maximizing the end performance criterion and continuously untying the parameters in learning process via deep unfolding inference (DUI). The inference procedure is treated as the layer-wise learning in a deep neural network. The end performance is iteratively improved by using the estimated topic parameters according to the exponentiated updates. Deep learning of topic models is therefore implemented through a back-propagation procedure. Experimental results show the merits of DUI with increasing number of layers compared with variational inference in unsupervised as well as supervised topic models.
```bibtex
@Article{Chien2018DeepUF,
 author = {Jen-Tzung Chien and Chao-Hsi Lee},
 booktitle = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
 journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
 pages = {318-331},
 title = {Deep Unfolding for Topic Models},
 volume = {40},
 year = {2018}
}

```


#### 86 multimodal image super-resolution via deep unfolding with side information
**Publication Date**: 09/01/2019, 00:00:00
**Citation Count**: 14
**Abstract**: Deep learning methods have been successfully applied to various computer vision tasks. However, existing neural network architectures do not per se incorporate domain knowledge about the addressed problem, thus, understanding what the model has learned is an open research topic. In this paper, we rely on the unfolding of an iterative algorithm for sparse approximation with side information, and design a deep learning architecture for multimodal image super-resolution that incorporates sparse priors and effectively utilizes information from another image modality. We develop two deep models performing reconstruction of a high-resolution image of a target image modality from its low-resolution variant with the aid of a high-resolution image from a second modality. We apply the proposed models to super-resolve near-infrared images using as side information high-resolution RGB images. Experimental results demonstrate the superior performance of the proposed models against state-of-the-art methods including unimodal and multimodal approaches.
```bibtex
@Article{Marivani2019MultimodalIS,
 author = {Iman Marivani and Evaggelia Tsiligianni and Bruno Cornelis and N. Deligiannis},
 booktitle = {European Signal Processing Conference},
 journal = {2019 27th European Signal Processing Conference (EUSIPCO)},
 pages = {1-5},
 title = {Multimodal Image Super-resolution via Deep Unfolding with Side Information},
 year = {2019}
}

```


#### 87 patch-aware deep hyperspectral and multispectral image fusion by unfolding subspace-based optimization model
**Citation Count**: 8
**Abstract**: Hyperspectral and multispectral image fusion aims to fuse a low-spatial-resolution hyperspectral image (HSI) and a high-spatial-resolution multispectral image to form a high-spatial-resolution HSI. Motivated by the success of model- and deep learning-based approaches, we propose a novel patch-aware deep fusion approach for HSI by unfolding a subspace-based optimization model, where moderate-sized patches are used in both training and test phases. The goal of this approach is to make full use of the information of patch under subspace representation, restrict the scale and enhance the interpretability of the deep network, thereby improving the fusion. First, a subspace-based fusion model was built with two regularization terms to localize pixels and extract texture. Then, the subspace-based fusion model was solved by the alternating direction method of multipliers algorithm, and the model was divided into one fidelity-based problem and two regularization-based problems. Finally, a structured deep fusion network was proposed by unfolding all steps of the algorithm as network layers. Specifically, the fidelity-based problem was solved by a gradient descent algorithm and implemented by a network. The two regularization-based problems were described by proximal operators and learnt by two u-shaped architectures. Moreover, an aggregation fusion technique was proposed to improve the performance by averaging the fused images in all iterations and aggregating the overlapping patches in the test phase. Experimental results, conducted on both synthetic and real datasets, demonstrated the effectiveness of the proposed approach.
```bibtex
@Article{Liu2022PatchAwareDH,
 author = {Jianjun Liu and Dunbin Shen and Zebin Wu and Liang Xiao and Jun Sun and Hong Yan},
 booktitle = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
 journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
 pages = {1024-1038},
 title = {Patch-Aware Deep Hyperspectral and Multispectral Image Fusion by Unfolding Subspace-Based Optimization Model},
 volume = {15},
 year = {2022}
}

```


#### 88 belief propagation with deep unfolding for high-dimensional inference in communication systems
**Citation Count**: 1
**Abstract**: Belief Propagation with Deep Unfolding for High-dimensional Inference in Communication Systems by Mengke Lian Department of Electrical and Computer Engineering Duke University
```bibtex
@Inproceedings{Lian2019BeliefPW,
 author = {Mengke Lian},
 title = {Belief Propagation with Deep Unfolding for High-dimensional Inference in Communication Systems},
 year = {2019}
}

```


#### 89 doubly iterative turbo equalization: optimization through deep unfolding
**Publication Date**: 09/01/2019, 00:00:00
**Citation Count**: 10
**Abstract**: This paper analyzes some emerging techniques from the broad area of Bayesian learning for the design of iterative receivers for single-carrier transmissions using bit-interleaved coded-modulation (BICM) in wideband channels. In particular, approximate Bayesian inference methods, such as expectation propagation (EP), and iterative signal-recovery methods, such as approximate message passing (AMP) algorithms are evaluated as frequency domain equalizers (FDE). These algorithms show that decoding performance can be improved by going beyond the established turbo-detection principles, by iterating over inner detection loops before decoding. A comparative analysis is performed for the case of quasistatic wideband communications channels, showing that the EP-based approach is more advantageous. Moreover, recent advances in structured learning are revisited for the iterative EP-based receiver by unfolding the inner detection loop, and obtaining a deep detection network with learnable parameters. To this end, a novel, mutual-information dependent learning cost function is proposed, suited to turbo detectors, and through learning, the detection performance of the deep EP network is optimized.
```bibtex
@Article{Şahin2019DoublyIT,
 author = {Serdar Şahin and C. Poulliat and A. Cipriano and M. Boucheret},
 booktitle = {IEEE International Symposium on Personal, Indoor and Mobile Radio Communications},
 journal = {2019 IEEE 30th Annual International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC)},
 pages = {1-6},
 title = {Doubly Iterative Turbo Equalization: Optimization through Deep Unfolding},
 year = {2019}
}

```


#### 90 deep unfolding: model-based inspiration of novel deep architectures
**Publication Date**: 09/08/2014, 00:00:00
**Citation Count**: 361
**Abstract**: Model-based methods and deep neural networks have both been tremendously successful paradigms in machine learning. In model-based methods, problem domain knowledge can be built into the constraints of the model, typically at the expense of difficulties during inference. In contrast, deterministic deep neural networks are constructed in such a way that inference is straightforward, but their architectures are generic and it is unclear how to incorporate knowledge. This work aims to obtain the advantages of both approaches. To do so, we start with a model-based approach and an associated inference algorithm, and \emph{unfold} the inference iterations as layers in a deep network. Rather than optimizing the original model, we \emph{untie} the model parameters across layers, in order to create a more powerful network. The resulting architecture can be trained discriminatively to perform accurate inference within a fixed network size. We show how this framework allows us to interpret conventional networks as mean-field inference in Markov random fields, and to obtain new architectures by instead using belief propagation as the inference algorithm. We then show its application to a non-negative matrix factorization model that incorporates the problem-domain knowledge that sound sources are additive. Deep unfolding of this model yields a new kind of non-negative deep neural network, that can be trained using a multiplicative backpropagation-style update algorithm. We present speech enhancement experiments showing that our approach is competitive with conventional neural networks despite using far fewer parameters.
```bibtex
@Article{Hershey2014DeepUM,
 author = {J. Hershey and Jonathan Le Roux and F. Weninger},
 booktitle = {arXiv.org},
 journal = {ArXiv},
 title = {Deep Unfolding: Model-Based Inspiration of Novel Deep Architectures},
 volume = {abs/1409.2574},
 year = {2014}
}

```


#### 91 dynamic path-controllable deep unfolding network for compressive sensing
**Publication Date**: 04/10/2023, 00:00:00
**Citation Count**: 6
**Abstract**: Deep unfolding network (DUN) that unfolds the optimization algorithm into a deep neural network has achieved great success in compressive sensing (CS) due to its good interpretability and high performance. Each stage in DUN corresponds to one iteration in optimization. At the test time, all the sampling images generally need to be processed by all stages, which comes at a price of computation burden and is also unnecessary for the images whose contents are easier to restore. In this paper, we focus on CS reconstruction and propose a novel Dynamic Path-Controllable Deep Unfolding Network (DPC-DUN). DPC-DUN with our designed path-controllable selector can dynamically select a rapid and appropriate route for each image and is slimmable by regulating different performance-complexity tradeoffs. Extensive experiments show that our DPC-DUN is highly flexible and can provide excellent performance and dynamic adjustment to get a suitable tradeoff, thus addressing the main requirements to become appealing in practice. Codes are available at https://github.com/songjiechong/DPC-DUN.
```bibtex
@Article{Song2023DynamicPD,
 author = {Jie Song and Bin Chen and Jian Zhang},
 booktitle = {IEEE Transactions on Image Processing},
 journal = {IEEE Transactions on Image Processing},
 pages = {2202-2214},
 title = {Dynamic Path-Controllable Deep Unfolding Network for Compressive Sensing},
 volume = {32},
 year = {2023}
}

```


#### 92 deep unfolding hybrid beamforming designs for thz massive mimo systems
**Publication Date**: 02/23/2023, 00:00:00
**Citation Count**: 4
**Abstract**: Hybrid beamforming (HBF) is a key enabler for wideband terahertz (THz) massive multiple-input multiple-output (mMIMO) communications systems. A core challenge with designing HBF systems stems from the fact their application often involves a non-convex, highly complex optimization of large dimensions. In this article, we propose HBF schemes that leverage data to enable efficient designs for both the fully-connected HBF (FC-HBF) and dynamic sub-connected HBF (SC-HBF) architectures. We develop a deep unfolding framework based on factorizing the optimal fully digital beamformer into analog and digital terms and formulating two corresponding equivalent least squares (LS) problems. Then, the digital beamformer is obtained via a closed-form LS solution, while the analog beamformer is obtained via ManNet, a lightweight sparsely-connected deep neural network based on unfolding projected gradient descent. Incorporating ManNet into the developed deep unfolding framework leads to the ManNet-based FC-HBF scheme. We show that the proposed ManNet can also be applied to SC-HBF designs after determining the connections between the radio frequency chain and antennas. We further develop a simplified version of ManNet, referred to as subManNet, that directly produces the sparse analog precoder for SC-HBF architectures. Both networks are trained with an unsupervised procedure. Numerical results verify that the proposed ManNet/subManNet-based HBF approaches outperform the conventional model-based and deep unfolded counterparts with very low complexity and a fast run time. For example, in a simulation with 128 transmit antennas, ManNet attains a slightly higher spectral efficiency than the Riemannian manifold scheme, but over 600 times faster and with a complexity reduction of more than by a factor of six (6).
```bibtex
@Article{Nguyen2023DeepUH,
 author = {N. Nguyen and Mengyuan Ma and Nir Shlezinger and Y. Eldar and A. L. Swindlehurst and M. Juntti},
 booktitle = {IEEE Transactions on Signal Processing},
 journal = {IEEE Transactions on Signal Processing},
 pages = {3788-3804},
 title = {Deep Unfolding Hybrid Beamforming Designs for THz Massive MIMO Systems},
 volume = {71},
 year = {2023}
}

```


#### 93 deep unfolding multi-scale regularizer network for image denoising
**Publication Date**: 01/03/2023, 00:00:00
**Citation Count**: 3
```bibtex
@Article{Xu2023DeepUM,
 author = {Jingzhao Xu and Mengke Yuan and Dong Yan and Tieru Wu},
 booktitle = {Computational Visual Media},
 journal = {Computational Visual Media},
 pages = {335-350},
 title = {Deep unfolding multi-scale regularizer network for image denoising},
 volume = {9},
 year = {2023}
}

```


#### 94 distributed admm with limited communications via deep unfolding
**Publication Date**: 06/04/2023, 00:00:00
**Citation Count**: 2
**Abstract**: Distributed optimization arises in various applications. A widely-used distributed optimizer is the distributed alternating direction method of multipliers (D-ADMM) algorithm, which enables agents to jointly minimize a shared objective by iteratively combining local computations and message exchanges. However, D-ADMM often involves a large number of possibly costly communications to reach convergence, limiting its applicability in communications-constrained networks. In this work we propose unfolded D-ADMM, which facilitates the application of D-ADMM with limited communications using the emerging deep unfolding methodology. We utilize the conventional D-ADMM algorithm with a fixed number of communications rounds, while leveraging data to tune the hyperparameters of each iteration of the algorithm. By doing so, we learn to optimize with limited communications, while preserving the interpretability and flexibility of the original D-ADMM algorithm. Our numerical results demonstrate that the proposed approach dramatically reduces the number of communications utilized by D-ADMM, without compromising on its performance.
```bibtex
@Conference{Noah2023DistributedAW,
 author = {Yoav Noah and Nir Shlezinger},
 booktitle = {IEEE International Conference on Acoustics, Speech, and Signal Processing},
 journal = {ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
 pages = {1-5},
 title = {Distributed Admm with Limited Communications Via Deep Unfolding},
 year = {2023}
}

```


#### 95 joint deep reinforcement learning and unfolding: beam selection and precoding for mmwave multiuser mimo with lens arrays
**Publication Date**: 01/05/2021, 00:00:00
**Citation Count**: 29
**Abstract**: The millimeter wave (mmWave) multiuser multiple-input multiple-output (MU-MIMO) systems with discrete lens arrays (DLA) have received great attention due to their simple hardware implementation and excellent performance. In this work, we investigate the joint design of beam selection and digital precoding matrices for mmWave MU-MIMO systems with DLA to maximize the sum-rate subject to the transmit power constraint and the constraints of the selection matrix structure. The investigated non-convex problem with discrete variables and coupled constraints is challenging to solve and an efficient framework of joint neural network (NN) design is proposed to tackle it. Specifically, the proposed framework consists of a deep reinforcement learning (DRL)-based NN and a deep-unfolding NN, which are employed to optimize the beam selection and digital precoding matrices, respectively. As for the DRL-based NN, we formulate the beam selection problem as a Markov decision process and a double deep Q-network algorithm is developed to solve it. The base station is considered to be an agent, where the state, action, and reward function are carefully designed. Regarding the design of the digital precoding matrix, we develop an iterative weighted minimum mean-square error algorithm induced deep-unfolding NN, which unfolds this algorithm into a layer-wise structure with introduced trainable parameters. Simulation results verify that this jointly trained NN remarkably outperforms the existing iterative algorithms with reduced complexity and stronger robustness.
```bibtex
@Article{Hu2021JointDR,
 author = {Qiyu Hu and Yanzhen Liu and Yunlong Cai and Guanding Yu and Z. Ding},
 booktitle = {IEEE Journal on Selected Areas in Communications},
 journal = {IEEE Journal on Selected Areas in Communications},
 pages = {2289-2304},
 title = {Joint Deep Reinforcement Learning and Unfolding: Beam Selection and Precoding for mmWave Multiuser MIMO With Lens Arrays},
 volume = {39},
 year = {2021}
}

```


#### 96 deep-unfolding-based bit-level csi feedback in massive mimo systems
**Publication Date**: 02/01/2023, 00:00:00
**Citation Count**: 1
**Abstract**: In frequency division duplex mode, downlink channel state information (CSI) should be sent to the base station via a feedback link to obtain the benefits of the massive multiple-input multiple-output technology, leading to a substantial overhead. The existing deep-unfolding-based works combine the advantages of high performance and interpretability. In practical systems, the compressed CSI must be quantized at the user equipment before feedback. In this letter, a deep-unfolding-based bit-level CSI feedback is proposed to reduce the effects of quantization errors by using the shortcut and approximate quantization schemes. The shortcut scheme provides the unquantized encoder output directly to the decoder through an additional network branch, which can effectively help the decoder training. The approximate quantization scheme allows the gradient to be propagated correctly by approximating the real quantization operation. The two schemes are combined as a training strategy to reduce quantization errors further. Results show that the combined scheme can substantially reduce the influence of the quantization layer and improve the reconstruction performance while having some generality.
```bibtex
@Article{Cao2023DeepUnfoldingBasedBC,
 author = {Zheng Cao and Jiajia Guo and Chao-Kai Wen and Shimei Jin},
 booktitle = {IEEE Wireless Communications Letters},
 journal = {IEEE Wireless Communications Letters},
 pages = {371-375},
 title = {Deep-Unfolding-Based Bit-Level CSI Feedback in Massive MIMO Systems},
 volume = {12},
 year = {2023}
}

```


#### 97 deep unfolding network for efficient mixed video noise removal
**Publication Date**: 09/01/2023, 00:00:00
**Citation Count**: 1
**Abstract**: Existing image and video denoising algorithms have focused on removing homogeneous Gaussian noise. However, this assumption with noise modeling is often too simplistic for the characteristics of real-world noise. Moreover, the design of network architectures in most deep learning-based video denoising methods is heuristic, ignoring valuable domain knowledge. In this paper, we propose a model-guided deep unfolding network for the more challenging and realistic mixed noise video denoising problem, named DU-MVDnet. First, we develop a novel observation model/likelihood function based on the correlations among adjacent degraded frames. In the framework of Bayesian deep learning, we introduce a deep image denoiser prior and obtain an iterative optimization algorithm based on the maximum a posterior (MAP) estimation. To facilitate end-to-end optimization, the iterative algorithm is transformed into a deep convolutional neural network (DCNN)-based implementation. Furthermore, recognizing the limitations of traditional motion estimation and compensation methods, we propose an efficient multistage recursive fusion strategy to exploit temporal dependencies. Specifically, we divide video frames into several overlapping groups and progressively integrate these frames into one frame. Toward this objective, we implement a multiframe adaptive aggregation operation to integrate feature maps of intragroup with those of intergroup frames. Extensive experimental results on different video test datasets have demonstrated that the proposed model-guided deep network outperforms current state-of-the-art video denoising algorithms such as FastDVDnet and MAP-VDNet.
```bibtex
@Article{Sun2023DeepUN,
 author = {Lu Sun and Yichen Wang and Fangfang Wu and Xin Li and W. Dong and Guangming Shi},
 booktitle = {IEEE transactions on circuits and systems for video technology (Print)},
 journal = {IEEE Transactions on Circuits and Systems for Video Technology},
 pages = {4715-4727},
 title = {Deep Unfolding Network for Efficient Mixed Video Noise Removal},
 volume = {33},
 year = {2023}
}

```


#### 98 a fast deep unfolding learning framework for robust mu-mimo downlink precoding
**Publication Date**: 04/01/2023, 00:00:00
**Citation Count**: 1
**Abstract**: This paper reformulates a worst-case sum-rate maximization problem for optimizing robust multi-user multiple-input multiple-output (MU-MIMO) downlink precoding under realistic per-antenna power constraints. We map the fixed number of iterations in the developed mean-square-error uplink-downlink duality iterative algorithm into a layer-wise trainable network to solve it. In contrast to black-box approximation neural networks, this proposed unfolding network has higher explanatory power due to fusing domain knowledge from existing iterative optimization approaches into deep learning architecture. Moreover, it could provide faster robust beamforming decisions by using several trainable key parameters. We optimize the determination of the channel error’s spectral norm constraint to improve the sum rate performance. The experimental results verify that the proposed deep unfolding “RMSED-Net” could combat channel errors in comparison with the non-robust baseline. It is also confirmed by the simulations that the proposed RMSED-Net in a fixed network depth could substantially reduce the computing time of the conventional iterative optimization method at the cost of a mild sum rate performance loss.
```bibtex
@Article{Xu2023AFD,
 author = {Jing Xu and Chaohui Kang and J. Xue and Yizhai Zhang},
 booktitle = {IEEE Transactions on Cognitive Communications and Networking},
 journal = {IEEE Transactions on Cognitive Communications and Networking},
 pages = {359-372},
 title = {A Fast Deep Unfolding Learning Framework for Robust MU-MIMO Downlink Precoding},
 volume = {9},
 year = {2023}
}

```


#### 99 designing transformer networks for sparse recovery of sequential data using deep unfolding
**Publication Date**: 06/04/2023, 00:00:00
**Citation Count**: 1
**Abstract**: Deep unfolding models are designed by unrolling an optimization algorithm into a deep learning network. These models have shown faster convergence and higher performance compared to the original optimization algorithms. Additionally, by incorporating domain knowledge from the optimization algorithm, they need much less training data to learn efficient representations. Current deep unfolding networks for sequential sparse recovery consist of recurrent neural networks (RNNs), which leverage the similarity between consecutive signals. We redesign the optimization problem to use correlations across the whole sequence, which unfolds into a Transformer architecture. Our model is used for the task of video frame reconstruction from low-dimensional measurements and is shown to outperform state-of-the-art deep unfolding RNN and Transformer models, as well as a traditional Vision Transformer on several video datasets.
```bibtex
@Conference{Weerdt2023DesigningTN,
 author = {B. D. Weerdt and Yonina C. Eldar and Nikos Deligiannis},
 booktitle = {IEEE International Conference on Acoustics, Speech, and Signal Processing},
 journal = {ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
 pages = {1-5},
 title = {Designing Transformer Networks for Sparse Recovery of Sequential Data Using Deep Unfolding},
 year = {2023}
}

```


#### 100 deep unfolding-enabled hybrid beamforming design for mmwave massive mimo systems
**Publication Date**: 06/04/2023, 00:00:00
**Citation Count**: 1
**Abstract**: Hybrid beamforming (HBF) is a key enabler for millimeter-wave (mmWave) communications systems, but HBF optimizations are often non-convex and of large dimension. In this paper, we propose an efficient deep unfolding-based HBF scheme, referred to as ManNet-HBF, that approximately maximizes the system spectral efficiency (SE). It first factorizes the optimal digital beamformer into analog and digital terms, and then reformulates the resultant matrix factorization problem as an equivalent maximum-likelihood problem, whose analog beamforming solution is vectorized and estimated efficiently with ManNet, a lightweight deep neural network. Numerical results verify that the proposed ManNet-HBF approach has near-optimal performance comparable to or better than conventional model-based counterparts, with very low complexity and a fast run time. For example, in a simulation with 128 transmit antennas, it attains 98.62% the SE of the Riemannian manifold scheme but 13250 times faster.
```bibtex
@Conference{Nguyen2023DeepUH,
 author = {N. Nguyen and Mengyuan Ma and Nir Shlezinger and Y. Eldar and A. L. Swindlehurst and M. Juntti},
 booktitle = {IEEE International Conference on Acoustics, Speech, and Signal Processing},
 journal = {ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
 pages = {1-5},
 title = {Deep Unfolding-Enabled Hybrid Beamforming Design for mmWave Massive MIMO Systems},
 year = {2023}
}

```


